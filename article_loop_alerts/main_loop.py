import time
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.errors import InvalidTextRepresentation
from pull_article import *
import json
from alive_progress import alive_bar
from lda_funcs import *
import threading
from datetime import datetime, timedelta, timezone
import os
from concurrent.futures import ThreadPoolExecutor
import queue
import sys
import signal
from tempfile import gettempdir
import os.path
from core.alerts.alerts_logger import AlertLogger

FETCH_LOCK_FILE = os.path.join(gettempdir(), "fetch_lock")
PROCESS_LOCK_FILE = os.path.join(gettempdir(), "process_lock")
PROCESS_INTERVAL = 3600  #1hr
FETCH_INTERVAL = 14400  # 4 hours
logger = AlertLogger('article-loop-main')
running = True

def signal_handler(signum, frame):
    """Handle shutdown"""
    global running
    print("\nSHUTDOWN")
    running = False

def process_article_batch(articles_chunk, prompt, thread_id):
    """Processes articles in a thread"""
    thread_name = f"Worker-{thread_id}"
    threading.current_thread().name = thread_name
    
    with psycopg2.connect(os.environ['CONN_STRING_ARTICLES']) as thread_cursor_conn:
        with thread_cursor_conn.cursor() as thread_cursor:
            with psycopg2.connect(os.environ['CONN_STRING_BACKEND']) as countries_conn:
                with countries_conn.cursor() as countries_cursor:
                    try:
                        print(f"{thread_name}: processing {len(articles_chunk)} articles")
                        articles_df = pd.DataFrame(articles_chunk, columns=['url', 'title', 'language', 'sourcecountry', 'category', 'code'])
                        
                        processed_articles = extract_translate(
                            articles_df['category'].iloc[0], 
                            prompt, 
                            articles_df, 
                            countries_cursor
                        )
                        
                        if not processed_articles.empty:
                            successful_urls = processed_articles[processed_articles['relevance'] == True]['url'].tolist()
                            if successful_urls:
                                insert_relevant_articles(processed_articles, thread_cursor)
                                placeholders = ','.join(['%s'] * len(successful_urls))
                                update_query = f"""
                                    UPDATE translated_articles 
                                    SET thread_status = 'processed' 
                                    WHERE url IN ({placeholders})
                                    AND thread_status = 'processing';
                                """
                                thread_cursor.execute(update_query, successful_urls)
                                thread_cursor_conn.commit()
                                print(f"{thread_name}: completed {len(successful_urls)} articles")
                    except Exception as e:
                        print(f"{thread_name}: error - {e}")
@logger.log_execution()
def process_pending_articles(cursor, num_threads=45, max_articles_per_thread=100):
    """Processes pending articles in parallel threads"""
    
    if not running:
        return  

    count_query = """
    SELECT COUNT(*) 
    FROM translated_articles 
    WHERE thread_status = 'pending'
    AND utc_datetime >= NOW() - INTERVAL '2 days';
    """
    cursor.execute(count_query)
    total_pending = cursor.fetchone()[0]

    if total_pending == 0 or not running:
        print("No recent pending articles")
        return

    batch_size = min(max(1, total_pending // num_threads), max_articles_per_thread)
    adjusted_threads = min(num_threads, total_pending)

    fetch_query = """
    SELECT url, title, language, sourcecountry, category, code 
    FROM translated_articles 
    WHERE thread_status = 'pending'
    AND utc_datetime >= NOW() - INTERVAL '2 days'
    ORDER BY RANDOM()
    LIMIT %s
    FOR UPDATE SKIP LOCKED;
    """
    cursor.execute(fetch_query, (batch_size * adjusted_threads,))
    articles = cursor.fetchall()

    if not articles or not running:
        return

    urls = [article[0] for article in articles]
    update_query = """
    UPDATE translated_articles 
    SET thread_status = 'processing' 
    WHERE url = ANY(%s);
    """
    cursor.execute(update_query, (urls,))
    cursor.connection.commit()

    article_chunks = [articles[i:i + batch_size] for i in range(0, len(articles), batch_size)]
    
    with ThreadPoolExecutor(max_workers=adjusted_threads) as executor:
        futures = []
        for i, chunk in enumerate(article_chunks):
            if not running:  
                break
            # Get the category of the first article in the chunk to determine the prompt
            if chunk:
                category = chunk[0][4]  # Category is at index 4 in the article tuple
                with open('gcam_config.json') as file:
                    data = json.load(file)
                if category in data:
                    prompt = data[category]['prompt']
                else:
                    prompt = "Extract and summarize the key facts and information from this article."
                
                futures.append(executor.submit(process_article_batch, chunk, prompt, i))
        
        executor.shutdown(wait=True, cancel_futures=True)

    reset_query = """
        UPDATE translated_articles 
        SET thread_status = 'pending' 
        WHERE thread_status = 'processing'
        AND utc_datetime >= NOW() - INTERVAL '2 days';
    """
    cursor.execute(reset_query)
    cursor.connection.commit()

@logger.log_execution()
def fetch_articles_loop(data, max_requests=35):
    """Fetches new articles"""
    while running:
        if os.path.exists(FETCH_LOCK_FILE):
            print("Fetch already running")
        else:
            try:
                open(FETCH_LOCK_FILE, "w").close()
                with psycopg2.connect(os.environ['CONN_STRING_ARTICLES']) as conn:
                    with conn.cursor() as cursor:
                        current_time = datetime.now(timezone.utc)
                        utc_datetime = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        past_time = (current_time - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S')

                        total_inserted = 0
                        for category in data:
                            if not running: 
                                break
                            category_articles = []
                            for code in data[category]['codes']:
                                if not running: 
                                    break
                                fetched_articles = fetch_articles_past(
                                    category, code, max_requests, data[category]['prompt'], 
                                    past_time, utc_datetime, cursor
                                )
                                if not fetched_articles.empty:
                                    category_articles.append(fetched_articles)
                                time.sleep(5)
                            
                            if category_articles:
                                category_df = pd.concat(category_articles, ignore_index=True)
                                insert_query = """
                                    INSERT INTO translated_articles 
                                    (url, title, language, sourcecountry, category, code, utc_datetime, thread_status)
                                    VALUES %s ON CONFLICT DO NOTHING;
                                """
                                execute_values(cursor, insert_query, category_df.to_records(index=False))
                                conn.commit()
                                total_inserted += len(category_df)
                                print(f"Inserted {len(category_df)} - {category}")

                        print(f"Fetch complete - {total_inserted} articles")
            except psycopg2.Error as e:
                print(f"Database error in fetch: {e}")
            finally:
                if os.path.exists(FETCH_LOCK_FILE):
                    os.remove(FETCH_LOCK_FILE)
        
        for _ in range(FETCH_INTERVAL):
            if not running:
                return
            time.sleep(1)
@logger.log_execution()
def process_articles_loop():
    """Processes pending articles"""
def process_articles_loop():
    """Processes pending articles"""
    while running:
        if os.path.exists(PROCESS_LOCK_FILE):
            print("Process already running")
        else:
            try:
                open(PROCESS_LOCK_FILE, "w").close()
                with psycopg2.connect(os.environ['CONN_STRING_ARTICLES']) as conn:
                    with conn.cursor() as cursor:
                        process_pending_articles(cursor)
            except psycopg2.Error as e:
                print(f"Database error in processing: {e}")
            finally:
                if os.path.exists(PROCESS_LOCK_FILE):
                    os.remove(PROCESS_LOCK_FILE)
        
        for _ in range(PROCESS_INTERVAL):
            if not running:
                return
            time.sleep(1)

def cleanup_stale_locks():
    print('remove locks')
    """Removes stale lock files and resets processing articles at startup"""
    if os.path.exists(FETCH_LOCK_FILE):
        os.remove(FETCH_LOCK_FILE)
        print("Removed stale fetch lock")
    if os.path.exists(PROCESS_LOCK_FILE):
        os.remove(PROCESS_LOCK_FILE)
        print("Removed stale process lock")

    try:
        with psycopg2.connect(os.environ['CONN_STRING_ARTICLES']) as conn:
            with conn.cursor() as cursor:
                print("Resetting stuck articles")
                cursor.execute("""
                UPDATE translated_articles 
                SET thread_status = 'pending' 
                WHERE thread_status = 'processing'
                AND utc_datetime >= NOW() - INTERVAL '2 days';
            """)
                reset_count = cursor.rowcount
                conn.commit()
                print(f"Reset {reset_count} stuck articles from 'processing' to 'pending'")
    except psycopg2.Error as e:
        print(f"Database error while resetting articles: {e}")
@logger.log_execution()
def run_daily_lda():
    """Runs LDA processing once per day"""
def run_daily_lda():
    """Runs LDA processing once per day"""
    while running:
        try:
            with psycopg2.connect(os.environ['CONN_STRING_ARTICLES']) as conn:
                with conn.cursor() as cursor:
                    check_and_process_lda(cursor)
            print("\nNo categories need processing yet. Sleeping for 24 hours...")
        except psycopg2.Error as e:
            print(f"Database error in LDA: {e}")
            
        for _ in range(86400):
            if not running:
                return
            time.sleep(1)

def check_environment():
    required_vars = [
        'LLAMA_3_ENDPOINT_URL',
        'LLAMA_3_ENDPOINT_KEY',
        'CONN_STRING_ARTICLES',
        'CONN_STRING_BACKEND'
    ]
    
    missing = [var for var in required_vars if var not in os.environ]
            
    if missing:
        print("Missing env var:", ", ".join(missing))
        sys.exit(1)
@logger.log_execution()
def main():
    global running  

    signal.signal(signal.SIGINT, signal_handler)  
    signal.signal(signal.SIGTERM, signal_handler) 

    check_environment()
    cleanup_stale_locks()

    with open('gcam_config.json') as file:
        data = json.load(file)

    print("Starting main loop")
    fetch_thread = threading.Thread(target=fetch_articles_loop, args=(data,), daemon=True)
    print("Fetch thread started")
    process_thread = threading.Thread(target=process_articles_loop, daemon=True)
    print("Process thread started")
    lda_thread = threading.Thread(target=run_daily_lda, daemon=True)

    threads = [fetch_thread, process_thread, lda_thread]
    for t in threads:
        t.start()

    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nSHUTDOWN triggered")
        running = False  

    print("Shutting now")
    cleanup_stale_locks()

    #giving worker threads some time to exit
    for _ in range(10):
        if not any(t.is_alive() for t in threads):
            break
        print("Waiting for threads to exit")
        time.sleep(1)

    print("Shutdown DONE")
    os._exit(0)  



if __name__ == '__main__':
    main()

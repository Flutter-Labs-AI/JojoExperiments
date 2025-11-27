import os
import orion
import signal
import logging
import argparse
import psycopg2
import threading
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from core.alerts.alerts_logger import AlertLogger
import warnings



load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'
)

warnings.filterwarnings("ignore")

logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

logging.getLogger("azure").propagate = False
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").propagate = False
#logger = logging.getLogger('orion-article-parser')
logger = AlertLogger('orion-article-parser')



shutdown_event = threading.Event()


def signal_handler(sig, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {sig}, initiating shutdown...")
    shutdown_event.set()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Process articles through a chain of steps.')

    parser.add_argument(
        '--steps',
        type=str,
        required=True,
        help='Comma-separated list of processing steps (e.g., fetch,content,translate)'
    )

    parser.add_argument(
        '--wait',
        type=str,
        default='5m',
        help='Wait time between checks for new articles (e.g., 5m, 30s)'
    )

    parser.add_argument(
        '--max-threads',
        type=int,
        default=5,
        help='Maximum number of worker threads'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of articles to process in each batch'
    )

    parser.add_argument(
        '-b',
        '--back-date',
        action='store_true',
        help='whether to run in backdate mode'
    )
    
    parser.add_argument(
        '--start-time',
        type = str,
        default = (datetime.now(timezone.utc)-timedelta(1)).strftime("%Y-%m-%d %H:%M:%S"),
        help = 'Argument used to identify start_date for backdating'
    )
    
    parser.add_argument(
        '--end-time',
        type = str,
        default = (datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S"),
        help = 'Argument used to identify end_date for backdating'
    )

    return parser.parse_args()
@logger.log_execution()
def main():
    """Main entry point."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse command-line arguments
    args = parse_args()

    # Parse steps
    steps = args.steps.split(',')
    

    with psycopg2.connect(os.environ['CONN_STRING_ARTICLES']) as article_conn:
        with psycopg2.connect(os.environ['CONN_STRING_BACKEND']) as backend_conn:
            db_cursors = {
                'articles': article_conn.cursor(),
                'backend': backend_conn.cursor()
            }
            
            if db_cursors['articles'] and db_cursors['backend']:
                
                # Create and start processor
                processor = orion.ArticleProcessor(
                    steps=steps,
                    max_threads=args.max_threads,
                    wait_time=args.wait,
                    batch_size=args.batch_size,
                    cursors=db_cursors,
                    logger=logger,
                    shutdown_event=shutdown_event,
                    extra_args={
                        'back_date': args.back_date,
                        'start_time': args.start_time,
                        'end_time': args.end_time
                    }
                )

                processor.start()
            
            else:
                print('Could not connect to articles or backend DBs. Make sure environment variables are initiaized.')
                exit()


if __name__ == "__main__":
    main()
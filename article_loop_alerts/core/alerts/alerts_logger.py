import logging
import time
import traceback
import os
import types
import threading
from functools import wraps
from datetime import datetime
from core.alerts.slack_notifier import SlackNotifier



class AlertLogger:
    def __init__(self, name: str, slack_webhook: str = os.getenv("SLACK_WEBHOOK_URL"), level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(process)d:%(threadName)s] - %(name)s - %(message)s'
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.slack = SlackNotifier(slack_webhook) if slack_webhook else None


    def log_execution(self, label=None):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                class_name = args[0].__class__.__name__ if args and hasattr(args[0], '__class__') else None
                func_name = func.__name__
                pid = os.getpid()
                tid = threading.current_thread().name

                full_name = f"{class_name}.{func_name}" if class_name else func_name
                self.logger.info(f"{full_name} STARTED at [{pid}:{tid}] | Start time: {datetime.fromtimestamp(start_time)}")
                if self.slack:
                    self.slack.send(f"""
                        *STARTED*
                        \n`{full_name}` in [{pid}:{tid}]
                        \n`{full_name}` *Start time: {datetime.fromtimestamp(start_time)}*
                    """)
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    tb = traceback.format_exc()
                    self.logger.error(f"{full_name} | EXCEPTION at [{pid}:{tid}] | Time: {datetime.fromtimestamp(time.time())}\n{tb}\n")
                    if self.slack:
                        self.slack.send(f"""
                            :warning: *EXCEPTION*
                            \n`{full_name}` in [{pid}:{tid}]*
                            \n`{full_name}` *Time: {datetime.fromtimestamp(time.time())}*
                            \n```\n{tb}\n```
                            \n----------------------------------------------------------
                        """)
                    raise
                else:
                    end_time = time.time()
                    duration = end_time - start_time
                    self.logger.info(f"{full_name} | ENDED at [{pid}:{tid}] | Duration: {duration:.2f}s | End time: {datetime.fromtimestamp(end_time)}")
                    if self.slack:
                        self.slack.send(f"""
                            \033\u2705\033 *ENDED*
                            \n`{full_name}` in [{pid}:{tid}]
                            \n`{full_name}` *Duration: {duration:.2f}s*
                            \n`{full_name}` *End time: {datetime.fromtimestamp(end_time)}*
                            \n----------------------------------------------------------
                        """)
                    return result
            return wrapper
        return decorator


    def log_all_methods(self, cls):
        for attr_name, attr_value in cls.__dict__.items():
            if isinstance(attr_value, types.FunctionType):
                setattr(cls, attr_name, self.log_execution()(attr_value))
        return cls
    

    def info(self, msg): 
        self.logger.info(msg)
        if self.slack:
            self.slack.send(msg)
    def warning(self, msg): 
        self.logger.warning(msg)
    def error(self, msg): 
        self.logger.error(msg)
    def critical(self, msg):
        self.logger.critical(msg)
        if self.slack:
            self.slack.send(f":rotating_light: CRITICAL: {msg}")



from datetime import datetime
from enum import Enum
from threading import Lock


class LogLevel(Enum):
    DEBUG = 'D'
    INFO = 'I'
    WARNING = 'W'
    ERROR = 'E'
    CRITICAL = 'C'


class _Logger:

    _instances = {}

    def __init__(self, name: str):
        self.name = name
        self._lock = Lock()

    def log(self, level: LogLevel, message: str, *args, **kwargs):
        with self._lock:
            print(f'[{datetime.now()} {level.value}] {self.name}: {message}', *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        self.log(LogLevel.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        self.log(LogLevel.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self.log(LogLevel.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        self.log(LogLevel.ERROR, message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        self.log(LogLevel.CRITICAL, message, *args, **kwargs)

    @classmethod
    def get_logger(cls, name: str):
        try:
            res = cls._instances[name]
        except KeyError:
            res = cls(name)
            cls._instances[name] = res
        return res


def getLogger(name: str = 'Program'):
    return _Logger.get_logger(name)

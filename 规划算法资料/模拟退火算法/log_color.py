from datetime import datetime
from enum import Enum


# color code, add your own color here if you need extra
class Color(Enum):
    WHITE = 0
    BLACK =30
    BLUE =34
    GREEN =32
    CYAN =36
    RED =31
    PURPLE =35
    BROWN =33

class LogLevel(Enum):
    CRITICAL = 'CRITICAL'
    ERROR = 'ERROR'
    WARN = 'WARN'
    INFO = 'INFO'
    DEBUG = 'DEBUG'
    SILLY = 'SILLY'
    PASS = 'PASS'

log_color_dict = {}
log_color_dict[LogLevel.CRITICAL] = Color.PURPLE
log_color_dict[LogLevel.ERROR] = Color.RED
log_color_dict[LogLevel.WARN] = Color.BLUE
log_color_dict[LogLevel.PASS] = Color.GREEN
log_color_dict[LogLevel.INFO] = Color.BROWN

def print_color(content: str, color: Color = Color.WHITE):
    print(f"\033[0;{color.value}m{content}\033[0m")


def log(content: str, level: LogLevel = LogLevel.DEBUG):
    print_color(f'{datetime.now()} - [{level.value}] - {content}', log_color_dict[level])

if __name__ == "__main__":

    log('Hello', LogLevel.ERROR)
    log('Hello', LogLevel.PASS)
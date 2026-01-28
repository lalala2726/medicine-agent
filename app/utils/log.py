from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


class PrintLogger:
    """使用 print 输出日志，带东八区时间戳。"""

    @staticmethod
    def _format(message: str, *args: object) -> str:
        if args:
            return message % args
        return message

    @staticmethod
    def _now() -> str:
        try:
            tz = ZoneInfo("Asia/Shanghai")
        except Exception:
            tz = timezone(timedelta(hours=8))
        now = datetime.now(tz=tz)
        milliseconds = int(now.microsecond / 1000)
        return f"{now:%Y-%m-%d %H:%M:%S}.{milliseconds:03d}"

    def info(self, message: str, *args: object) -> None:
        print(f"{self._now()} [INFO] {self._format(message, *args)}")

    def warning(self, message: str, *args: object) -> None:
        print(f"{self._now()} [WARN] {self._format(message, *args)}")

    def error(self, message: str, *args: object) -> None:
        print(f"{self._now()} [ERROR] {self._format(message, *args)}")


logger = PrintLogger()

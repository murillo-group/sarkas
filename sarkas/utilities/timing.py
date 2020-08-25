import time
from dataclasses import dataclass, field
from typing import Optional


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class SarkasTimer:
    """
    Timer class modified from https://realpython.com/python-timer/
    """
    name: Optional[str] = None
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        return elapsed_time

    @staticmethod
    def current() -> float:
        return time.perf_counter()


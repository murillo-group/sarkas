import time
from dataclasses import dataclass, field
from typing import Optional

from sarkas.utilities.exceptions import TimerError


@dataclass
class SarkasTimer:
    """
    Timer class modified from https://realpython.com/python-timer/
    """

    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter_ns()

    def stop(self) -> float:
        """
        Stop the timer, and report the elapsed time.

        Returns
        -------
        elapsed_time: float
            Elapsed time in nanoseconds.

        """
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter_ns() - self._start_time
        self._start_time = None

        return elapsed_time

    @staticmethod
    def current() -> float:
        """Grab the current time in nanoseconds."""
        return time.perf_counter_ns()

    @staticmethod
    def time_division(tme: float) -> list:
        """
        Divide time into hours, min, sec, msec, microsec (usec), and nanosec.

        Parameters
        ----------
        tme : float
            Time in nanoseconds.

        Returns
        -------
        : list
            [hours, min, sec, msec, microsec (usec), nanosec]

        """
        t_hrs, rem = divmod(tme, 3.6e12)
        t_min, rem_m = divmod(rem, 6e10)
        t_sec, rem_s = divmod(rem_m, 1e9)
        t_msec, rem_ms = divmod(rem_s, 1e6)
        t_usec, rem_us = divmod(rem_ms, 1e3)
        t_nsec, _ = divmod(rem_us, 1)

        return [t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec]

"""
Module for handling the timing in a MD run.
"""
import datetime
import sys
import time
from dataclasses import dataclass, field
from os.path import exists as os_path_exists
from typing import Optional

from .exceptions import TimerError


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

    def stop(self) -> int:
        """
        Stop the timer, and report the elapsed time.

        Returns
        -------
        elapsed_time: int
            Elapsed time in nanoseconds.

        """
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter_ns() - self._start_time
        self._start_time = None

        return elapsed_time

    @staticmethod
    def current() -> int:
        """Grab the current time in nanoseconds."""
        return time.perf_counter_ns()

    @staticmethod
    def time_division(tme: int) -> list:
        """
        Divide time into hours, min, sec, msec, microsec (usec), and nanosec.

        Parameters
        ----------
        tme : int
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


def datetime_stamp(log_file):
    """Add a Date and Time stamp to log file. If the file exists it appends three paragraph so that it is easier to see
    the new line.

    Parameters
    ----------
    log_file : str
        Path to file on which date time stamp should be appended to.

    """

    if os_path_exists(log_file):
        with open(log_file, "a+") as f_log:
            # Add some space to better distinguish the new beginning
            print(f"\n\n\n", file=f_log)

    with open(log_file, "a+") as f_log:
        ct = datetime.datetime.now()
        print(f"{'':~^80}", file=f_log)
        print(f"Date: {ct.year} - {ct.month} - {ct.day}", file=f_log)
        print(f"Time: {ct.hour}:{ct.minute}:{ct.second}", file=f_log)
        print(f"{'':~^80}\n", file=f_log)


def time_stamp(log_file: str, message: str, timing: tuple, print_to_screen: bool = False):
    """
    Print out to screen elapsed times. If verbose output, print to file first and then to screen.

    Parameters
    ----------
    log_file : str
        Path to file on which date time stamp should be appended to.

    message : str
        Message to print.

    timing : tuple
        Time in hrs, min, sec, msec, usec, nsec.

    print_to_screen : bool
        Flag for printing the message to screen. Default is False.
    """

    screen = sys.stdout
    f_log = open(log_file, "a+")
    repeat = 2 if print_to_screen else 1
    t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec = timing

    # redirect printing to file
    sys.stdout = f_log
    while repeat > 0:

        if t_hrs == 0 and t_min == 0 and t_sec <= 2:
            print(f"\n{message} Time: {int(t_sec)} sec {int(t_msec)} msec {int(t_usec)} usec {int(t_nsec)} nsec")
        else:
            print(f"\n{message} Time: {int(t_hrs)} hrs {int(t_min)} min {int(t_sec)} sec")

        repeat -= 1
        sys.stdout = screen

    f_log.close()

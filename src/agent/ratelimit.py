"""Sliding-window rate limiter (at most `max_calls` per `period` seconds), thread-safe, with an injectable clock.

Two uses:
  * OUTBOUND: keeps our LLM calls under the provider's requests-per-minute quota (LLM_MAX_RPM), so a batch run
    paces itself instead of hitting 429 errors and backing off.
  * INBOUND: stops one demo visitor from flooding the app (and our API bills) with turns.
max_calls <= 0 disables the limiter.
"""
import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_calls: int, period: float = 60.0, clock=time.monotonic, sleep=time.sleep):
        self.max_calls, self.period = int(max_calls), float(period)
        self._clock, self._sleep = clock, sleep
        self._hits: deque = deque()
        self._lock = threading.Lock()

    def try_acquire(self) -> tuple[bool, float]:
        """Non-blocking. Returns (allowed, seconds_to_wait_if_not_allowed)."""
        if self.max_calls <= 0:
            return True, 0.0
        with self._lock:
            now = self._clock()
            while self._hits and now - self._hits[0] >= self.period:
                self._hits.popleft()
            if len(self._hits) < self.max_calls:
                self._hits.append(now)
                return True, 0.0
            return False, max(self.period - (now - self._hits[0]), 0.0)

    def acquire(self) -> float:
        """Blocking. Waits for a free slot and returns how many seconds it waited."""
        waited = 0.0
        while True:
            ok, wait = self.try_acquire()
            if ok:
                return waited
            self._sleep(wait + 0.01)
            waited += wait + 0.01

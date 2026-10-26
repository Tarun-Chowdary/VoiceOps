import os, sys, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("EMBED_BACKEND", "hash")

from src.agent.ratelimit import RateLimiter


class Clock:
    """Fake time: sleeping simply moves the clock forward."""
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def limiter(n, period=60.0):
    c = Clock()
    return RateLimiter(n, period, clock=c.now, sleep=c.sleep), c


def test_allows_up_to_the_limit_then_says_how_long_to_wait():
    rl, c = limiter(3)
    assert [rl.try_acquire()[0] for _ in range(3)] == [True, True, True]
    ok, wait = rl.try_acquire()
    assert not ok and 59.9 <= wait <= 60.0


def test_the_window_slides():
    rl, c = limiter(2, 60)
    rl.try_acquire(); c.t += 30; rl.try_acquire()
    assert not rl.try_acquire()[0]
    c.t += 31                                               # the first call is now 61 s old, so one slot is free again
    assert rl.try_acquire()[0] and not rl.try_acquire()[0]


def test_blocking_acquire_waits_just_long_enough():
    rl, c = limiter(1, 10)
    assert rl.acquire() == 0
    waited = rl.acquire()
    assert 9.9 <= waited <= 10.1 and len(c.slept) == 1


def test_zero_disables_the_limiter():
    rl, c = limiter(0)
    assert all(rl.try_acquire()[0] for _ in range(1000)) and c.slept == []


def test_llm_wrapper_is_paced_but_cache_hits_are_free(monkeypatch):
    import types
    from src.agent import llm
    rl, c = limiter(2, 60)
    monkeypatch.setattr(llm, "_limiter", rl)
    calls = []

    def create(**kw):
        calls.append(kw)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))])

    monkeypatch.setattr(llm, "_client", types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))))
    tag = uuid.uuid4().hex                                  # unique prompts, so earlier runs' disk cache cannot interfere
    for i in range(3):
        llm.chat("sys", [{"role": "user", "text": f"{tag}-{i}"}])
    assert len(calls) == 3 and len(c.slept) == 1            # the 3rd request had to wait for the window to slide
    before = len(calls)
    llm.chat("sys", [{"role": "user", "text": f"{tag}-0"}])  # identical request: served from the cache
    assert len(calls) == before and llm.LAST_CACHED and len(c.slept) == 1

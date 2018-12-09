import time
from collections import Counter


class SimpleTimer:
    total_times = Counter()
    total_calls = Counter()

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        finish = time.time()
        SimpleTimer.total_calls[self.name] += 1
        SimpleTimer.total_times[self.name] += finish - self.start

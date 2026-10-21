"""Small bounded CPU/GPU pipeline; no hardware tasks or nested submissions."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import threading


_RESOURCES = {'gpu': threading.RLock(), 'ocr_cpu': threading.RLock()}


def serialized_resource(name):
    lock = _RESOURCES[name]
    def decorate(function):
        @wraps(function)
        def invoke(*args, **kwargs):
            with lock:
                return function(*args, **kwargs)
        return invoke
    return decorate


def ordered_bounded_map(function, items, workers=2):
    """At most workers unfinished jobs; deterministic input-order publication."""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('candidate workers must be an integer in [1, 4]')
    iterator = iter(items)
    sentinel = object()
    if workers == 1:
        yield from map(function, iterator)
        return
    pending = deque()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='perception') as pool:
        try:
            for _ in range(workers):
                item = next(iterator, sentinel)
                if item is sentinel:
                    break
                pending.append(pool.submit(function, item))
            while pending:
                result = pending.popleft().result()
                item = next(iterator, sentinel)
                if item is not sentinel:
                    pending.append(pool.submit(function, item))
                yield result
        finally:
            for future in pending:
                future.cancel()

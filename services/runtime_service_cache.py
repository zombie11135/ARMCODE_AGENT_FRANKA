import json
import threading
from typing import Any, Callable, Dict, Tuple


_SERVICE_CACHE: Dict[Tuple[str, str], Any] = {}
_CACHE_LOCK = threading.RLock()


def _make_key(params: Dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)


def get_or_create_service(name: str, params: Dict[str, Any], factory: Callable[[], Any]):
    with _CACHE_LOCK:
        return _get_or_create_service(name, params, factory)


def _get_or_create_service(name, params, factory):
    key = (str(name), _make_key(params))
    if key not in _SERVICE_CACHE:
        print(f"[ServiceCache] loading {name}")
        _SERVICE_CACHE[key] = factory()
    else:
        print(f"[ServiceCache] reuse {name}")
    return _SERVICE_CACHE[key]


def clear_service_cache():
    with _CACHE_LOCK:
        _SERVICE_CACHE.clear()


def close_cached_services():
    with _CACHE_LOCK:
        services = list(_SERVICE_CACHE.values())
        _SERVICE_CACHE.clear()
    for service in services:
        stop = getattr(service, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception as e:
                print(f"[ServiceCache][WARN] failed to stop cached service: {e}")


def service_cache_summary() -> Dict[str, int]:
    with _CACHE_LOCK:
        return _service_cache_summary()


def _service_cache_summary():
    summary: Dict[str, int] = {}
    for name, _ in _SERVICE_CACHE.keys():
        summary[name] = summary.get(name, 0) + 1
    return summary

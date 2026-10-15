import json
from typing import Any, Callable, Dict, Tuple


_SERVICE_CACHE: Dict[Tuple[str, str], Any] = {}


def _make_key(params: Dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)


def get_or_create_service(name: str, params: Dict[str, Any], factory: Callable[[], Any]):
    key = (str(name), _make_key(params))
    if key not in _SERVICE_CACHE:
        print(f"[ServiceCache] loading {name}")
        _SERVICE_CACHE[key] = factory()
    else:
        print(f"[ServiceCache] reuse {name}")
    return _SERVICE_CACHE[key]


def clear_service_cache():
    _SERVICE_CACHE.clear()


def close_cached_services():
    for service in list(_SERVICE_CACHE.values()):
        stop = getattr(service, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception as e:
                print(f"[ServiceCache][WARN] failed to stop cached service: {e}")
    _SERVICE_CACHE.clear()


def service_cache_summary() -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for name, _ in _SERVICE_CACHE.keys():
        summary[name] = summary.get(name, 0) + 1
    return summary

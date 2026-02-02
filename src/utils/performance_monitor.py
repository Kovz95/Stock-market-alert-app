"""
Performance monitoring utilities for timing, metrics, and optional system metrics.

Provides:
- record_metric() / get_metrics() for custom metrics
- @timed decorator and timer() context manager for function/block timing
- start_timer() / stop_timer() / elapsed() for manual timing
- get_system_metrics() for memory (and CPU when available)
- Optional periodic system metrics and metric saving when enabled via performance_config
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from threading import Lock, Thread
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

# Optional integration with performance_config
try:
    from src.config.performance_config import (
        METRICS_SAVE_INTERVAL,
        PERFORMANCE_MONITORING_ENABLED,
        SYSTEM_METRICS_INTERVAL,
    )
except ImportError:
    PERFORMANCE_MONITORING_ENABLED = True
    METRICS_SAVE_INTERVAL = 60
    SYSTEM_METRICS_INTERVAL = 10

# In-memory metrics store
_metrics: Dict[str, list[tuple[float, float, Optional[Dict[str, Any]]]]] = {}  # name -> [(timestamp, value, tags), ...]
_timers: Dict[str, float] = {}  # name -> start time
_lock = Lock()

# Optional background thread for periodic system metrics / save
_background_thread: Optional[Thread] = None
_background_stop = False

F = TypeVar("F", bound=Callable[..., Any])


def record_metric(
    name: str,
    value: float,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a single metric value (timestamp is recorded automatically)."""
    if not PERFORMANCE_MONITORING_ENABLED:
        return
    with _lock:
        if name not in _metrics:
            _metrics[name] = []
        _metrics[name].append((time.time(), value, tags))


def get_metrics(
    name: Optional[str] = None,
    since: Optional[float] = None,
) -> Dict[str, list[tuple[float, float, Optional[Dict[str, Any]]]]]:
    """Return recorded metrics, optionally filtered by name and start time."""
    with _lock:
        if name:
            data = {k: v for k, v in _metrics.items() if k == name}
        else:
            data = dict(_metrics)
        if since is not None:
            data = {
                k: [(ts, val, tags) for ts, val, tags in v if ts >= since]
                for k, v in data.items()
            }
            data = {k: v for k, v in data.items() if v}
        return data


def clear_metrics(name: Optional[str] = None) -> None:
    """Clear recorded metrics, optionally for a single metric name."""
    with _lock:
        if name is None:
            _metrics.clear()
        elif name in _metrics:
            del _metrics[name]


def start_timer(name: str) -> None:
    """Start a named timer. Use stop_timer(name) or elapsed(name) to get duration."""
    if not PERFORMANCE_MONITORING_ENABLED:
        return
    with _lock:
        _timers[name] = time.perf_counter()


def stop_timer(name: str, record_as_metric: bool = True) -> Optional[float]:
    """Stop a named timer and return elapsed seconds. Optionally record as metric."""
    with _lock:
        if name not in _timers:
            return None
        elapsed = time.perf_counter() - _timers[name]
        del _timers[name]
    if record_as_metric and PERFORMANCE_MONITORING_ENABLED:
        record_metric(f"timer_{name}", elapsed * 1000.0, tags={"unit": "ms"})
    return elapsed


def elapsed(name: str) -> Optional[float]:
    """Return elapsed seconds for a running timer without stopping it."""
    with _lock:
        if name not in _timers:
            return None
        return time.perf_counter() - _timers[name]


@contextmanager
def timer(name: str, record_metric: bool = True):
    """Context manager to time a block and optionally record duration as a metric."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        if record_metric and PERFORMANCE_MONITORING_ENABLED:
            record_metric(f"timer_{name}", duration * 1000.0, tags={"unit": "ms"})


def timed(metric_name: Optional[str] = None, record: bool = True) -> Callable[[F], F]:
    """Decorator to time a function and optionally record duration as a metric."""

    def decorator(func: F) -> F:
        name = metric_name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                if record and PERFORMANCE_MONITORING_ENABLED:
                    record_metric(
                        f"timer_{name}",
                        duration * 1000.0,
                        tags={"unit": "ms", "callable": name},
                    )

        return wrapper  # type: ignore[return-value]

    return decorator


def get_system_metrics() -> Dict[str, Any]:
    """Return current system metrics (memory; CPU if available)."""
    out: Dict[str, Any] = {}
    try:
        import resource  # Unix only

        usage = resource.getrusage(resource.RUSAGE_SELF)
        out["memory_rss_mb"] = usage.ru_maxrss / 1024.0  # typically KB on Linux
    except (ImportError, AttributeError):
        pass
    try:
        import psutil

        proc = psutil.Process()
        out["memory_rss_mb"] = proc.memory_info().rss / (1024 * 1024)
        out["memory_vms_mb"] = proc.memory_info().vms / (1024 * 1024)
        out["cpu_percent"] = proc.cpu_percent(interval=0.1)
    except ImportError:
        pass
    return out


def _record_system_metrics_once() -> None:
    """Record current system metrics into the metrics store."""
    if not PERFORMANCE_MONITORING_ENABLED:
        return
    sm = get_system_metrics()
    for key, value in sm.items():
        if isinstance(value, (int, float)):
            record_metric(f"system_{key}", float(value), tags={"source": "system"})


def _background_metrics_loop() -> None:
    """Background loop: record system metrics and optionally save/rotate metrics."""
    last_system = 0.0
    last_save = 0.0
    while not _background_stop:
        now = time.time()
        if now - last_system >= SYSTEM_METRICS_INTERVAL:
            try:
                _record_system_metrics_once()
            except Exception as e:
                logger.debug("Performance monitor system metrics: %s", e)
            last_system = now
        if now - last_save >= METRICS_SAVE_INTERVAL:
            # Placeholder: could persist _metrics to file/DB here
            last_save = now
        time.sleep(min(SYSTEM_METRICS_INTERVAL, METRICS_SAVE_INTERVAL, 10))


def start_background_monitoring() -> None:
    """Start background thread for periodic system metrics and metric saving."""
    global _background_thread, _background_stop
    if not PERFORMANCE_MONITORING_ENABLED:
        return
    if _background_thread is not None and _background_thread.is_alive():
        return
    _background_stop = False
    _background_thread = Thread(target=_background_metrics_loop, daemon=True)
    _background_thread.start()
    logger.info("Performance monitor background thread started.")


def stop_background_monitoring() -> None:
    """Stop the background monitoring thread."""
    global _background_stop
    _background_stop = True


def get_snapshot() -> Dict[str, Any]:
    """Return a snapshot of current metrics and optional system metrics."""
    with _lock:
        metrics_copy = {
            k: list(v) for k, v in _metrics.items()
        }
    return {
        "metrics": metrics_copy,
        "active_timers": list(_timers.keys()),
        "system": get_system_metrics(),
    }


__all__ = [
    "record_metric",
    "get_metrics",
    "clear_metrics",
    "start_timer",
    "stop_timer",
    "elapsed",
    "timer",
    "timed",
    "get_system_metrics",
    "get_snapshot",
    "start_background_monitoring",
    "stop_background_monitoring",
]

"""
Deps -- Optional dependency upgrades with stdlib fallback.

Everything in sovrun works without external packages. If tenacity, httpx,
or rich are pip-installed, sovrun auto-detects and uses them for better
retry logic, HTTP performance, and CLI output. If not, stdlib equivalents
handle it.

Usage:
    from sovrun.core.deps import get_retry, get_http_client, dep_status

    retry = get_retry()           # tenacity.retry or stdlib wrapper
    client = get_http_client()    # httpx.Client or urllib wrapper
    dep_status()                  # show what's installed

CLI:
    python3 -m sovrun.core.deps --status
    python3 -m sovrun.core.deps --install
"""
from __future__ import annotations

import argparse
import functools
import json
import ssl
import time
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Optional dependency registry
# ---------------------------------------------------------------------------

OPTIONAL_DEPS: dict[str, str] = {
    "tenacity": "Better retry with jitter, wait strategies, and stop conditions",
    "httpx": "Modern HTTP client with connection pooling and timeout control",
    "rich": "Terminal formatting, tables, progress bars for CLI output",
}


def _is_available(name: str) -> bool:
    """Check if a package is importable."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def dep_status() -> dict[str, bool]:
    """Return dict of optional dependency name -> installed bool."""
    return {name: _is_available(name) for name in OPTIONAL_DEPS}


# ---------------------------------------------------------------------------
# Retry: tenacity if available, else stdlib
# ---------------------------------------------------------------------------

def _stdlib_retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Stdlib retry decorator with exponential backoff."""

    def decorator(fn: Callable[..., _T]) -> Callable[..., _T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        delay = min(backoff_base * (2 ** attempt), backoff_max)
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


def get_retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Return a retry decorator. Uses tenacity if installed, else stdlib.

    Usage:
        retry = get_retry(max_attempts=5)

        @retry
        def flaky_call():
            ...
    """
    if _is_available("tenacity"):
        import tenacity
        return tenacity.retry(  # type: ignore[return-value]
            stop=tenacity.stop_after_attempt(max_attempts),
            wait=tenacity.wait_exponential(
                multiplier=backoff_base, max=backoff_max,
            ),
            retry=tenacity.retry_if_exception_type(retryable_exceptions),
            reraise=True,
        )
    return _stdlib_retry(
        max_attempts=max_attempts,
        backoff_base=backoff_base,
        backoff_max=backoff_max,
        retryable_exceptions=retryable_exceptions,
    )


# ---------------------------------------------------------------------------
# HTTP client: httpx if available, else urllib wrapper
# ---------------------------------------------------------------------------

class _UrllibClient:
    """Minimal HTTP client wrapping urllib.request.

    Matches the subset of httpx.Client API used by sovrun modules:
    get, post, patch, delete, close, and response.status_code / .json() / .text.
    """

    def __init__(self, base_url: str = "", headers: dict[str, str] | None = None,
                 timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_headers = headers or {}
        self.timeout = timeout
        # Allow self-signed certs in dev (n8n local)
        self._ssl_ctx = ssl.create_default_context()

    def _request(self, method: str, url: str,
                 json_body: Any = None,
                 headers: dict[str, str] | None = None) -> _UrllibResponse:
        full_url = f"{self.base_url}{url}" if not url.startswith("http") else url
        merged = {**self.default_headers, **(headers or {})}

        body_bytes: bytes | None = None
        if json_body is not None:
            body_bytes = json.dumps(json_body).encode("utf-8")
            merged.setdefault("Content-Type", "application/json")

        req = Request(full_url, data=body_bytes, headers=merged, method=method)
        try:
            resp = urlopen(req, timeout=self.timeout, context=self._ssl_ctx)
            data = resp.read().decode("utf-8")
            return _UrllibResponse(
                status_code=resp.getcode() or 200,
                text=data,
                headers=dict(resp.headers),
            )
        except HTTPError as exc:
            data = exc.read().decode("utf-8") if exc.fp else ""
            return _UrllibResponse(
                status_code=exc.code,
                text=data,
                headers=dict(exc.headers) if exc.headers else {},
            )

    def get(self, url: str, **kwargs: Any) -> _UrllibResponse:
        return self._request("GET", url, **kwargs)

    def post(self, url: str, json: Any = None, **kwargs: Any) -> _UrllibResponse:
        return self._request("POST", url, json_body=json, **kwargs)

    def patch(self, url: str, json: Any = None, **kwargs: Any) -> _UrllibResponse:
        return self._request("PATCH", url, json_body=json, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> _UrllibResponse:
        return self._request("DELETE", url, **kwargs)

    def close(self) -> None:
        pass

    def __enter__(self) -> _UrllibClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class _UrllibResponse:
    """Minimal response object matching httpx.Response interface."""

    def __init__(self, status_code: int, text: str,
                 headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPError(
                url="", code=self.status_code, msg=self.text,
                hdrs={}, fp=None,  # type: ignore[arg-type]
            )


def get_http_client(
    base_url: str = "",
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    """Return an HTTP client. Uses httpx.Client if installed, else urllib wrapper.

    Both support: .get(), .post(), .patch(), .delete(), .close()
    Response objects support: .status_code, .text, .json(), .raise_for_status()
    """
    if _is_available("httpx"):
        import httpx
        return httpx.Client(
            base_url=base_url,
            headers=headers or {},
            timeout=timeout,
        )
    return _UrllibClient(
        base_url=base_url,
        headers=headers or {},
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _show_status() -> None:
    statuses = dep_status()
    print("\n=== Sovrun Optional Dependencies ===\n")
    for name, installed in statuses.items():
        mark = "+" if installed else "-"
        desc = OPTIONAL_DEPS[name]
        status_text = "installed" if installed else "not installed"
        print(f"  [{mark}] {name:12s}  {status_text:16s}  {desc}")
    print()
    installed_count = sum(1 for v in statuses.values() if v)
    print(f"  {installed_count}/{len(statuses)} optional deps installed.")
    print(f"  All modules work without them. They just get better with them.\n")


def _show_install() -> None:
    missing = [name for name, installed in dep_status().items() if not installed]
    if not missing:
        print("\nAll optional dependencies already installed.\n")
        return
    cmd = f"pip install {' '.join(missing)}"
    print(f"\nInstall missing optional dependencies:\n")
    print(f"  {cmd}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deps -- optional dependency status and install helper",
    )
    parser.add_argument("--status", action="store_true",
                        help="Show which optional deps are installed")
    parser.add_argument("--install", action="store_true",
                        help="Print pip install command for missing deps")
    args = parser.parse_args()

    if not any([args.status, args.install]):
        parser.print_help()
        return

    if args.status:
        _show_status()
    if args.install:
        _show_install()


if __name__ == "__main__":
    main()

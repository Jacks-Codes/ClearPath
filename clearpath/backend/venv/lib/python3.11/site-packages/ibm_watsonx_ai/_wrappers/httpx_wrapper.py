#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import asyncio
import json as js
import os
import queue
import ssl
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from random import random
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    TypeVar,
    overload,
)

import httpx

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


HTTPX_DEFAULT_TIMEOUT = httpx.Timeout(timeout=30 * 60, connect=10)

HTTPX_KEEPALIVE_EXPIRY = 5
HTTPX_DEFAULT_LIMIT = httpx.Limits(
    max_connections=10,
    max_keepalive_connections=10,
    keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY,
)
DEFAULT_RETRY_STATUS_CODES = [429, 503, 504, 520]
MAX_RETRY_DELAY = 8
DEFAULT_DELAY = 0.5

_MAX_RETRIES = 10  # number of retries after the first failure
REMAINING_LIMIT_HEADER = "x-requests-limit-remaining"

RETRY_CONFIG: dict = {
    "retries": 3,
    "backoff_factor": 0.3,
    "status_forcelist": (401, 500, 502, 503, 504, 520, 521, 524),
}

additional_settings: dict = {}
verify: bool | str | None = None

TTransport = TypeVar(
    "TTransport",
    httpx.HTTPTransport,
    httpx.AsyncHTTPTransport,
)


def set_verify_for_httpx(func: Callable) -> Callable:
    """
    This decorator passes through the function with verify parameter from environment and global verify.
    Priority order: environment variable > global verify > default (True)
    """

    @wraps(func)
    def wrapper(*args: Any, **kw: Any) -> Any:
        # Use the centralized function to get effective verify value
        effective_verify = _get_effective_verify()

        if "verify" not in kw:
            kw.update({"verify": effective_verify})

        return func(*args, **kw)

    return wrapper


def _get_effective_verify() -> bool | str:
    """
    Get the effective verify value from global verify and environment variable.
    Priority order: environment variable > global verify > default (True)

    Returns the verify value to use for SSL verification.
    """
    global verify

    env_verify = os.environ.get("WX_CLIENT_VERIFY_REQUESTS")

    if env_verify is not None:
        if env_verify == "True" or env_verify == "":
            # Empty string means True (default verification)
            return True
        elif env_verify == "False":
            return False
        else:
            return env_verify

    if verify is not None:
        return verify

    return True


def _raise_verify_error(error: Exception) -> None:
    """
    Raise OSError with detailed message for SSL verification errors.

    Args:
        error: The original exception
    """
    raise OSError(
        f"Connection cannot be verified with default trusted CAs. "
        f"Please provide correct path to a CA_BUNDLE file or directory with "
        f"certificates of trusted CAs. Error: {error}"
    ) from error


def set_additional_settings_for_requests(
    func: Callable,
) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kw: Any) -> Any:
        kwargs = {}
        kwargs.update(additional_settings)
        kwargs.update(kw)
        return func(*args, **kwargs)

    return wrapper


def _build_transport(
    transport_cls: type[TTransport],
    api_client: APIClient,
    limits: httpx.Limits = HTTPX_DEFAULT_LIMIT,
) -> TTransport:
    global verify

    # Get credentials_verify from api_client
    credentials_verify = getattr(api_client.credentials, "verify", None)

    # Check if environment variable is set (even if empty)
    env_verify = os.environ.get("WX_CLIENT_VERIFY_REQUESTS")

    # Parse env_verify to proper type
    if env_verify is not None:
        if env_verify == "True" or env_verify == "":
            env_verify_parsed = True
        elif env_verify == "False":
            env_verify_parsed = False
        else:
            env_verify_parsed = env_verify  # type: ignore[assignment]
    else:
        env_verify_parsed = None

    # Calculate verify_initial with priority: credentials > env > global
    verify_initial = (
        credentials_verify
        if credentials_verify is not None
        else (env_verify_parsed if env_verify_parsed is not None else verify)
    )

    # Allow SSL fallback only if all verify sources (credentials, env var, global verify) are None
    allow_ssl_fallback = (
        credentials_verify is None and env_verify_parsed is None and verify is None
    )

    try:
        return transport_cls(  # type: ignore[call-arg]
            retries=RETRY_CONFIG["retries"],
            backoff_factor=RETRY_CONFIG["backoff_factor"],
            status_forcelist=RETRY_CONFIG["status_forcelist"],
            verify_initial=verify_initial,
            allow_ssl_fallback=allow_ssl_fallback,
            limits=limits,
        )
    except FileNotFoundError as e:
        # When verify is a string path that doesn't exist
        if isinstance(verify_initial, str):
            _raise_verify_error(e)
        raise


def _httpx_transport_params(
    api_client: APIClient, limits: httpx.Limits = HTTPX_DEFAULT_LIMIT
) -> httpx.HTTPTransport:
    return _build_transport(RetryTransport, api_client, limits)


def _httpx_async_transport_params(
    api_client: APIClient, limits: httpx.Limits = HTTPX_DEFAULT_LIMIT
) -> httpx.AsyncHTTPTransport:
    return _build_transport(AsyncRetryTransport, api_client, limits)


@overload
def _create_httpx_client(
    client_cls: type[HTTPXClient],
    transport: httpx.HTTPTransport,
    **kwargs: Any,
) -> HTTPXClient: ...


@overload
def _create_httpx_client(
    client_cls: type[HTTPXAsyncClient],
    transport: httpx.AsyncHTTPTransport,
    **kwargs: Any,
) -> HTTPXAsyncClient: ...


def _create_httpx_client(
    client_cls: type[HTTPXClient] | type[HTTPXAsyncClient],
    transport: httpx.HTTPTransport | httpx.AsyncHTTPTransport,
    **kwargs: Any,
) -> HTTPXClient | HTTPXAsyncClient:
    return client_cls(transport=transport, **kwargs)


@set_additional_settings_for_requests
def _get_httpx_client(transport: httpx.HTTPTransport, **kwargs: Any) -> HTTPXClient:
    # Get verify from transport if it's a RetryTransport
    if isinstance(transport, RetryTransport):
        verify_value = transport.get_effective_verify_for_client()
        kwargs["verify"] = verify_value
    return _create_httpx_client(HTTPXClient, transport, **kwargs)


@set_additional_settings_for_requests
def _get_async_httpx_client(
    transport: httpx.AsyncHTTPTransport, **kwargs: Any
) -> HTTPXAsyncClient:
    # Get verify from transport if it's an AsyncRetryTransport
    if isinstance(transport, AsyncRetryTransport):
        verify_value = transport.get_effective_verify_for_client()
        kwargs["verify"] = verify_value
    return _create_httpx_client(HTTPXAsyncClient, transport, **kwargs)


class HTTPXClient(httpx.Client):
    """Wrapper for httpx Sync Client"""

    def __init__(
        self, verify: ssl.SSLContext | str | bool | None = None, **kwargs: Any
    ):
        # Remove proxies from kwargs as they should be handled via transport or mounts
        kwargs.pop("proxies", None)
        super().__init__(
            verify=verify if verify is not None else bool(verify),
            timeout=kwargs.pop("timeout", None) or HTTPX_DEFAULT_TIMEOUT,
            limits=kwargs.pop("limits", None) or HTTPX_DEFAULT_LIMIT,
            **kwargs,
        )

    def post(  # type: ignore[override]
        self,
        url: str,
        *,
        content: str | bytes | None = None,
        json: dict | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        if json is not None and content is None:
            from ibm_watsonx_ai.utils.utils import NumpyTypeEncoder

            content = js.dumps(json, cls=NumpyTypeEncoder)

            if headers is not None and headers.get("Content-Type") is not None:
                headers["Content-Type"] = "application/json"

        response = super().post(
            url=url,
            content=content,
            headers=headers,
            params=params,
            **kwargs,
        )
        return response

    @contextmanager
    def post_stream(
        self,
        method: str,
        url: str,
        *,
        content: str | bytes | None = None,
        json: dict | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> Iterator[httpx.Response]:
        if json is not None and content is None:
            from ibm_watsonx_ai.utils.utils import NumpyTypeEncoder

            content = js.dumps(json, cls=NumpyTypeEncoder)

            if headers is not None and headers.get("Content-Type") is not None:
                headers["Content-Type"] = "application/json"

        with super().stream(
            method=method,
            url=url,
            content=content,
            headers=headers,
            params=params,
            **kwargs,
        ) as response:
            try:
                yield response
            finally:
                response.close()

    def __del__(self) -> None:
        try:
            # Closing the connection pool when the object is deleted
            self.close()
        except Exception:
            pass


class HTTPXAsyncClient(httpx.AsyncClient):
    def __init__(
        self, verify: ssl.SSLContext | str | bool | None = None, **kwargs: Any
    ):
        # Remove proxies from kwargs as they should be handled via transport or mounts
        kwargs.pop("proxies", None)
        super().__init__(
            verify=verify if verify is not None else bool(verify),
            timeout=kwargs.pop("timeout", None) or HTTPX_DEFAULT_TIMEOUT,
            limits=kwargs.pop("limits", None) or HTTPX_DEFAULT_LIMIT,
            **kwargs,
        )

    async def post(  # type: ignore[override]
        self,
        url: str,
        *,
        content: str | bytes | None = None,
        json: dict | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        if json is not None and content is None:
            from ibm_watsonx_ai.utils.utils import NumpyTypeEncoder

            content = js.dumps(json, cls=NumpyTypeEncoder)

            if headers and not headers.get("Content-Type"):
                headers["Content-Type"] = "application/json"

        response = await super().post(
            url=url,
            content=content,
            headers=headers,
            params=params,
            **kwargs,
        )
        return response

    @asynccontextmanager
    async def post_stream(
        self,
        method: str,
        url: str,
        *,
        content: str | bytes | None = None,
        json: dict | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[httpx.Response]:
        if json is not None and content is None:
            from ibm_watsonx_ai.utils.utils import NumpyTypeEncoder

            content = js.dumps(json, cls=NumpyTypeEncoder)

            if headers is not None and headers.get("Content-Type") is not None:
                headers["Content-Type"] = "application/json"

        async with super().stream(
            method=method,
            url=url,
            content=content,
            headers=headers,
            params=params,
            **kwargs,
        ) as response:
            try:
                yield response
            finally:
                await response.aclose()

    def __del__(self) -> None:
        try:
            # Closing the connection pool when the object is deleted
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:
            pass


def backoff_timeout(wx_delay_time: float, attempt: int) -> float:
    jitter = 1 + 0.25 * random()
    sleep_seconds = min(wx_delay_time * pow(2.0, attempt), MAX_RETRY_DELAY)
    return sleep_seconds * jitter


def _get_max_retries(
    instance_max_retries: int | None, decorator_max_retries: int
) -> int:
    if isinstance(instance_max_retries, int):
        wx_max_retries = instance_max_retries
    elif (env_max_retries := os.environ.get("WATSONX_MAX_RETRIES")) is not None:
        wx_max_retries = int(env_max_retries)
    else:
        wx_max_retries = decorator_max_retries
    return wx_max_retries


def _get_delay_time(
    instance_delay_time: float | None, decorator_delay_time: float
) -> float:
    if isinstance(instance_delay_time, float):
        wx_delay_time = instance_delay_time
    elif (env_delay_time := os.environ.get("WATSONX_DELAY_TIME")) is not None:
        wx_delay_time = float(env_delay_time)
    else:
        wx_delay_time = decorator_delay_time
    return wx_delay_time


def _get_retry_status_codes(
    instance_retry_status_codes: list | None, decorator_retry_status_codes: list
) -> list:
    wx_retry_status_codes = (
        instance_retry_status_codes
        or (
            list(
                map(
                    int,
                    os.environ.get("WATSONX_RETRY_STATUS_CODES", "")
                    .strip("[]")
                    .split(","),
                )
            )
            if os.environ.get("WATSONX_RETRY_STATUS_CODES")
            else []
        )
        or decorator_retry_status_codes
    )
    return wx_retry_status_codes


def _with_retry(
    max_retries: int = _MAX_RETRIES,
    delay_time: float = DEFAULT_DELAY,
    retry_status_codes: list[int] = DEFAULT_RETRY_STATUS_CODES,
) -> Callable:
    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> httpx.Response:
            response: httpx.Response | None = None

            wx_max_retries = _get_max_retries(self.max_retries, max_retries)

            wx_delay_time = _get_delay_time(self.delay_time, delay_time)

            wx_retry_status_codes = _get_retry_status_codes(
                self.retry_status_codes, retry_status_codes
            )

            for attempt in range(max_retries + 1):
                if response is not None:
                    response.close()
                response = function(self, *args, **kwargs)

                if (
                    response is not None
                    and (response.status_code in wx_retry_status_codes)
                    and attempt != wx_max_retries
                ):
                    if self._client.CLOUD_PLATFORM_SPACES:
                        rate_limit_remaining = int(
                            response.headers.get(
                                REMAINING_LIMIT_HEADER,
                                self.rate_limiter.capacity,
                            )
                        )
                        if rate_limit_remaining == 0:
                            self.rate_limiter.adjust_tokens(rate_limit_remaining)
                        else:
                            time.sleep(backoff_timeout(wx_delay_time, attempt))
                        self.rate_limiter.acquire()
                    else:
                        time.sleep(backoff_timeout(wx_delay_time, attempt))
                else:
                    break

            return response  # type:ignore[return-value]

        return wrapper

    return decorator


def _with_retry_stream(
    max_retries: int = _MAX_RETRIES,
    delay_time: float = DEFAULT_DELAY,
    retry_status_codes: list[int] = DEFAULT_RETRY_STATUS_CODES,
) -> Callable:
    """Decorator to retry the function if it encounters a 429 HTTP status."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @contextmanager  # Ensure the wrapped function remains a context manager
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Iterator[httpx.Response]:
            _exception = None
            response: httpx.Response | None = None

            wx_max_retries = _get_max_retries(self.max_retries, max_retries)

            wx_delay_time = _get_delay_time(self.delay_time, delay_time)

            wx_retry_status_codes = _get_retry_status_codes(
                self.retry_status_codes, retry_status_codes
            )

            for attempt in range(max_retries + 1):
                if response is not None:
                    response.close()
                with func(
                    self, *args, **kwargs
                ) as response:  # Call the original context manager
                    if (
                        response is not None
                        and (response.status_code in wx_retry_status_codes)
                        and attempt != wx_max_retries
                    ):
                        #  If the environment is set to cloud, the Token Bucket (rate_limiter here) is used to control traffic flow.
                        if self._client.CLOUD_PLATFORM_SPACES:
                            rate_limit_remaining = int(
                                response.headers.get(
                                    REMAINING_LIMIT_HEADER,
                                    self.rate_limiter.capacity,
                                )
                            )
                            if rate_limit_remaining == 0:
                                self.rate_limiter.adjust_tokens(rate_limit_remaining)
                            else:
                                time.sleep(backoff_timeout(wx_delay_time, attempt))
                            self.rate_limiter.acquire()
                        else:  # If CDP, don't use Token Bucket
                            time.sleep(backoff_timeout(wx_delay_time, attempt))
                        continue  # Retry the request
                    if response is not None:
                        yield response
                    return  # Ensure exit the loop after yielding

        return wrapper

    return decorator


def _with_async_retry(
    max_retries: int = _MAX_RETRIES,
    delay_time: float = DEFAULT_DELAY,
    retry_status_codes: list[int] = DEFAULT_RETRY_STATUS_CODES,
) -> Callable:
    def decorator(function: Callable) -> Callable:
        @wraps(function)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> httpx.Response:
            response: httpx.Response | None = None

            wx_max_retries = _get_max_retries(self.max_retries, max_retries)
            wx_delay_time = _get_delay_time(self.delay_time, delay_time)
            wx_retry_status_codes = _get_retry_status_codes(
                self.retry_status_codes, retry_status_codes
            )
            for attempt in range(wx_max_retries + 1):
                if response is not None:
                    await response.aclose()
                response = await function(self, *args, **kwargs)

                if (
                    response is not None
                    and (response.status_code in wx_retry_status_codes)
                    and attempt != wx_max_retries
                ):
                    if self._client.CLOUD_PLATFORM_SPACES:
                        rate_limit_remaining = int(
                            response.headers.get(
                                REMAINING_LIMIT_HEADER,
                                self.rate_limiter.capacity,
                            )
                        )
                        if rate_limit_remaining == 0:
                            await self.rate_limiter.async_adjust_tokens(
                                rate_limit_remaining
                            )
                        else:
                            await asyncio.sleep(backoff_timeout(wx_delay_time, attempt))
                        await self.rate_limiter.acquire_async()
                    else:
                        await asyncio.sleep(backoff_timeout(wx_delay_time, attempt))
                else:
                    break

            return response  # type:ignore[return-value]

        return wrapper

    return decorator


def _with_async_retry_stream(
    max_retries: int = _MAX_RETRIES,
    delay_time: float = DEFAULT_DELAY,
    retry_status_codes: list[int] = DEFAULT_RETRY_STATUS_CODES,
) -> Callable:
    """Async decorator to retry the streaming function if it encounters a HTTP status code from `retry_status_codes` or env variable"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @asynccontextmanager
        async def wrapper(
            self: Any, *args: Any, **kwargs: Any
        ) -> AsyncIterator[httpx.Response]:
            wx_max_retries = _get_max_retries(self.max_retries, max_retries)
            wx_delay_time = _get_delay_time(self.delay_time, delay_time)
            wx_retry_status_codes = _get_retry_status_codes(
                self.retry_status_codes, retry_status_codes
            )

            response: httpx.Response | None = None
            for attempt in range(wx_max_retries + 1):
                if response is not None:
                    await response.aclose()

                async with func(self, *args, **kwargs) as response:
                    if response is not None and (
                        response.status_code in wx_retry_status_codes
                        and attempt != wx_max_retries
                    ):
                        #  If the environment is set to cloud, the Token Bucket (rate_limiter here) is used to control traffic flow.

                        if self._client.CLOUD_PLATFORM_SPACES:
                            rate_limit_remaining = int(
                                response.headers.get(
                                    REMAINING_LIMIT_HEADER,
                                    self.rate_limiter.capacity,
                                )
                            )
                            if rate_limit_remaining == 0:
                                await self.rate_limiter.async_adjust_tokens(
                                    rate_limit_remaining
                                )
                            else:
                                await asyncio.sleep(
                                    backoff_timeout(wx_delay_time, attempt)
                                )
                            await self.rate_limiter.acquire_async()
                        else:  # If CDP, don't use Token Bucket
                            await asyncio.sleep(backoff_timeout(wx_delay_time, attempt))
                        continue
                    if response is not None:
                        yield response
                    break

        return wrapper

    return decorator


class TokenBucket:
    """Thread-safe rate limiter with dynamic token adjustments."""

    def __init__(self, rate: float, capacity: int) -> None:
        self.capacity = capacity  # Max tokens
        self.rate = rate  # Tokens per second
        self.tokens: float = capacity  # Start full
        self.lock = threading.Lock()
        self.last_refill = time.time()
        self.condition_lock = threading.Condition(self.lock)
        self.async_lock = asyncio.Lock()
        self.waiting_threads: queue.Queue[int] = queue.Queue()

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        if new_tokens >= 1:  # Only update if at least one token is added
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now

    def acquire(self) -> None:
        """Wait for a token and process threads in correct order."""
        thread_id = threading.get_ident()

        with self.condition_lock:
            # Add to queue if not already in front
            if (
                self.waiting_threads.empty()
                or self.waiting_threads.queue[-1] != thread_id
            ):
                self.waiting_threads.put(thread_id)

            while True:
                self.refill()

                # Allow thread to proceed only if it's at the front of the queue and tokens are available
                if self.tokens >= 1 and self.waiting_threads.queue[0] == thread_id:
                    self.waiting_threads.get()  # Remove from queue
                    self.tokens -= 1  # Consume token
                    self.condition_lock.notify()  # Wake next in line
                    return

                # Wait only until the next expected refill time
                next_refill = self.last_refill + (1 / self.rate)
                wait_time_float = max(0.0, next_refill - time.time())
                self.condition_lock.wait(wait_time_float)

    async def acquire_async(self) -> None:
        """Asynchronous acquire: Wait until a token is available."""
        async with self.async_lock:
            while self.tokens < 1:
                self.refill()
                wait_time = (1 / self.rate) if self.tokens < 1 else 0
                await asyncio.sleep(wait_time)
            self.tokens -= 1

    def adjust_tokens(self, remaining_tokens: int) -> None:
        """Adjust token count based on RateLimit-Remaining."""
        with self.lock:
            self.tokens = min(self.capacity, remaining_tokens)

    async def async_adjust_tokens(self, remaining_tokens: int) -> None:
        """Adjust token count based on RateLimit-Remaining."""
        async with self.async_lock:
            self.tokens = min(self.capacity, remaining_tokens)


class RetryTransport(httpx.HTTPTransport):
    def __init__(
        self,
        retries: int,
        backoff_factor: float,
        status_forcelist: Iterable[int],
        verify_initial: bool | str | None,
        allow_ssl_fallback: bool,
        **kwargs: Any,
    ) -> None:
        verify_for_transport = True if verify_initial is None else verify_initial
        # Get proxies from additional_settings if available
        self.proxies = additional_settings.get("proxies")
        if self.proxies:
            proxy_url = self.proxies.get("https") or self.proxies.get("http")
            if proxy_url:
                kwargs["proxy"] = httpx.Proxy(proxy_url)
        super().__init__(verify=verify_for_transport, **kwargs)
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.status_forcelist = status_forcelist
        self.allow_ssl_fallback = allow_ssl_fallback
        self.original_verify = verify_initial
        self._ssl_fallback_attempted = False
        self._effective_verify: bool | str | None = None

    def get_effective_verify_for_client(self) -> bool | str:
        """Get the effective verify value for the HTTP client."""
        if self._effective_verify is None:
            effective = _get_effective_verify()
            self._effective_verify = effective
            return effective
        return self._effective_verify

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = None
        effective_verify = self.get_effective_verify_for_client()

        for attempt in range(self.retries + 1):
            if response is not None:
                response.close()

            try:
                response = super().handle_request(request)

            except (
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ) as e:
                error_str = str(e)
                is_ssl_error = any(
                    ssl_keyword in error_str
                    for ssl_keyword in [
                        "CERTIFICATE_VERIFY_FAILED",
                        "certificate verify failed",
                        "SSL",
                        "TLS",
                        "self-signed certificate",
                    ]
                )

                # Retry on server disconnect (stale keep-alive connection) - but not SSL errors
                if isinstance(e, httpx.RemoteProtocolError) and not is_ssl_error:
                    if attempt < self.retries:
                        continue
                    raise

                if (
                    is_ssl_error
                    and not self._ssl_fallback_attempted
                    and self.original_verify is None
                    and self.allow_ssl_fallback
                ):
                    self._ssl_fallback_attempted = True

                    # Update global verify to False for SSL fallback
                    global verify
                    verify = False

                    self.close()
                    RetryTransport.__init__(
                        self,
                        retries=self.retries,
                        backoff_factor=self.backoff_factor,
                        status_forcelist=self.status_forcelist,
                        verify_initial=False,
                        allow_ssl_fallback=self.allow_ssl_fallback,
                    )
                    response = super().handle_request(request)
                elif is_ssl_error and (
                    effective_verify is True or isinstance(effective_verify, str)
                ):
                    # When verify is explicitly set to True or a path to CA bundle
                    _raise_verify_error(e)
                else:
                    # If proxies are configured, and we get a connection error,
                    # raise httpx.ProxyError
                    if self.proxies and isinstance(
                        e, (httpx.ConnectError, httpx.ConnectTimeout)
                    ):
                        raise httpx.ProxyError(str(e)) from e
                    raise

            if (
                response is not None
                and response.status_code in self.status_forcelist
                and attempt != self.retries
            ):
                sleep_time = min(self.backoff_factor * (2**attempt), self.retries)
                time.sleep(sleep_time)
            else:
                break

        return response  # type:ignore[return-value]


class AsyncRetryTransport(httpx.AsyncHTTPTransport):
    def __init__(
        self,
        retries: int,
        backoff_factor: float,
        status_forcelist: Iterable[int],
        verify_initial: bool | str | None,
        allow_ssl_fallback: bool,
        **kwargs: Any,
    ) -> None:
        verify_for_transport = True if verify_initial is None else verify_initial
        # Get proxies from additional_settings if available
        self.proxies = additional_settings.get("proxies")
        if self.proxies:
            proxy_url = self.proxies.get("https") or self.proxies.get("http")
            if proxy_url:
                kwargs["proxy"] = httpx.Proxy(proxy_url)
        super().__init__(verify=verify_for_transport, **kwargs)
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.status_forcelist = status_forcelist
        self.allow_ssl_fallback = allow_ssl_fallback
        self.original_verify = verify_initial
        self._ssl_fallback_attempted = False
        self._effective_verify: bool | str | None = None  # Will be computed when needed

    def get_effective_verify_for_client(self) -> bool | str:
        """Get the effective verify value for the HTTP client."""
        if self._effective_verify is None:
            effective = _get_effective_verify()
            self._effective_verify = effective
            return effective
        return self._effective_verify

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = None
        effective_verify = self.get_effective_verify_for_client()

        for attempt in range(self.retries + 1):
            if response is not None:
                await response.aclose()

            try:
                response = await super().handle_async_request(request)

            except (
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ) as e:
                error_str = str(e)
                is_ssl_error = any(
                    ssl_keyword in error_str
                    for ssl_keyword in [
                        "CERTIFICATE_VERIFY_FAILED",
                        "certificate verify failed",
                        "SSL",
                        "TLS",
                        "self-signed certificate",
                    ]
                )

                # Retry on server disconnect (stale keep-alive connection) - but not SSL errors
                if isinstance(e, httpx.RemoteProtocolError) and not is_ssl_error:
                    if attempt < self.retries:
                        continue
                    raise

                if (
                    is_ssl_error
                    and not self._ssl_fallback_attempted
                    and self.original_verify is None
                    and self.allow_ssl_fallback
                ):
                    self._ssl_fallback_attempted = True

                    # Update global verify to False for SSL fallback
                    global verify
                    verify = False

                    await self.aclose()
                    AsyncRetryTransport.__init__(
                        self,
                        retries=self.retries,
                        backoff_factor=self.backoff_factor,
                        status_forcelist=self.status_forcelist,
                        verify_initial=False,
                        allow_ssl_fallback=self.allow_ssl_fallback,
                    )
                    response = await super().handle_async_request(request)
                elif is_ssl_error and (
                    effective_verify is True or isinstance(effective_verify, str)
                ):
                    # When verify is explicitly set to True or a path to CA bundle
                    _raise_verify_error(e)
                else:
                    # If proxies are configured, and we get a connection error,
                    # raise httpx.ProxyError
                    if self.proxies and isinstance(
                        e, (httpx.ConnectError, httpx.ConnectTimeout)
                    ):
                        raise httpx.ProxyError(str(e)) from e
                    raise

            if (
                response is not None
                and response.status_code in self.status_forcelist
                and attempt != self.retries
            ):
                sleep_time = min(self.backoff_factor * (2**attempt), self.retries)
                await asyncio.sleep(sleep_time)
            else:
                break

        return response  # type:ignore[return-value]

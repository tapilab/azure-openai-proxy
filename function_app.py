import json
import os
import time
import uuid
import random
import logging
import traceback
import datetime
import email.utils
from contextlib import ExitStack
from typing import Any, Dict, Optional, Tuple, Iterable, Set

import azure.functions as func
import httpx
from azure.identity import DefaultAzureCredential

# HTTP streaming for Python in Azure Functions requires the FastAPI extension types.
# See Microsoft docs on "HTTP streams" for Python v2 programming model.
from azurefunctions.extensions.http.fastapi import Request, JSONResponse, StreamingResponse


# ----------------------------
# Azure Functions app
# ----------------------------

app = func.FunctionApp()  # Auth level is set per route to match your current style.


# ----------------------------
# Required environment variables
# ----------------------------

BASE = os.environ["AZURE_OPENAI_BASE"].rstrip("/")
DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]

API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
V1_API_VERSION = os.getenv("AZURE_OPENAI_V1_API_VERSION", "preview")

DEFAULT_MODEL_ALIAS = os.getenv("AZURE_OPENAI_DEFAULT_MODEL_ALIAS")
MODEL_MAP = json.loads(os.getenv("AZURE_OPENAI_MODEL_MAP", "{}"))

SCOPE = "https://cognitiveservices.azure.com/.default"


# ----------------------------
# Helper: env parsing
# ----------------------------

def _getenv_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)

def _getenv_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)

def _getenv_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


# ----------------------------
# Retry configuration (env-configurable)
# ----------------------------

RETRYABLE_STATUS_CODES: Set[int] = {408, 409, 425, 429, 500, 502, 503, 504}

MAX_RETRIES = _getenv_int("PROXY_MAX_RETRIES", 6)

RETRY_BASE_DELAY_SECONDS = _getenv_float("PROXY_RETRY_BASE_DELAY_SECONDS", 0.8)
RETRY_MAX_DELAY_SECONDS = _getenv_float("PROXY_RETRY_MAX_DELAY_SECONDS", 30.0)
RETRY_JITTER_SECONDS = _getenv_float("PROXY_RETRY_JITTER_SECONDS", 0.5)

RESPECT_RETRY_AFTER = _getenv_bool("PROXY_RESPECT_RETRY_AFTER", True)
RETRY_AFTER_MAX_SECONDS = _getenv_float("PROXY_RETRY_AFTER_MAX_SECONDS", 30.0)

# Caps for how long we sleep across retries, and how long we allow the overall operation to run.
# These defaults are chosen to stay under Azure Functions HTTP trigger ~230s boundary.
RETRY_TOTAL_SLEEP_BUDGET_SECONDS = _getenv_float("PROXY_RETRY_TOTAL_SLEEP_BUDGET_SECONDS", 60.0)
OPERATION_DEADLINE_SECONDS = _getenv_float("PROXY_OPERATION_DEADLINE_SECONDS", 220.0)


# ----------------------------
# Timeout configuration (httpx semantics: connect/read/write/pool)
# ----------------------------

TIMEOUT_GROWTH_FACTOR = _getenv_float("PROXY_TIMEOUT_GROWTH_FACTOR", 1.6)

CONNECT_TIMEOUT_BASE_SECONDS = _getenv_float("PROXY_CONNECT_TIMEOUT_BASE_SECONDS", 20.0)
READ_TIMEOUT_BASE_SECONDS = _getenv_float("PROXY_READ_TIMEOUT_BASE_SECONDS", 120.0)
WRITE_TIMEOUT_BASE_SECONDS = _getenv_float("PROXY_WRITE_TIMEOUT_BASE_SECONDS", 60.0)
POOL_TIMEOUT_BASE_SECONDS = _getenv_float("PROXY_POOL_TIMEOUT_BASE_SECONDS", 20.0)

# Hard caps to avoid exceeding the HTTP trigger boundary.
CONNECT_TIMEOUT_MAX_SECONDS = _getenv_float("PROXY_CONNECT_TIMEOUT_MAX_SECONDS", 60.0)
READ_TIMEOUT_MAX_SECONDS = _getenv_float("PROXY_READ_TIMEOUT_MAX_SECONDS", 225.0)
WRITE_TIMEOUT_MAX_SECONDS = _getenv_float("PROXY_WRITE_TIMEOUT_MAX_SECONDS", 120.0)
POOL_TIMEOUT_MAX_SECONDS = _getenv_float("PROXY_POOL_TIMEOUT_MAX_SECONDS", 60.0)


# ----------------------------
# httpx connection pool configuration
# ----------------------------

HTTPX_MAX_CONNECTIONS = _getenv_int("PROXY_HTTPX_MAX_CONNECTIONS", 200)
HTTPX_MAX_KEEPALIVE_CONNECTIONS = _getenv_int("PROXY_HTTPX_MAX_KEEPALIVE_CONNECTIONS", 50)
HTTPX_KEEPALIVE_EXPIRY_SECONDS = _getenv_float("PROXY_HTTPX_KEEPALIVE_EXPIRY_SECONDS", 30.0)

HTTPX_HTTP2 = _getenv_bool("PROXY_HTTPX_HTTP2", False)
HTTPX_VERIFY_TLS = _getenv_bool("PROXY_HTTPX_VERIFY_TLS", True)

DIAGNOSTIC_HEADERS_ENABLED = _getenv_bool("PROXY_DIAGNOSTIC_HEADERS_ENABLED", True)
LOG_BACKEND_ERROR_BODY = _getenv_bool("PROXY_LOG_BACKEND_ERROR_BODY", True)


# ----------------------------
# Shared credential + shared httpx client (reuse strongly recommended)
# ----------------------------

# Reuse credential instance to benefit from token caching and reduce auth requests.
_AZURE_CREDENTIAL = DefaultAzureCredential()

_HTTP_LIMITS = httpx.Limits(
    max_connections=HTTPX_MAX_CONNECTIONS,
    max_keepalive_connections=HTTPX_MAX_KEEPALIVE_CONNECTIONS,
    keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY_SECONDS,
)

_HTTP_CLIENT = httpx.Client(
    limits=_HTTP_LIMITS,
    http2=HTTPX_HTTP2,
    verify=HTTPX_VERIFY_TLS,
    follow_redirects=False,
)


# ----------------------------
# Small utilities
# ----------------------------

def _safe_str(value: Any, limit: int = 4000) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    if len(s) <= limit:
        return s
    return s[:limit] + "...(truncated)"

def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)

def _remaining_deadline_seconds(start_monotonic: float) -> float:
    if OPERATION_DEADLINE_SECONDS <= 0:
        return float("inf")
    return max(0.0, OPERATION_DEADLINE_SECONDS - (time.monotonic() - start_monotonic))

def _build_timeout_for_attempt(attempt: int) -> httpx.Timeout:
    factor = TIMEOUT_GROWTH_FACTOR ** max(0, attempt)

    def grow(base: float, cap: float) -> float:
        return min(base * factor, cap)

    return httpx.Timeout(
        connect=grow(CONNECT_TIMEOUT_BASE_SECONDS, CONNECT_TIMEOUT_MAX_SECONDS),
        read=grow(READ_TIMEOUT_BASE_SECONDS, READ_TIMEOUT_MAX_SECONDS),
        write=grow(WRITE_TIMEOUT_BASE_SECONDS, WRITE_TIMEOUT_MAX_SECONDS),
        pool=grow(POOL_TIMEOUT_BASE_SECONDS, POOL_TIMEOUT_MAX_SECONDS),
    )

def _should_retry_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES

def _should_stream(payload: Dict[str, Any]) -> bool:
    return bool(payload.get("stream"))

def _parse_retry_after_seconds(headers: httpx.Headers) -> Optional[float]:
    raw = headers.get("retry-after")
    if not raw:
        return None
    raw = raw.strip()

    # 1) delta-seconds
    try:
        seconds = int(raw)
        return max(0.0, float(seconds))
    except Exception:
        pass

    # 2) HTTP date
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return max(0.0, (dt - _now_utc()).total_seconds())
    except Exception:
        return None

def _compute_backoff_seconds(attempt: int, retry_after_seconds: Optional[float]) -> float:
    if retry_after_seconds is not None and RESPECT_RETRY_AFTER:
        sleep_s = min(float(retry_after_seconds), RETRY_AFTER_MAX_SECONDS)
        jitter = random.uniform(0.0, RETRY_JITTER_SECONDS) if RETRY_JITTER_SECONDS > 0 else 0.0
        return sleep_s + jitter

    base = RETRY_BASE_DELAY_SECONDS * (2 ** max(0, attempt))
    sleep_s = min(base, RETRY_MAX_DELAY_SECONDS)
    if RETRY_JITTER_SECONDS > 0:
        sleep_s += random.uniform(0.0, RETRY_JITTER_SECONDS)
    return sleep_s

def _can_sleep(start_monotonic: float, slept_total: float, sleep_seconds: float) -> bool:
    if sleep_seconds <= 0:
        return True
    if RETRY_TOTAL_SLEEP_BUDGET_SECONDS > 0 and (slept_total + sleep_seconds) > RETRY_TOTAL_SLEEP_BUDGET_SECONDS:
        return False
    remaining = _remaining_deadline_seconds(start_monotonic)
    if remaining <= 0 or sleep_seconds >= remaining:
        return False
    return True

def _get_token(request_id: str) -> str:
    try:
        return _AZURE_CREDENTIAL.get_token(SCOPE).token
    except Exception as e:
        logging.exception("Token acquisition failed request_id=%s", request_id)
        raise RuntimeError("TokenAcquisitionError") from e

def _proxy_headers(token: str, request_id: str) -> Dict[str, str]:
    # Keep request headers minimal to avoid custom header limits and reduce variability.
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-ms-client-request-id": request_id,
    }

def _diagnostic_headers(
    request_id: str,
    deployment: Optional[str],
    backend_status: Optional[int],
    attempt: Optional[int],
    stream_mode: bool,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    if not DIAGNOSTIC_HEADERS_ENABLED:
        return {}
    h: Dict[str, str] = {
        "x-proxy-request-id": request_id,
        "x-proxy-deployment": deployment or "",
        "x-proxy-backend-status": "" if backend_status is None else str(backend_status),
        "x-proxy-retry-attempt": "" if attempt is None else str(attempt),
        "x-proxy-stream": "1" if stream_mode else "0",
    }
    if extra:
        h.update(extra)
    return h

def _resolve_deployment(payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    requested_model = payload.get("model")

    if not MODEL_MAP:
        return DEPLOYMENT, requested_model

    if requested_model in MODEL_MAP:
        return str(MODEL_MAP[requested_model]), requested_model

    if requested_model and requested_model not in MODEL_MAP:
        raise ValueError(f"Unsupported model '{requested_model}'")

    if DEFAULT_MODEL_ALIAS and DEFAULT_MODEL_ALIAS in MODEL_MAP:
        return str(MODEL_MAP[DEFAULT_MODEL_ALIAS]), DEFAULT_MODEL_ALIAS

    return DEPLOYMENT, requested_model

def _map_httpx_exception(e: Exception) -> Tuple[str, int]:
    # Map to proxy-facing error type & HTTP status.
    if isinstance(e, httpx.ConnectTimeout):
        return "ConnectTimeout", 504
    if isinstance(e, httpx.ReadTimeout):
        return "ReadTimeout", 504
    if isinstance(e, httpx.WriteTimeout):
        return "WriteTimeout", 504
    if isinstance(e, httpx.PoolTimeout):
        return "PoolTimeout", 503
    if isinstance(e, httpx.TimeoutException):
        return "TimeoutException", 504
    if isinstance(e, httpx.ConnectError):
        return "ConnectError", 502
    if isinstance(e, httpx.NetworkError):
        return "NetworkError", 502
    if isinstance(e, httpx.ProtocolError):
        return "ProtocolError", 502
    if isinstance(e, httpx.ProxyError):
        return "ProxyError", 502
    if isinstance(e, httpx.RequestError):
        return "RequestError", 502
    return "UnexpectedProxyError", 500

def _error_json(
    request_id: str,
    error_type: str,
    message: str,
    stage: str,
    status_code: int,
    requested_model: Optional[str],
    deployment: Optional[str],
    backend_url: Optional[str],
    backend_status_code: Optional[int] = None,
    backend_response_text: Optional[str] = None,
    retry_attempt: Optional[int] = None,
    exception: Optional[Exception] = None,
) -> JSONResponse:
    body: Dict[str, Any] = {
        "ok": False,
        "proxy_request_id": request_id,
        "error_type": error_type,
        "error_stage": stage,
        "message": message,
        "requested_model": requested_model,
        "deployment": deployment,
        "backend_url": backend_url,
        "backend_status_code": backend_status_code,
        "backend_response_text": _safe_str(backend_response_text),
        "retry_attempt": retry_attempt,
        "max_retries": MAX_RETRIES,
    }
    if exception is not None:
        body["exception_class"] = exception.__class__.__name__
        body["exception_message"] = _safe_str(exception)
        body["traceback"] = traceback.format_exc()

    logging.error("Proxy error request_id=%s body=%s", request_id, json.dumps(body, ensure_ascii=False))
    return JSONResponse(
        content=body,
        status_code=status_code,
        headers=_diagnostic_headers(
            request_id=request_id,
            deployment=deployment,
            backend_status=backend_status_code,
            attempt=retry_attempt,
            stream_mode=False,
        ),
    )


# ----------------------------
# Non-stream forwarding (JSON)
# ----------------------------

def _forward_non_stream(
    request_id: str,
    payload: Dict[str, Any],
    backend_url: str,
    params: Dict[str, str],
    requested_model: Optional[str],
    deployment: Optional[str],
) -> JSONResponse:
    start = time.monotonic()
    slept_total = 0.0

    try:
        token = _get_token(request_id)
    except Exception as e:
        return _error_json(
            request_id=request_id,
            error_type="TokenAcquisitionError",
            message="Failed to acquire Azure credential token.",
            stage="credential",
            status_code=500,
            requested_model=requested_model,
            deployment=deployment,
            backend_url=backend_url,
            exception=e,
        )

    headers = _proxy_headers(token, request_id)
    last_backend_text: Optional[str] = None
    last_status: Optional[int] = None
    last_exception: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        timeout = _build_timeout_for_attempt(attempt)

        try:
            r = _HTTP_CLIENT.post(
                backend_url,
                params=params,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            last_status = r.status_code

            if _should_retry_status(r.status_code) and attempt < MAX_RETRIES:
                if LOG_BACKEND_ERROR_BODY:
                    try:
                        last_backend_text = r.text
                    except Exception:
                        last_backend_text = None

                retry_after = _parse_retry_after_seconds(r.headers)
                sleep_seconds = _compute_backoff_seconds(attempt, retry_after)

                logging.warning(
                    "Retryable status=%s attempt=%s sleep=%.3fs request_id=%s",
                    r.status_code, attempt, sleep_seconds, request_id
                )

                if not _can_sleep(start, slept_total, sleep_seconds):
                    break

                time.sleep(sleep_seconds)
                slept_total += sleep_seconds
                continue

            # Return response as JSON (Azure OpenAI responses are JSON).
            try:
                body_obj = r.json()
            except Exception:
                body_obj = {"raw": _safe_str(r.text)}

            return JSONResponse(
                content=body_obj,
                status_code=r.status_code,
                headers=_diagnostic_headers(
                    request_id=request_id,
                    deployment=deployment,
                    backend_status=r.status_code,
                    attempt=attempt,
                    stream_mode=False,
                ),
            )

        except Exception as e:
            last_exception = e
            error_type, mapped_status = _map_httpx_exception(e)

            if attempt < MAX_RETRIES:
                sleep_seconds = _compute_backoff_seconds(attempt, None)
                logging.warning(
                    "Retryable exception=%s attempt=%s sleep=%.3fs request_id=%s",
                    error_type, attempt, sleep_seconds, request_id
                )

                if not _can_sleep(start, slept_total, sleep_seconds):
                    break

                time.sleep(sleep_seconds)
                slept_total += sleep_seconds
                continue

            return _error_json(
                request_id=request_id,
                error_type=error_type,
                message="Proxy call failed (non-stream).",
                stage="proxy_to_backend",
                status_code=mapped_status,
                requested_model=requested_model,
                deployment=deployment,
                backend_url=backend_url,
                backend_status_code=last_status,
                backend_response_text=last_backend_text,
                retry_attempt=attempt,
                exception=e,
            )

    # Stopped due to budgets/deadline (avoid hitting ~230s boundary)
    return _error_json(
        request_id=request_id,
        error_type="RetryBudgetExhausted",
        message="Stopped retrying due to retry sleep budget or operation deadline.",
        stage="proxy_to_backend",
        status_code=502,
        requested_model=requested_model,
        deployment=deployment,
        backend_url=backend_url,
        backend_status_code=last_status,
        backend_response_text=last_backend_text,
        retry_attempt=MAX_RETRIES,
        exception=last_exception,
    )


# ----------------------------
# Stream forwarding (true chunked proxy)
# ----------------------------

def _prepare_backend_stream_with_retries(
    request_id: str,
    payload: Dict[str, Any],
    backend_url: str,
    params: Dict[str, str],
    headers: Dict[str, str],
) -> Tuple[httpx.Response, ExitStack, int]:
    """
    Open the backend stream with retries BEFORE sending any bytes to the client.
    This ensures we can retry safely without corrupting client stream.
    """
    start = time.monotonic()
    slept_total = 0.0
    last_exception: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        timeout = _build_timeout_for_attempt(attempt)
        stack = ExitStack()

        try:
            r = stack.enter_context(
                _HTTP_CLIENT.stream(
                    "POST",
                    backend_url,
                    params=params,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
            )

            if _should_retry_status(r.status_code) and attempt < MAX_RETRIES:
                # Drain and close before retrying.
                try:
                    if LOG_BACKEND_ERROR_BODY:
                        _ = r.text  # best-effort read for logging/debug
                except Exception:
                    pass
                try:
                    r.read()
                except Exception:
                    pass
                stack.close()

                retry_after = _parse_retry_after_seconds(r.headers)
                sleep_seconds = _compute_backoff_seconds(attempt, retry_after)

                logging.warning(
                    "Retryable stream status=%s attempt=%s sleep=%.3fs request_id=%s",
                    r.status_code, attempt, sleep_seconds, request_id
                )

                if not _can_sleep(start, slept_total, sleep_seconds):
                    break

                time.sleep(sleep_seconds)
                slept_total += sleep_seconds
                continue

            return r, stack, attempt

        except Exception as e:
            last_exception = e
            stack.close()

            if attempt < MAX_RETRIES:
                sleep_seconds = _compute_backoff_seconds(attempt, None)
                logging.warning(
                    "Retryable stream exception=%s attempt=%s sleep=%.3fs request_id=%s",
                    e.__class__.__name__, attempt, sleep_seconds, request_id
                )

                if not _can_sleep(start, slept_total, sleep_seconds):
                    break

                time.sleep(sleep_seconds)
                slept_total += sleep_seconds
                continue

            raise

    raise RuntimeError("RetryBudgetExhausted") from last_exception

def _forward_stream(
    request_id: str,
    payload: Dict[str, Any],
    backend_url: str,
    params: Dict[str, str],
    requested_model: Optional[str],
    deployment: Optional[str],
) -> Any:
    try:
        token = _get_token(request_id)
    except Exception as e:
        return _error_json(
            request_id=request_id,
            error_type="TokenAcquisitionError",
            message="Failed to acquire Azure credential token.",
            stage="credential",
            status_code=500,
            requested_model=requested_model,
            deployment=deployment,
            backend_url=backend_url,
            exception=e,
        )

    headers = _proxy_headers(token, request_id)

    try:
        r, stack, attempt = _prepare_backend_stream_with_retries(
            request_id=request_id,
            payload=payload,
            backend_url=backend_url,
            params=params,
            headers=headers,
        )
    except Exception as e:
        error_type, status_code = _map_httpx_exception(e)
        if isinstance(e, RuntimeError) and str(e) == "RetryBudgetExhausted":
            error_type, status_code = "RetryBudgetExhausted", 502

        return _error_json(
            request_id=request_id,
            error_type=error_type,
            message="Proxy failed (stream) before sending any bytes.",
            stage="proxy_to_backend_stream",
            status_code=status_code,
            requested_model=requested_model,
            deployment=deployment,
            backend_url=backend_url,
            retry_attempt=MAX_RETRIES,
            exception=e,
        )

    content_type = r.headers.get("content-type", "text/event-stream")

    # Pass through a small set of useful hop-by-hop safe headers (optional, conservative).
    passthrough_headers: Dict[str, str] = {}
    for key in ("cache-control", "pragma", "content-encoding"):
        v = r.headers.get(key)
        if v:
            passthrough_headers[key] = v

    diag_headers = _diagnostic_headers(
        request_id=request_id,
        deployment=deployment,
        backend_status=r.status_code,
        attempt=attempt,
        stream_mode=True,
        extra=passthrough_headers,
    )

    def iterator() -> Iterable[bytes]:
        """
        Stream raw chunks edge-to-edge.
        IMPORTANT: After yielding begins, do NOT retry, or you'll corrupt the client stream.
        """
        try:
            for chunk in r.iter_raw():
                yield chunk
        except Exception:
            logging.exception("Streaming forward error request_id=%s", request_id)
        finally:
            try:
                stack.close()
            except Exception:
                pass

    return StreamingResponse(
        iterator(),
        media_type=content_type,
        status_code=r.status_code,
        headers=diag_headers,
    )


# ----------------------------
# Unified forward entry
# ----------------------------

def _forward_request(
    request_id: str,
    payload: Dict[str, Any],
    backend_url: str,
    params: Dict[str, str],
    requested_model: Optional[str],
    deployment: Optional[str],
) -> Any:
    if _should_stream(payload):
        return _forward_stream(
            request_id=request_id,
            payload=payload,
            backend_url=backend_url,
            params=params,
            requested_model=requested_model,
            deployment=deployment,
        )
    return _forward_non_stream(
        request_id=request_id,
        payload=payload,
        backend_url=backend_url,
        params=params,
        requested_model=requested_model,
        deployment=deployment,
    )


# ----------------------------
# Routes
# ----------------------------

@app.route(route="v1/chat/completions", methods=[func.HttpMethod.POST], auth_level=func.AuthLevel.FUNCTION)
async def chat_completions(req: Request) -> Any:
    request_id = str(uuid.uuid4())
    start = time.monotonic()

    try:
        payload = await req.json()
        if not isinstance(payload, dict):
            raise ValueError("Request JSON body must be an object.")
    except Exception as e:
        return _error_json(
            request_id=request_id,
            error_type="InvalidJSON",
            message="Request body is not valid JSON object.",
            stage="request_parse",
            status_code=400,
            requested_model=None,
            deployment=None,
            backend_url=None,
            exception=e,
        )

    try:
        deployment, requested_model = _resolve_deployment(payload)
    except Exception as e:
        return _error_json(
            request_id=request_id,
            error_type="ModelResolutionError",
            message=str(e),
            stage="model_resolution",
            status_code=400,
            requested_model=payload.get("model"),
            deployment=None,
            backend_url=None,
            exception=e,
        )

    if requested_model:
        payload["model"] = requested_model

    backend_url = f"{BASE}/openai/deployments/{deployment}/chat/completions"
    params = {"api-version": API_VERSION}

    resp = _forward_request(
        request_id=request_id,
        payload=payload,
        backend_url=backend_url,
        params=params,
        requested_model=requested_model,
        deployment=deployment,
    )

    logging.info(
        "chat_completions done request_id=%s stream=%s duration=%.3fs",
        request_id, "1" if _should_stream(payload) else "0", time.monotonic() - start
    )
    return resp

@app.route(route="v1/responses", methods=[func.HttpMethod.POST], auth_level=func.AuthLevel.FUNCTION)
async def responses(req: Request) -> Any:
    request_id = str(uuid.uuid4())
    start = time.monotonic()

    try:
        payload = await req.json()
        if not isinstance(payload, dict):
            raise ValueError("Request JSON body must be an object.")
    except Exception as e:
        return _error_json(
            request_id=request_id,
            error_type="InvalidJSON",
            message="Request body is not valid JSON object.",
            stage="request_parse",
            status_code=400,
            requested_model=None,
            deployment=None,
            backend_url=None,
            exception=e,
        )

    try:
        deployment, requested_model = _resolve_deployment(payload)
    except Exception as e:
        return _error_json(
            request_id=request_id,
            error_type="ModelResolutionError",
            message=str(e),
            stage="model_resolution",
            status_code=400,
            requested_model=payload.get("model"),
            deployment=None,
            backend_url=None,
            exception=e,
        )

    # v1 Responses API expects deployed model name (deployment) as "model"
    payload["model"] = deployment

    backend_url = f"{BASE}/openai/v1/responses"
    params = {"api-version": V1_API_VERSION}

    resp = _forward_request(
        request_id=request_id,
        payload=payload,
        backend_url=backend_url,
        params=params,
        requested_model=requested_model,
        deployment=deployment,
    )

    logging.info(
        "responses done request_id=%s stream=%s duration=%.3fs",
        request_id, "1" if _should_stream(payload) else "0", time.monotonic() - start
    )
    return resp

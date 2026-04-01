import json
import os
import time
import uuid
import traceback
import logging
import threading
from pathlib import Path

import azure.functions as func
import httpx
from azure.identity import DefaultAzureCredential

app = func.FunctionApp()

BASE = os.environ["AZURE_OPENAI_BASE"].rstrip("/")
DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
V1_API_VERSION = os.getenv("AZURE_OPENAI_V1_API_VERSION", "preview")
MODEL_MAP = json.loads(os.getenv("AZURE_OPENAI_MODEL_MAP", "{}"))
DEFAULT_MODEL_ALIAS = os.getenv("AZURE_OPENAI_DEFAULT_MODEL_ALIAS")

SCOPE = "https://cognitiveservices.azure.com/.default"

# Retry and timeout settings
MAX_RETRIES = int(os.getenv("PROXY_MAX_RETRIES", "2"))
RETRY_BACKOFF_SECONDS = float(os.getenv("PROXY_RETRY_BACKOFF_SECONDS", "1.5"))

CONNECT_TIMEOUT_SECONDS = float(os.getenv("PROXY_CONNECT_TIMEOUT_SECONDS", "20"))
READ_TIMEOUT_SECONDS = float(os.getenv("PROXY_READ_TIMEOUT_SECONDS", "120"))
WRITE_TIMEOUT_SECONDS = float(os.getenv("PROXY_WRITE_TIMEOUT_SECONDS", "60"))
POOL_TIMEOUT_SECONDS = float(os.getenv("PROXY_POOL_TIMEOUT_SECONDS", "20"))

# Async fallback settings
ASYNC_FALLBACK_ENABLED = os.getenv("PROXY_ASYNC_FALLBACK_ENABLED", "true").lower() == "true"
ASYNC_FALLBACK_POLL_AFTER_SECONDS = int(os.getenv("PROXY_ASYNC_FALLBACK_POLL_AFTER_SECONDS", "30"))
ASYNC_JOB_TTL_SECONDS = int(os.getenv("PROXY_ASYNC_JOB_TTL_SECONDS", "86400"))
ASYNC_JOB_DIR = Path(os.getenv("PROXY_ASYNC_JOB_DIR", "/tmp/azure-openai-proxy-jobs"))
ASYNC_FALLBACK_STATUS_CODES = {429, 500, 502, 503, 504}
ASYNC_FALLBACK_ERROR_TYPES = {
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "ConnectError",
    "RequestError",
    "RetryExhausted",
}

ASYNC_JOB_DIR.mkdir(parents=True, exist_ok=True)
_JOB_LOCK = threading.Lock()


def _build_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT_SECONDS,
        read=READ_TIMEOUT_SECONDS,
        write=WRITE_TIMEOUT_SECONDS,
        pool=POOL_TIMEOUT_SECONDS,
    )


def _json_response(body: dict, status_code: int, headers: dict | None = None) -> func.HttpResponse:
    response_headers = headers or {}
    return func.HttpResponse(
        json.dumps(body, ensure_ascii=False),
        status_code=status_code,
        mimetype="application/json",
        headers=response_headers,
    )


def _safe_str(value, limit: int = 4000) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _build_error_body(
    *,
    request_id: str,
    error_type: str,
    message: str,
    stage: str,
    requested_model: str | None = None,
    deployment: str | None = None,
    backend_url: str | None = None,
    backend_status_code: int | None = None,
    backend_response_text: str | None = None,
    retry_attempt: int | None = None,
    max_retries: int | None = None,
    exception: Exception | None = None,
) -> dict:
    body = {
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
        "max_retries": max_retries,
    }

    if exception is not None:
        body["exception_class"] = exception.__class__.__name__
        body["exception_message"] = _safe_str(exception)
        body["traceback"] = traceback.format_exc()

    return body


def _error_response(
    *,
    request_id: str,
    error_type: str,
    message: str,
    status_code: int,
    stage: str,
    requested_model: str | None = None,
    deployment: str | None = None,
    backend_url: str | None = None,
    backend_status_code: int | None = None,
    backend_response_text: str | None = None,
    retry_attempt: int | None = None,
    max_retries: int | None = None,
    exception: Exception | None = None,
) -> func.HttpResponse:
    body = _build_error_body(
        request_id=request_id,
        error_type=error_type,
        message=message,
        stage=stage,
        requested_model=requested_model,
        deployment=deployment,
        backend_url=backend_url,
        backend_status_code=backend_status_code,
        backend_response_text=backend_response_text,
        retry_attempt=retry_attempt,
        max_retries=max_retries,
        exception=exception,
    )
    logging.exception("Proxy error body: %s", json.dumps(body, ensure_ascii=False))
    return _json_response(body, status_code=status_code)


def _resolve_deployment(payload: dict) -> tuple[str, str | None]:
    """
    Resolve incoming OpenAI `model` alias to an Azure deployment name.
    Backward compatibility:
    - If no model map is configured, use AZURE_OPENAI_DEPLOYMENT.
    - If model is omitted, use the default deployment.
    """
    requested_model = payload.get("model")

    if not MODEL_MAP:
        return DEPLOYMENT, requested_model

    if requested_model in MODEL_MAP:
        return MODEL_MAP[requested_model], requested_model

    if requested_model and requested_model not in MODEL_MAP:
        raise ValueError(f"Unsupported model '{requested_model}'")

    if DEFAULT_MODEL_ALIAS and DEFAULT_MODEL_ALIAS in MODEL_MAP:
        return MODEL_MAP[DEFAULT_MODEL_ALIAS], DEFAULT_MODEL_ALIAS

    return DEPLOYMENT, requested_model


def _should_retry_response(response: httpx.Response) -> bool:
    return response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _should_stream(payload: dict) -> bool:
    return bool(payload.get("stream"))


def _get_token(request_id: str) -> str:
    try:
        return DefaultAzureCredential().get_token(SCOPE).token
    except Exception as e:
        logging.exception("Token acquisition failed for request_id=%s", request_id)
        raise RuntimeError("TokenAcquisitionError") from e


def _proxy_headers(token: str, request_id: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-proxy-request-id": request_id,
    }


def _job_file(job_id: str) -> Path:
    return ASYNC_JOB_DIR / f"{job_id}.json"


def _cleanup_expired_jobs() -> None:
    now = time.time()
    for path in ASYNC_JOB_DIR.glob("*.json"):
        try:
            if now - path.stat().st_mtime > ASYNC_JOB_TTL_SECONDS:
                path.unlink(missing_ok=True)
        except Exception:
            logging.exception("Failed to cleanup async job file: %s", path)


def _save_job(job: dict) -> None:
    with _JOB_LOCK:
        _cleanup_expired_jobs()
        _job_file(job["job_id"]).write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")


def _load_job(job_id: str) -> dict | None:
    path = _job_file(job_id)
    if not path.exists():
        return None
    with _JOB_LOCK:
        return json.loads(path.read_text(encoding="utf-8"))


def _build_job_status_url(job_id: str) -> str:
    return f"/api/v1/jobs/{job_id}"


def _parse_json_bytes(raw: bytes | None) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _response_text(response: func.HttpResponse) -> str:
    try:
        return response.get_body().decode("utf-8")
    except Exception:
        return response.get_body().decode("utf-8", errors="replace")


def _is_async_fallback_candidate(response: func.HttpResponse, payload: dict) -> bool:
    if not ASYNC_FALLBACK_ENABLED:
        return False

    if _should_stream(payload):
        return False

    if response.status_code not in ASYNC_FALLBACK_STATUS_CODES:
        return False

    parsed = _parse_json_bytes(response.get_body())
    if parsed is None:
        return response.status_code in {502, 503, 504}

    error_type = parsed.get("error_type")
    backend_status_code = parsed.get("backend_status_code")

    if error_type in ASYNC_FALLBACK_ERROR_TYPES:
        return True

    if isinstance(backend_status_code, int) and backend_status_code in ASYNC_FALLBACK_STATUS_CODES:
        return True

    return response.status_code in {502, 503, 504}


def _accepted_async_response(*, request_id: str, job_id: str, deployment: str | None, requested_model: str | None) -> func.HttpResponse:
    body = {
        "ok": True,
        "mode": "async_fallback",
        "proxy_request_id": request_id,
        "job_id": job_id,
        "status": "queued",
        "status_url": _build_job_status_url(job_id),
        "poll_after_seconds": ASYNC_FALLBACK_POLL_AFTER_SECONDS,
        "deployment": deployment,
        "requested_model": requested_model,
    }
    return _json_response(
        body,
        status_code=202,
        headers={
            "x-proxy-request-id": request_id,
            "x-proxy-mode": "async_fallback",
            "x-proxy-job-id": job_id,
            "Retry-After": str(ASYNC_FALLBACK_POLL_AFTER_SECONDS),
        },
    )


def _run_async_job(job_id: str) -> None:
    job = _load_job(job_id)
    if job is None:
        return

    job["status"] = "running"
    job["started_at"] = time.time()
    _save_job(job)

    request_data = job["request"]
    response = _forward_request(
        request_id=f"{job_id}:async",
        payload=request_data["payload"],
        backend_url=request_data["backend_url"],
        params=request_data["params"],
        requested_model=request_data["requested_model"],
        deployment=request_data["deployment"],
    )

    response_text = _response_text(response)
    content_type = response.mimetype or "application/json"

    if 200 <= response.status_code < 300:
        job["status"] = "succeeded"
    else:
        job["status"] = "failed"

    job["finished_at"] = time.time()
    job["result"] = {
        "status_code": response.status_code,
        "content_type": content_type,
        "body_text": response_text,
        "headers": {
            "x-proxy-backend-status": response.headers.get("x-proxy-backend-status", ""),
            "x-proxy-deployment": response.headers.get("x-proxy-deployment", ""),
            "x-proxy-retry-attempt": response.headers.get("x-proxy-retry-attempt", ""),
        },
    }
    _save_job(job)


def _submit_async_job(
    *,
    request_id: str,
    payload: dict,
    backend_url: str,
    params: dict,
    requested_model: str | None,
    deployment: str | None,
) -> func.HttpResponse:
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "request": {
            "payload": payload,
            "backend_url": backend_url,
            "params": params,
            "requested_model": requested_model,
            "deployment": deployment,
        },
        "result": None,
    }
    _save_job(job)

    worker = threading.Thread(target=_run_async_job, args=(job_id,), daemon=True)
    worker.start()

    return _accepted_async_response(
        request_id=request_id,
        job_id=job_id,
        deployment=deployment,
        requested_model=requested_model,
    )


def _stream_backend_response(
    *,
    request_id: str,
    payload: dict,
    backend_url: str,
    params: dict,
    headers: dict,
    requested_model: str | None,
    deployment: str | None,
) -> func.HttpResponse:
    try:
        with httpx.stream(
            "POST",
            backend_url,
            params=params,
            headers=headers,
            json=payload,
            timeout=_build_timeout(),
        ) as r:
            content_type = r.headers.get("content-type", "text/event-stream")
            chunks = []
            for chunk in r.iter_raw():
                chunks.append(chunk)

            return func.HttpResponse(
                body=b"".join(chunks),
                status_code=r.status_code,
                mimetype=content_type,
                headers={
                    "x-proxy-request-id": request_id,
                    "x-proxy-backend-status": str(r.status_code),
                    "x-proxy-deployment": deployment or "",
                },
            )
    except httpx.ConnectTimeout as e:
        return _error_response(
            request_id=request_id,
            error_type="ConnectTimeout",
            message="Timed out while connecting to Azure backend.",
            status_code=504,
            stage="proxy_to_backend_stream",
            requested_model=requested_model,
            deployment=deployment,
            backend_url=backend_url,
            exception=e,
        )
    except httpx.ReadTimeout as e:
        return _error_response(
            request_id=request_id,
            error_type="ReadTimeout",
            message="Timed out while reading streaming response from Azure backend.",
            status_code=504,
            stage="proxy_to_backend_stream",
            requested_model=requested_model,
            deployment=deployment,
            backend_url=backend_url,
            exception=e,
        )
    except httpx.WriteTimeout as e:
        return _error_response(
            request_id=request_id,
            error_type="WriteTimeout",
            message="Timed out while sending streaming request to Azure backend.",
            status_code=504,
            stage="proxy_to_backend_stream",
            requested_model=requested_model,
            deployment=deployment,
            backend_url=backend_url,
            exception=e,
        )
    except httpx.PoolTimeout as e:
        return _error_response(
            request_id=request_id,
            error_type="PoolTimeout",
            message="Timed out while waiting for a pooled HTTP connection.",
            status_code=503,
            stage="proxy_to_backend_stream",
            requested_model=requested_model,
            deployment=deployment,
            backend_url=backend_url,
            exception=e,
        )
    except httpx.ConnectError as e:
        return _error_response(
            request_id=request_id,
            error_type="ConnectError",
            message="Failed to connect to Azure backend.",
            status_code=502,
            stage="proxy_to_backend_stream",
            requested_model=requested_model,
            deployment=deployment,
            backend_url=backend_url,
            exception=e,
        )
    except httpx.RequestError as e:
        return _error_response(
            request_id=request_id,
            error_type="RequestError",
            message="HTTPX request error while streaming from Azure backend.",
            status_code=502,
            stage="proxy_to_backend_stream",
            requested_model=requested_model,
            deployment=deployment,
            backend_url=backend_url,
            exception=e,
        )
    except Exception as e:
        return _error_response(
            request_id=request_id,
            error_type="UnexpectedProxyError",
            message="Unexpected proxy error while streaming from Azure backend.",
            status_code=500,
            stage="proxy_to_backend_stream",
            requested_model=requested_model,
            deployment=deployment,
            backend_url=backend_url,
            exception=e,
        )


def _forward_request(
    *,
    request_id: str,
    payload: dict,
    backend_url: str,
    params: dict,
    requested_model: str | None,
    deployment: str | None,
) -> func.HttpResponse:
    try:
        token = _get_token(request_id)
    except Exception as e:
        return _error_response(
            request_id=request_id,
            error_type="TokenAcquisitionError",
            message="Failed to acquire Azure credential token.",
            status_code=500,
            stage="credential",
            requested_model=requested_model,
            deployment=deployment,
            backend_url=backend_url,
            exception=e,
        )

    headers = _proxy_headers(token, request_id)

    if _should_stream(payload):
        return _stream_backend_response(
            request_id=request_id,
            payload=payload,
            backend_url=backend_url,
            params=params,
            headers=headers,
            requested_model=requested_model,
            deployment=deployment,
        )

    last_exception = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=_build_timeout()) as client:
                r = client.post(
                    backend_url,
                    params=params,
                    headers=headers,
                    json=payload,
                )

            if _should_retry_response(r) and attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue

            return func.HttpResponse(
                r.content,
                status_code=r.status_code,
                mimetype=r.headers.get("content-type", "application/json"),
                headers={
                    "x-proxy-request-id": request_id,
                    "x-proxy-backend-status": str(r.status_code),
                    "x-proxy-deployment": deployment or "",
                    "x-proxy-retry-attempt": str(attempt),
                },
            )

        except httpx.ConnectTimeout as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            return _error_response(
                request_id=request_id,
                error_type="ConnectTimeout",
                message="Timed out while connecting to Azure backend.",
                status_code=504,
                stage="proxy_to_backend",
                requested_model=requested_model,
                deployment=deployment,
                backend_url=backend_url,
                retry_attempt=attempt,
                max_retries=MAX_RETRIES,
                exception=e,
            )

        except httpx.ReadTimeout as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            return _error_response(
                request_id=request_id,
                error_type="ReadTimeout",
                message="Timed out while waiting for response data from Azure backend.",
                status_code=504,
                stage="proxy_to_backend",
                requested_model=requested_model,
                deployment=deployment,
                backend_url=backend_url,
                retry_attempt=attempt,
                max_retries=MAX_RETRIES,
                exception=e,
            )

        except httpx.WriteTimeout as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            return _error_response(
                request_id=request_id,
                error_type="WriteTimeout",
                message="Timed out while sending request data to Azure backend.",
                status_code=504,
                stage="proxy_to_backend",
                requested_model=requested_model,
                deployment=deployment,
                backend_url=backend_url,
                retry_attempt=attempt,
                max_retries=MAX_RETRIES,
                exception=e,
            )

        except httpx.PoolTimeout as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            return _error_response(
                request_id=request_id,
                error_type="PoolTimeout",
                message="Timed out while waiting for a pooled HTTP connection.",
                status_code=503,
                stage="proxy_to_backend",
                requested_model=requested_model,
                deployment=deployment,
                backend_url=backend_url,
                retry_attempt=attempt,
                max_retries=MAX_RETRIES,
                exception=e,
            )

        except httpx.ConnectError as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            return _error_response(
                request_id=request_id,
                error_type="ConnectError",
                message="Failed to connect to Azure backend.",
                status_code=502,
                stage="proxy_to_backend",
                requested_model=requested_model,
                deployment=deployment,
                backend_url=backend_url,
                retry_attempt=attempt,
                max_retries=MAX_RETRIES,
                exception=e,
            )

        except httpx.RequestError as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            return _error_response(
                request_id=request_id,
                error_type="RequestError",
                message="HTTPX request error while calling Azure backend.",
                status_code=502,
                stage="proxy_to_backend",
                requested_model=requested_model,
                deployment=deployment,
                backend_url=backend_url,
                retry_attempt=attempt,
                max_retries=MAX_RETRIES,
                exception=e,
            )

        except Exception as e:
            last_exception = e
            return _error_response(
                request_id=request_id,
                error_type="UnexpectedProxyError",
                message="Unexpected proxy-side error while calling Azure backend.",
                status_code=500,
                stage="proxy_to_backend",
                requested_model=requested_model,
                deployment=deployment,
                backend_url=backend_url,
                retry_attempt=attempt,
                max_retries=MAX_RETRIES,
                exception=e,
            )

    return _error_response(
        request_id=request_id,
        error_type="RetryExhausted",
        message="All retry attempts were exhausted.",
        status_code=502,
        stage="proxy_to_backend",
        requested_model=requested_model,
        deployment=deployment,
        backend_url=backend_url,
        retry_attempt=MAX_RETRIES,
        max_retries=MAX_RETRIES,
        exception=last_exception,
    )


def _handle_proxy_request(
    *,
    request_id: str,
    payload: dict,
    backend_url: str,
    params: dict,
    requested_model: str | None,
    deployment: str | None,
) -> func.HttpResponse:
    response = _forward_request(
        request_id=request_id,
        payload=payload,
        backend_url=backend_url,
        params=params,
        requested_model=requested_model,
        deployment=deployment,
    )

    if _is_async_fallback_candidate(response, payload):
        logging.warning(
            "Sync request failed and will fallback to async. request_id=%s status=%s deployment=%s",
            request_id,
            response.status_code,
            deployment,
        )
        return _submit_async_job(
            request_id=request_id,
            payload=payload,
            backend_url=backend_url,
            params=params,
            requested_model=requested_model,
            deployment=deployment,
        )

    return response


@app.route(route="v1/chat/completions", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def chat_completions(req: func.HttpRequest) -> func.HttpResponse:
    request_id = str(uuid.uuid4())

    try:
        payload = req.get_json()
    except ValueError as e:
        return _error_response(
            request_id=request_id,
            error_type="InvalidJSON",
            message="Request body is not valid JSON.",
            status_code=400,
            stage="request_parse",
            exception=e,
        )

    try:
        deployment, requested_model = _resolve_deployment(payload)
    except ValueError as e:
        return _error_response(
            request_id=request_id,
            error_type="ModelResolutionError",
            message=str(e),
            status_code=400,
            stage="model_resolution",
            exception=e,
        )

    if requested_model:
        payload["model"] = requested_model

    backend_url = f"{BASE}/openai/deployments/{deployment}/chat/completions"
    params = {"api-version": API_VERSION}

    return _handle_proxy_request(
        request_id=request_id,
        payload=payload,
        backend_url=backend_url,
        params=params,
        requested_model=requested_model,
        deployment=deployment,
    )


@app.route(route="v1/responses", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def responses(req: func.HttpRequest) -> func.HttpResponse:
    request_id = str(uuid.uuid4())

    try:
        payload = req.get_json()
    except ValueError as e:
        return _error_response(
            request_id=request_id,
            error_type="InvalidJSON",
            message="Request body is not valid JSON.",
            status_code=400,
            stage="request_parse",
            exception=e,
        )

    try:
        deployment, requested_model = _resolve_deployment(payload)
    except ValueError as e:
        return _error_response(
            request_id=request_id,
            error_type="ModelResolutionError",
            message=str(e),
            status_code=400,
            stage="model_resolution",
            exception=e,
        )

    payload["model"] = deployment

    backend_url = f"{BASE}/openai/v1/responses"
    params = {"api-version": V1_API_VERSION}

    return _handle_proxy_request(
        request_id=request_id,
        payload=payload,
        backend_url=backend_url,
        params=params,
        requested_model=requested_model,
        deployment=deployment,
    )


@app.route(route="v1/jobs/{job_id}", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def get_job(req: func.HttpRequest) -> func.HttpResponse:
    request_id = str(uuid.uuid4())
    job_id = req.route_params.get("job_id")

    if not job_id:
        return _error_response(
            request_id=request_id,
            error_type="MissingJobId",
            message="job_id is required.",
            status_code=400,
            stage="job_lookup",
        )

    job = _load_job(job_id)
    if job is None:
        return _error_response(
            request_id=request_id,
            error_type="JobNotFound",
            message=f"Async job '{job_id}' was not found.",
            status_code=404,
            stage="job_lookup",
        )

    body = {
        "ok": True,
        "job_id": job_id,
        "status": job["status"],
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "status_url": _build_job_status_url(job_id),
        "poll_after_seconds": ASYNC_FALLBACK_POLL_AFTER_SECONDS,
    }

    if job["status"] in {"queued", "running"}:
        return _json_response(
            body,
            status_code=200,
            headers={
                "x-proxy-request-id": request_id,
                "Retry-After": str(ASYNC_FALLBACK_POLL_AFTER_SECONDS),
            },
        )

    result = job.get("result") or {}
    result_body_text = result.get("body_text")
    result_body_json = None
    if isinstance(result_body_text, str):
        try:
            result_body_json = json.loads(result_body_text)
        except Exception:
            result_body_json = None

    body["result"] = {
        "status_code": result.get("status_code"),
        "content_type": result.get("content_type"),
        "headers": result.get("headers", {}),
        "body_json": result_body_json,
        "body_text": None if result_body_json is not None else result_body_text,
    }
    return _json_response(body, status_code=200, headers={"x-proxy-request-id": request_id})

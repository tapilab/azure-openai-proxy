import json
import os
import time
import uuid
import traceback
import logging

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


def _build_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT_SECONDS,
        read=READ_TIMEOUT_SECONDS,
        write=WRITE_TIMEOUT_SECONDS,
        pool=POOL_TIMEOUT_SECONDS,
    )


def _json_response(body: dict, status_code: int) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(body, ensure_ascii=False),
        status_code=status_code,
        mimetype="application/json",
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

    return _forward_request(
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

    return _forward_request(
        request_id=request_id,
        payload=payload,
        backend_url=backend_url,
        params=params,
        requested_model=requested_model,
        deployment=deployment,
    )
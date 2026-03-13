import json
import os
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

    # Final fallback preserves old single-deployment behavior.
    return DEPLOYMENT, requested_model

@app.route(route="v1/chat/completions", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def chat_completions(req: func.HttpRequest) -> func.HttpResponse:
    try:
        payload = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON"}),
            status_code=400,
            mimetype="application/json",
        )

    try:
        deployment, requested_model = _resolve_deployment(payload)
    except ValueError as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json",
        )

    # Keep model in payload stable for clients that expect echoing.
    if requested_model:
        payload["model"] = requested_model

    token = DefaultAzureCredential().get_token(SCOPE).token

    backend_url = f"{BASE}/openai/deployments/{deployment}/chat/completions"
    params = {"api-version": API_VERSION}

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=60.0) as client:
        r = client.post(backend_url, params=params, headers=headers, json=payload)

    return func.HttpResponse(
        r.content,
        status_code=r.status_code,
        mimetype=r.headers.get("content-type", "application/json"),
    )

@app.route(route="v1/responses", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
def responses(req: func.HttpRequest) -> func.HttpResponse:
    try:
        payload = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON"}),
            status_code=400,
            mimetype="application/json",
        )

    try:
        deployment, requested_model = _resolve_deployment(payload)
    except ValueError as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json",
        )

    # Azure OpenAI responses API expects the deployment in `model`.
    payload["model"] = deployment

    token = DefaultAzureCredential().get_token(SCOPE).token

    backend_url = f"{BASE}/openai/v1/responses"
    params = {"api-version": V1_API_VERSION}

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=60.0) as client:
        r = client.post(backend_url, params=params, headers=headers, json=payload)

    return func.HttpResponse(
        r.content,
        status_code=r.status_code,
        mimetype=r.headers.get("content-type", "application/json"),
    )

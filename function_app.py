import json
import os
import azure.functions as func
import httpx
from azure.identity import DefaultAzureCredential

app = func.FunctionApp()

BASE = os.environ["AZURE_OPENAI_BASE"].rstrip("/")
DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

SCOPE = "https://cognitiveservices.azure.com/.default"

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

    token = DefaultAzureCredential().get_token(SCOPE).token

    backend_url = f"{BASE}/openai/deployments/{DEPLOYMENT}/chat/completions"
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

    token = DefaultAzureCredential().get_token(SCOPE).token

    backend_url = f"{BASE}/openai/v1/responses"
    params = {"api-version": os.getenv("AZURE_OPENAI_V1_API_VERSION", "preview")}

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

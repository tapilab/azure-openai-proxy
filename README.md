# Azure OpenAI Proxy (Azure Functions)

This repo contains a **lightweight Azure Function** that exposes an **OpenAI-compatible API** (`/v1/chat/completions` and `/v1/responses`) backed by **Azure OpenAI**, using:

* **Azure Functions** as a proxy
* **Managed Identity** for Azure authentication (no API keys in code)
* **Function / Host keys** as a shared access token
* Compatibility with:

  * OpenAI Python SDK
  * DSPy (via chat completions)

This setup is ideal for **university / research tenants** where Entra ID user management is restricted.

---

## Quickstart

See [Quickstart.ipynb](Quickstart.ipynb)

---

## Architecture (high level)

```
Client (OpenAI SDK / DSPy)
  └─ x-functions-key (shared token)
       ↓
Azure Function (this repo)
  └─ Managed Identity (RBAC)
       ↓
Azure OpenAI
```

---

## Public Endpoints

Once deployed, the Function exposes:

* `POST /api/v1/chat/completions`
* `POST /api/v1/responses`

Base URL format:

```
https://<function-app>.azurewebsites.net/api/v1
```

---

## Repository Structure

```
.
├─ function_app.py     # Azure Function (HTTP proxy)
├─ requirements.txt    # Python dependencies
├─ host.json           # Functions host config
└─ README.md
```

---

## Required Azure Configuration

### 1. Enable Managed Identity

On the **Function App**:

* Settings → Identity → **System-assigned = On**

### 2. Grant RBAC

On the **Azure OpenAI resource**:

* Assign the Function’s managed identity the role:

  * **Cognitive Services OpenAI User**
  * (or `Cognitive Services User` if required by tenant)

### 3. Application Settings (Environment Variables)

Set these in **Function App → Configuration → Application settings**:

```text
AZURE_OPENAI_BASE=https://<your-openai-resource>.cognitiveservices.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-05-01-preview
```

Restart the Function App after saving.

---

## Authentication (Shared Access)

Access is controlled via **Azure Functions keys**.

* Use a **Host key** (recommended) to authorize all endpoints
* Send it as an HTTP header:

```http
x-functions-key: <HOST_KEY>
```

⚠️ Do **not** share the `_master` key.

---

## Local Development

### Prerequisites

* Python 3.11+
* Azure CLI (`az login`)
* Azure Functions Core Tools (optional)

### Run locally

```bash
pip install -r requirements.txt
func start
```

Local settings can be placed in `local.settings.json` (not committed).

---

## Example: OpenAI SDK (Responses API)

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://<function-app>.azurewebsites.net/api/v1",
    api_key="unused",
    default_headers={"x-functions-key": os.environ["FUNCTION_HOST_KEY"]},
)

resp = client.responses.create(
    model="gpt-4o-mini",
    input="Tell me a one line story.",
    temperature=0.7,
)

print(resp.output_text)
```

---

## Example: DSPy

```python
import dspy
import os

lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_base="https://<function-app>.azurewebsites.net/api/v1",
    api_key="unused",
    headers={"x-functions-key": os.environ["FUNCTION_HOST_KEY"]},
)

dspy.configure(lm=lm)

pred = dspy.Predict("question -> answer")(question="Tell me a one line story.")
print(pred.answer)
```

---

## Notes & Limitations

* This proxy is **pass-through**: all OpenAI parameters (e.g. `temperature`) are forwarded as-is.
* Streaming requires additional handling (not included here).
* For rate limiting, analytics, or multiple keys, consider placing this Function behind **Azure API Management (APIM)**.


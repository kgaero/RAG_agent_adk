# Vertex AI Agent Engine Deployment

This deployment flow packages the ADK agent from `agents/ragagent` and deploys
it to Vertex AI Agent Engine with environment-driven configuration.

## Prerequisites

- Authenticate with Google Cloud ADC, for example:
  `gcloud auth application-default login`
- Set the required runtime values in the repo-root `.env`.
- Add these deployment/runtime values to `.env` as well:
  `GOOGLE_GENAI_USE_VERTEXAI=1`
  `GOOGLE_CLOUD_STORAGE_BUCKET=<existing-staging-bucket>`
  `RAG_AGENT_LLM_MODEL=<gemini model id>`
  `DEFAULT_EMBEDDING_MODEL=<embedding model id>`
  `DEFAULT_CHUNK_SIZE=<chunk size>`
  `DEFAULT_CHUNK_OVERLAP=<chunk overlap>`
  `DEFAULT_TOP_K=<top k>`
  `DEFAULT_DISTANCE_THRESHOLD=<distance threshold>`

## Deploy

Run the deployment command from the repository root:

```bash
python deployment/deploy.py
```

On success the script prints:

- `Deployment status: success`
- `Agent resource name: projects/.../locations/.../reasoningEngines/...`
- `Python SDK get: client.agent_engines.get(name='projects/...')`
- `Metadata URL: https://<location>-aiplatform.googleapis.com/v1/projects/...`
- `Query URL: https://<location>-aiplatform.googleapis.com/v1/projects/...:query`
- `Stream query URL: https://<location>-aiplatform.googleapis.com/v1/projects/...:streamQuery`

## Invoke

Set the deployed resource name, then send a message from the repository root:

```bash
export AGENT_ENGINE_RESOURCE_NAME=projects/.../locations/.../reasoningEngines/...
python deployment/invoke.py "list all corpora"
```

The script:

- loads the same `.env` values as the local agent
- creates a Vertex AI session when `--session-id` is omitted
- streams the agent response to stdout
- prints the created session id to stderr so you can reuse it later

Reuse an existing session:

```bash
python deployment/invoke.py \
  --session-id=<returned-session-id> \
  "add below document to \"techbriefs\" corpus https://drive.google.com/..."
```

Print raw streamed events instead of extracted text:

```bash
python deployment/invoke.py --raw-events "list all corpora"
```

## Local ADK Web

For local testing, run the ADK dev UI from the repository root against the
`agents` directory:

```bash
adk web agents
```

Then select `ragagent` in the agent picker.

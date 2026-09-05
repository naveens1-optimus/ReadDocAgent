# Multi-Agent Document Processing (IDP)

A LangGraph pipeline that classifies uploaded documents with Azure AI, pauses
for human approval when it is unsure, and stores the result.

```
classify -> human_approval -> extract -> extraction_review -> validate
                                                                 |
                    +--> complete_fields <--(missing required)---+
                    |          |                                 |
                    +----------+                          (complete)
                                                                 |
                                        save <-- enrich <--------+
```

Four agents:

| Agent | Node(s) | What it does |
|---|---|---|
| 1 Classifier | `classify` | Azure OpenAI vision over the rendered first page, falling back to Document Intelligence `prebuilt-read` text |
| 2 Extraction | `extract` | Runs the Document Intelligence model for the approved type |
| 3 Validation & Enrichment | `validate`, `complete_fields`, `enrich` | Checks the data against the entity schema, then standardises and summarises it with Azure OpenAI |
| 4 Output & Storage | `save` | Stores the entity in Cosmos DB, and the output plus processing report in Blob Storage |

A failure in any node routes to `error_handler`.

### Three human checkpoints

Each gates something different:

| Checkpoint | When it pauses |
|---|---|
| `human_approval` | Only when the classification is doubtful — below `CONFIDENCE_THRESHOLD`, or type `unsupported` |
| `extraction_review` | **Always.** The reviewer is signing off the data itself, and may edit it first |
| `complete_fields` | Only when the schema requires a field the extraction could not supply |

`complete_fields` loops back through `validate`, so supplied values are checked
like any others. A reviewer who cannot supply a value can tick *continue
regardless* — the run finishes and the output is saved, but no entity is
stored, because the data never validated.

### Extraction

Each document type is extracted with its own prebuilt model:

| Type | Model | Path |
|---|---|---|
| invoice | `prebuilt-invoice` | typed fields, with per-field confidence |
| receipt | `prebuilt-receipt` | typed fields, with per-field confidence |
| contract | `prebuilt-layout` | text + tables, structured by Azure OpenAI |
| resume | `prebuilt-read` | text, structured by Azure OpenAI |
| id_card | *none yet* | classified and stored, extraction skipped |

Only the document-specific models return per-field confidence scores. On the
layout/read path the fields report `None` rather than inventing a number.

Adding a type is one line in `EXTRACTION_MODEL_BY_TYPE`
([document_type.py](backend/src/domain/enum/document_type.py)) plus an entity
schema.

### Entity schemas

Validation runs against a Pydantic entity per type
([domain/entity/](backend/src/domain/entity/)). The schema *is* the
completeness rule: a field declared without a default is required, and if the
extraction cannot supply it the pipeline pauses and asks a human.

| Type | Required fields |
|---|---|
| invoice | `vendor_name`, `invoice_id`, `invoice_total` |
| receipt | `merchant_name`, `total` |
| contract | `title`, `parties` |
| resume | `full_name` |

Validated entities are stored in Cosmos DB, keyed on document id, in their own
container (`AZURE_COSMOS_ENTITY_CONTAINER`, default `entities`) — separate
from the checkpointer's container.

### Processing report

Written to `{document_id}/report.json` in the output container, with the four
sections the specification asks for: classification result and confidence;
extraction summary with field-level confidence scores (weakest first, with
bounding boxes and page numbers); validation results, passed and failed; and
anything flagged for human review.

Every field carries **its own confidence score** where Document Intelligence
reports one; a blank score means it reports none, which is different from
reporting a low one. At `extraction_review` the reviewer sees each field with
its score and can change values, drop fields or add new ones. Whatever they
approve is what gets saved, with edited fields marked so the output
distinguishes an extracted value from a corrected one.

## Setup

Requires **Python 3.11**.

```bash
cd backend
python -m venv venv
./venv/Scripts/python.exe -m pip install -r src/requirements.txt   # Windows
./venv/Scripts/python.exe -m pip install -r ../frontend/requirements.txt
```

Then fill in `backend/src/.env`. It already contains the **required** settings
with placeholder values -- replace each `REPLACE-ME`:

| Variable | Where to find it |
|---|---|
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` / `_KEY` | Document Intelligence resource → Keys and Endpoint |
| `AZURE_OPENAI_ENDPOINT` / `_KEY` | Azure OpenAI resource → Keys and Endpoint |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Your **deployment name** in AI Foundry (must be vision-capable, e.g. a `gpt-4o` deployment) |
| `AZURE_STORAGE_CONNECTION_STRING` | Storage account → Access keys → Connection string |
| `AZURE_COSMOS_ENDPOINT` / `_KEY` | Cosmos DB account (NoSQL API) → Keys |

Everything else has a working default and is documented in
[backend/src/.env.example](backend/src/.env.example) -- add a line to `.env`
only to override one. Blob containers and the Cosmos database/container are
created automatically; only the accounts themselves must exist.

Authentication is **key-based** throughout. `.env` is git-ignored.

## Running

Two processes.

```bash
# Terminal 1 -- API on http://127.0.0.1:8000
cd backend/src
../venv/Scripts/python.exe main.py

# Terminal 2 -- UI on http://localhost:8501
./backend/venv/Scripts/python.exe -m streamlit run frontend/src/app.py
```

The UI sidebar shows backend readiness. If it says *Not ready*, it names the
setting that is wrong.

`http://127.0.0.1:8000/docs` has the interactive API docs.

## Endpoints

| | |
|---|---|
| `POST /documents` | Upload, store and classify |
| `POST /documents/{thread_id}/approval` | Resume a paused run |
| `GET /documents/{thread_id}` | Status, including any pending approval |
| `GET /health` | Liveness — stays green even when misconfigured, so the process is diagnosable |
| `GET /health/ready` | Readiness — 503 naming the blocking component |

## How the two checkpoints work

**Classification** is confidence-gated by `CONFIDENCE_THRESHOLD` (default `0.80`):

- **At or above it**, and a recognised type → auto-approved, no pause.
- **Below it**, or type `unsupported` → pauses. The reviewer can approve,
  reject, or correct the document type while approving.

**Extraction review always pauses** — there is no confidence shortcut, because
the reviewer is signing off the data that gets saved.

Both use the same endpoint. The `approval_request` payload's `stage` field says
which gate is open (`"classification"` or `"extraction"`), and the run's status
is `awaiting_approval` or `awaiting_extraction_review` accordingly.

State is checkpointed in **Cosmos DB**, so a run paused on one request resumes
on a later one -- even from a different process.

To review *every* document, remove the auto-approve branch in
`_human_approval` ([document_processing_workflow.py](backend/src/infrastructure/workflows/document_processing_workflow.py)).

## Tracing

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`. Every graph
run, node and model call is then traced.

## Tests

```bash
cd backend
./venv/Scripts/python.exe -m pytest -q
```

264 tests, ~5s. They run entirely against fakes -- no Azure calls, no network,
and independent of whatever is in your `.env`.

## Layout

```
backend/src/
  api/            FastAPI app and routes
  application/    use-case handlers + the ports they depend on
  domain/         entities, enums, schemas (no infrastructure imports)
  infrastructure/ Azure adapters, agents, graph, DI wiring
frontend/src/     Streamlit UI
```

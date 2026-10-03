# Document Intelligence Agent

A Copilot-style document Q&A tool built on Retrieval-Augmented Generation (RAG) with Azure OpenAI and ChromaDB. Load a PDF, Word, or text document and ask questions about its content; answers are grounded in the most relevant passages, checked, reviewed by a human, and logged.

## Overview

This project demonstrates enterprise-level Generative AI capabilities:
- **RAG pipeline** — ingestion, chunking, embeddings, and semantic vector search (ChromaDB) feeding retrieved context to GPT-4o
- **Document Summarization** — automatically generates concise summaries
- **Intelligent Q&A** — answers cite the retrieved excerpts they are based on
- **Entity Extraction** — identifies and categorizes key information (names, dates, amounts, etc.)
- **Responsible AI** — prompt validation on the way in, output filtering on the way out, and human-in-the-loop review before an answer is shown
- **Query History** — SQLite logs every question, its outcome, and the source chunks for audit and analysis

## Architecture

```
Document (PDF / DOCX / TXT)
    ↓
Text extraction
    ↓
Chunking (1,000 characters, 200 overlap)
    ↓
Embeddings (Azure OpenAI) → ChromaDB vector store (persistent, ./chroma_db)

Question
    ↓
Prompt validation (is the question about the document?)
    ↓
Semantic search in ChromaDB (top 4 chunks)
    ↓
GPT-4o answers from the retrieved chunks, with citations
    ↓
Output filter (Azure content filter + groundedness check)
    ↓
Human review (approve / edit / reject)
    ↓
Answer shown + SQLite log (status, sources, review note)
```

## Requirements

- Python 3.10+ (developed on 3.11)
- An Azure OpenAI resource with two deployments:
  - a chat model (default name `gpt-4o`)
  - an embedding model (default name `text-embedding-3-small`)
- Supported document formats: PDF (text-based), DOCX, TXT

## Setup

### 1. Create a virtual environment and install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI
Create a `.env` file in the project folder (it is git-ignored):
```
AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_API_KEY=<your-key>
AZURE_API_VERSION=2024-12-01-preview
AZURE_DEPLOYMENT_NAME=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
HUMAN_REVIEW=true
```

| Setting | Required | Default | Purpose |
| --- | --- | --- | --- |
| `AZURE_ENDPOINT` | yes | — | Your Azure OpenAI endpoint |
| `AZURE_API_KEY` | yes | — | Your Azure OpenAI key |
| `AZURE_API_VERSION` | no | `2024-12-01-preview` | API version |
| `AZURE_DEPLOYMENT_NAME` | no | `gpt-4o` | Chat model deployment |
| `AZURE_EMBEDDING_DEPLOYMENT` | no | `text-embedding-3-small` | Embedding model deployment |
| `HUMAN_REVIEW` | no | `true` | Set to `false` to let answers that pass the output filter through without review |

### 3. Run the application
```bash
python doc_intelligence_agent.py                    # uses sample_document.txt
python doc_intelligence_agent.py path/to/contract.pdf
```

## Usage

The system will:
1. Load the document and extract its text
2. Chunk it, embed the chunks, and store them in ChromaDB
3. Generate a summary and extract entities
4. Start an interactive Q&A session

### Example Interaction
```
Indexing document (chunking and embedding)...
Indexed 2 chunks into ChromaDB

Your question: What happens if payment is late?

------------------------------------------------------------
REVIEW REQUIRED
------------------------------------------------------------
Question: What happens if payment is late?

Draft answer:
A late payment penalty of 2% per month applies [1].

Automated check: Grounded in retrieved excerpts

Sources:
  - sample_document.txt::0 (distance 0.21): SERVICE AGREEMENT This Service Agreement ...

[a]pprove, [e]dit or [r]eject? a

ANSWER:
A late payment penalty of 2% per month applies [1].
```

## Database Schema

Query history is stored in `document_queries.db`:

```sql
CREATE TABLE queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT,       -- approved, edited, rejected, auto_approved, blocked_by_filter, rejected_by_validation, error
    sources TEXT,      -- JSON list of the chunk ids used
    review_note TEXT   -- reviewer's note or the filter's reason
);
```

Older databases are upgraded in place on start-up.

## Key Components

| Function | Purpose |
| --- | --- |
| `load_document(file_path)` | Extracts text from PDF, DOCX (paragraphs and tables), or TXT |
| `chunk_text(text, chunk_size=1000, overlap=200)` | Splits text into overlapping chunks, breaking at paragraph or line ends |
| `embed_texts(texts)` | Creates embeddings with the Azure OpenAI embedding deployment |
| `index_document(collection, document_name, text)` | Chunks, embeds, and stores a document in ChromaDB (re-indexing replaces old chunks) |
| `retrieve_chunks(collection, question, document_name)` | Semantic vector search for the most relevant chunks |
| `validate_prompt(query, document_summary)` | Input check: rejects off-topic questions; fails closed on errors |
| `filter_output(answer, chunks, finish_reason)` | Output check: Azure content filter result + LLM groundedness check |
| `human_review(question, answer, chunks, filter_reason)` | Reviewer approves, edits, or rejects the answer before it is shown |
| `answer_question(collection, question, document_name, summary)` | The full RAG flow, with logging |
| `summarize_document(text)` / `extract_entities(text)` | Summary and JSON entity extraction |

## Limitations

- Summary and entity extraction read the first 3,000 / 4,000 characters; Q&A uses the whole document through retrieval
- Image-only (scanned) PDFs produce no text
- API rate limits apply (based on Azure subscription)

## Future Enhancements

- [ ] Web UI with Streamlit
- [ ] Support for image-based PDFs (OCR)
- [ ] Multi-document queries (cross-document search)
- [ ] Map-reduce summarization for long documents
- [ ] Integration with Microsoft 365 via Microsoft Graph API

## License

MIT

## Contact

For questions or feedback, reach out to digsubba1@gmail.com

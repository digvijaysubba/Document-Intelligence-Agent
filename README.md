# Document Intelligence Agent

A Copilot-style document Q&A and summarization tool powered by Azure OpenAI. Upload any PDF, Word, or text document and ask intelligent questions about its content.

## Overview

This project demonstrates enterprise-level Generative AI capabilities:
- **Document Summarization** — automatically generates concise summaries
- **Intelligent Q&A** — answers questions grounded in document content
- **Entity Extraction** — identifies and categorizes key information (names, dates, amounts, etc.)
- **Prompt Validation** — responsible AI layer that filters irrelevant queries
- **Query History** — SQLite database logs all questions and answers for audit and analysis

## Architecture

```
User Input
    ↓
Document Loading (PDF/DOCX/TXT extraction)
    ↓
Text Chunking (handles large documents)
    ↓
Prompt Validation (responsible AI filter)
    ↓
Azure OpenAI API (GPT-4o)
    ↓
Response Generated
    ↓
SQLite Logging (query history)
```

## Features

### 1. Document Summarization
Automatically generates a concise summary when you load a document. Useful for quickly understanding document scope and content.

### 2. Question Answering
Ask any question about the document. The system retrieves relevant sections and generates an accurate answer grounded in the document content.

**Example:**
- Document: Service Agreement
- Query: "What is the termination clause?"
- Answer: "Either party may terminate with 30 days written notice..."

### 3. Entity Extraction
Automatically identifies and categorizes key information:
- People and organizations
- Dates and time references
- Dollar amounts and numerical values
- Locations
- Other critical terms

### 4. Responsible AI Layer
- **Prompt Validation**: Before answering, the system validates that your question is actually related to the document
- **Prevents hallucination**: Won't make up answers; tells you if information isn't in the document
- **Audit Trail**: All queries logged to SQLite for transparency and compliance

## Requirements

- Python 3.8+
- Azure OpenAI API key and endpoint
- Supported document formats: PDF, DOCX, TXT

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI
Update the configuration in `doc_intelligence_agent.py`:
```python
AZURE_ENDPOINT = "your-azure-endpoint"
AZURE_API_KEY = "your-api-key"
DEPLOYMENT_NAME = "your-deployment-name"
```

### 3. Run the Application
```bash
python doc_intelligence_agent.py
```

Or use the Streamlit UI:
```bash
streamlit run app.py
```

## Usage Examples

### Terminal Mode
```bash
python doc_intelligence_agent.py
```

The system will:
1. Load `sample_document.txt`
2. Generate a summary
3. Extract entities
4. Start an interactive Q&A session

### Example Interaction
```
DOCUMENT INTELLIGENCE AGENT

Loading document: sample_document.txt

Generating summary...

SUMMARY:
This Service Agreement between ABC Corporation and XYZ Industries outlines 
software development services with a $5,000 monthly retainer, 12-month 
initial term, and 30-day termination clause.

KEY ENTITIES:
{
  "names": ["ABC Corporation", "XYZ Industries"],
  "dates": ["January 15, 2024"],
  "amounts": ["$5,000", "$150", "2%"],
  "locations": []
}

Ask questions about the document (type 'quit' to exit)

Your question: What happens if payment is late?

Thinking...

ANSWER:
According to Section 2 (Payment Terms), if payment is late, there is a 
2% per month late payment penalty applied until the balance is resolved.
```

## Database Schema

Query history is stored in `document_queries.db`:

```sql
CREATE TABLE queries (
    id INTEGER PRIMARY KEY,
    document_name TEXT,
    query TEXT,
    response TEXT,
    timestamp DATETIME
);
```

## Key Components

### `load_document(file_path)`
Loads text from PDF, DOCX, or TXT files.

### `chunk_text(text, chunk_size=2000)`
Splits long documents into overlapping chunks for efficient processing.

### `validate_prompt(query, document_summary)`
Validates query relevance before sending to Azure OpenAI. This is the "responsible AI" feature.

### `summarize_document(text)`
Generates a concise summary using GPT-4o.

### `extract_entities(text)`
Identifies and categorizes key information as structured JSON.

### `answer_question(text, question, document_name)`
Answers a user question based on document content, with validation and logging.

## Technical Details

- **Model**: GPT-4o via Azure OpenAI
- **API Version**: 2024-12-01-preview
- **Storage**: SQLite for local query history
- **Responsible AI**: Prompt validation before API calls to prevent off-topic queries

## Use Cases

- **Contract Review**: Summarize agreements and extract key terms
- **Financial Documents**: Extract amounts, dates, and payment terms
- **Technical Documentation**: Answer questions about system architecture and requirements
- **Compliance**: Audit trail of all document queries for regulatory requirements
- **Research**: Quickly extract information from long reports

## Limitations

- Document size: Tested up to 50+ pages
- API rate limits apply (based on Azure subscription)
- Accuracy depends on document clarity and formatting

## Future Enhancements

- [ ] Web UI with Streamlit
- [ ] Support for image-based PDFs (OCR)
- [ ] Multi-document queries (cross-document search)
- [ ] Fine-tuned models for domain-specific documents
- [ ] Integration with Microsoft 365 via Microsoft Graph API
- [ ] Power Automate workflow integration

## Project Alignment with Job Requirements

This project demonstrates the core skills required for the AI & Application Intern role:

✅ **Building Copilot Agents** — document Q&A is a core Copilot capability  
✅ **Document Summarization** — automatic summary generation  
✅ **Classification & Entity Extraction** — key information categorization  
✅ **Intelligent Search** — grounded question answering over enterprise data  
✅ **Responsible AI** — prompt validation and output filtering  
✅ **Azure Integration** — uses Azure OpenAI API  
✅ **Data Engineering** — document processing and SQLite logging  

## License

MIT

## Contact

For questions or feedback, reach out to digsubba1@gmail.com

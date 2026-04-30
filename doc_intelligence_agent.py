"""
Document Intelligence Agent
A Copilot-style document Q&A and summarization tool using Azure OpenAI
"""

import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import PyPDF2
from docx import Document as DocxDocument
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ==================== CONFIG ====================

AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
AZURE_API_KEY = os.environ["AZURE_API_KEY"]
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

DB_PATH = "document_queries.db"
CHUNK_SIZE = 2000  # characters per chunk

# ==================== DATABASE SETUP ====================

def init_database():
    """Initialize SQLite database for query history"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def save_query(document_name, query, response):
    """Save query and response to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO queries (document_name, query, response, timestamp)
        VALUES (?, ?, ?, ?)
    """, (document_name, query, response, datetime.now()))
    
    conn.commit()
    conn.close()

def get_query_history(document_name=None, limit=10):
    """Retrieve query history from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if document_name:
        cursor.execute("""
            SELECT document_name, query, response, timestamp FROM queries
            WHERE document_name = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (document_name, limit))
    else:
        cursor.execute("""
            SELECT document_name, query, response, timestamp FROM queries
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
    
    results = cursor.fetchall()
    conn.close()
    return results

# ==================== DOCUMENT PROCESSING ====================

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extract text from Word document"""
    text = ""
    try:
        doc = DocxDocument(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text

def extract_text_from_txt(file_path):
    """Extract text from plain text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading TXT: {e}")
    return ""

def load_document(file_path):
    """Load and extract text from any supported document type"""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext == '.docx':
        return extract_text_from_docx(file_path)
    elif file_ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

# ==================== TEXT CHUNKING ====================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=200):
    """Split text into overlapping chunks for processing"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

# ==================== PROMPT VALIDATION ====================

def validate_prompt(query, document_summary):
    """
    Validate that the query is relevant to the document.
    This is the 'responsible AI' layer.
    """
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )
    
    validation_prompt = f"""
You are a query validator. Determine if the following query is relevant to a document.

Document Summary: {document_summary[:500]}

User Query: {query}

Respond with ONLY "VALID" or "INVALID" (no explanation).
A query is VALID if it's asking for information that could be found in the document.
A query is INVALID if it's completely unrelated or asking you to do something outside the document scope.
"""
    
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a validation assistant. Respond with only VALID or INVALID."},
                {"role": "user", "content": validation_prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        return "VALID" in result
    except Exception as e:
        print(f"Validation error: {e}")
        return True  # Allow by default if validation fails

# ==================== DOCUMENT SUMMARIZATION ====================

def summarize_document(text):
    """Generate a concise summary of the document"""
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )
    
    # Use first chunk if document is very long
    content = text[:3000] if len(text) > 3000 else text
    
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a document analyst. Provide concise, clear summaries."},
                {"role": "user", "content": f"Summarize this document in 2-3 sentences:\n\n{content}"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Summarization error: {e}")
        return "Unable to generate summary"

# ==================== ENTITY EXTRACTION ====================

def extract_entities(text):
    """Extract key entities (names, dates, amounts, etc.) from document"""
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )
    
    content = text[:2000] if len(text) > 2000 else text
    
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an entity extraction expert. Extract and categorize key information."},
                {"role": "user", "content": f"""Extract key entities from this text and organize them as JSON:
- names (people/organizations)
- dates
- amounts/numbers
- locations
- other key terms

Text:
{content}

Respond with ONLY valid JSON."""}
            ],
            temperature=0.0,
            max_tokens=300
        )
        
        response_text = response.choices[0].message.content
        try:
            return json.loads(response_text)
        except:
            return {"raw_extraction": response_text}
    except Exception as e:
        print(f"Entity extraction error: {e}")
        return {}

# ==================== QUESTION ANSWERING ====================

def answer_question(text, question, document_name, summary):
    """Answer a question based on document content"""
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )

    # Validate the prompt first
    if not validate_prompt(question, summary):
        return "Your question doesn't appear to be related to this document. Please ask something about the document content."
    
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful document assistant. Answer questions based ONLY on the provided document. If the answer is not in the document, say so."},
                {"role": "user", "content": f"""Based on this document, answer the following question:

Document:
{text[:4000]}

Question: {question}

Provide a clear, concise answer based only on information from the document."""}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
        # Save to database
        save_query(document_name, question, answer)
        
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

# ==================== MAIN INTERACTIVE SESSION ====================

def run_interactive_session(file_path):
    """Run an interactive Q&A session with a document"""
    print("\n" + "="*60)
    print("DOCUMENT INTELLIGENCE AGENT")
    print("="*60)
    
    # Load document
    print(f"\nLoading document: {file_path}")
    try:
        document_text = load_document(file_path)
        if not document_text:
            print("Error: Could not extract text from document")
            return
    except Exception as e:
        print(f"Error loading document: {e}")
        return
    
    document_name = Path(file_path).name
    
    # Generate summary
    print("\nGenerating summary...")
    summary = summarize_document(document_text)
    print(f"\nSUMMARY:\n{summary}")
    
    # Extract entities
    print("\nExtracting key entities...")
    entities = extract_entities(document_text)
    print(f"\nKEY ENTITIES:\n{json.dumps(entities, indent=2)}")
    
    # Q&A loop
    print("\n" + "-"*60)
    print("Ask questions about the document (type 'quit' to exit)")
    print("-"*60)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        print("\nThinking...")
        answer = answer_question(document_text, question, document_name, summary)
        print(f"\nANSWER:\n{answer}")
    
    # Show query history
    print("\n" + "-"*60)
    print("QUERY HISTORY:")
    print("-"*60)
    history = get_query_history(document_name, limit=5)
    for doc, q, a, ts in history:
        print(f"\n[{ts}]")
        print(f"Q: {q}")
        print(f"A: {a[:100]}...")

# ==================== MAIN ====================

if __name__ == "__main__":
    # Initialize database
    init_database()
    
    # Example: run with a test document
    # For now, create a simple test file
    test_file = "sample_document.txt"
    
    # Create a sample document if it doesn't exist
    if not os.path.exists(test_file):
        with open(test_file, 'w') as f:
            f.write("""
SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 15, 2024, 
between ABC Corporation ("Service Provider") and XYZ Industries ("Client").

1. SERVICES
The Service Provider agrees to provide software development and consulting services 
as requested by the Client. Services include system design, implementation, testing, 
and deployment.

2. PAYMENT TERMS
- Monthly retainer: $5,000
- Additional hourly work: $150 per hour
- Payment due within 30 days of invoice
- Late payment penalty: 2% per month

3. TERM AND TERMINATION
- Initial term: 12 months from the date above
- Either party may terminate with 30 days written notice
- Upon termination, all project deliverables must be transferred to Client

4. CONFIDENTIALITY
Both parties agree to maintain confidentiality of proprietary information 
shared during the engagement. This includes source code, business strategies, 
and client data.

5. LIABILITY
Service Provider's total liability shall not exceed the total fees paid in the 
preceding 12 months. This excludes data breaches and gross negligence.
""")
        print(f"Created sample document: {test_file}")
    
    # Run the interactive session
    run_interactive_session(test_file)

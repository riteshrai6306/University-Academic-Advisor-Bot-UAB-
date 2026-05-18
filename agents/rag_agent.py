# ================================================================
# agents/rag_agent.py
# ================================================================
# WHAT: RAG (Retrieval Augmented Generation) Agent
# WHY:  Student questions about course policies, prerequisites,
#       syllabus, fees etc. live in PDFs — we can't SQL query them.
#       RAG lets us search PDFs semantically and answer accurately.
#
# FLOW:
#   PDF files → split into chunks → embed into vectors → FAISS store
#   Student question → embed → search FAISS → top chunks → GPT answer
# ================================================================
 
 
# ── IMPORTS ─────────────────────────────────────────────────────
 
import os
import re

# WHY: OCR support for scanned PDF pages
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_DPI = 300

# WHY: loads our API keys from .env file automatically
from dotenv import load_dotenv

# WHY: LangChain's PDF loader — reads PDF files page by page
from langchain_community.document_loaders import PyPDFLoader
import pdfplumber # WHY: extracts tables from PDFs as structured data

# WHY: splits large PDF text into smaller overlapping chunks
# WHY: RecursiveCharacterTextSplitter is the best default splitter —
#      it tries to split on paragraphs → sentences → words in order
from langchain_text_splitters import RecursiveCharacterTextSplitter

# WHY: converts text chunks into vector numbers (embeddings)
# WHY: OpenAI's embedding model is accurate & works with GPT-4o
from langchain_openai import OpenAIEmbeddings

# WHY: FAISS is our vector store — stores & searches embeddings locally
# WHY: no server needed, just local files in vectorstore/
from langchain_community.vectorstores import FAISS

# WHY: GPT-4o is our LLM — synthesizes the final answer from chunks
from langchain_openai import ChatOpenAI

# WHY: builds our RAG chain — connects retriever → LLM
from langchain_core.runnables import RunnablePassthrough

# WHY: formats our custom prompt template for RAG answers
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser


# ── LOAD ENVIRONMENT VARIABLES ──────────────────────────────────
 
# WHY: reads OPENAI_API_KEY from .env file
# NOTE: must be called before any OpenAI calls
load_dotenv()

# ── CONSTANTS ───────────────────────────────────────────────────
 
# WHY: this is where our PDF files live
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")
 
# WHY: this is where FAISS saves its index files after ingestion
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
 
# WHY: full paths to the two FAISS files that get auto-created
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
FAISS_PKL_PATH   = os.path.join(VECTORSTORE_DIR, "index.pkl")

# WHY: chunk size = 1000 chars gives enough context per chunk
# WHY: overlap = 200 chars prevents losing meaning at chunk boundaries
# NOTE: tune these if answers feel incomplete or too broad
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
 
# WHY: top 5 chunks retrieved per question — balance of context vs noise
TOP_K_CHUNKS = 5

# WHY: cache objects to avoid repeated expensive reloads
_rag_vectorstore = None
_rag_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

 
 
# ── PROMPT TEMPLATE ─────────────────────────────────────────────
 
# WHY: custom prompt makes GPT answer like a university advisor
# WHY: {context} = retrieved chunks, {question} = student's question
# NOTE: "If not in context, say you don't know" prevents hallucination
RAG_PROMPT_TEMPLATE = """
You are a helpful and professional University Academic Advisor.
Use the following context from university documents to answer the student's question.
If the answer is not found in the context, say:
"I don't have that information in the current university documents."

If the context contains a table (in markdown format with | symbols),
present it as a proper formatted table in your answer exactly as structured.

Context:
{context}

Student Question: {question}

Your Answer:
"""
 
# WHY: PromptTemplate wraps our string into a LangChain-compatible format
RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def is_low_quality_text(text: str) -> bool:
    """
    WHAT: Detects if extracted text is garbage (scanned page with no real text layer).
    WHY:  PyPDFLoader sometimes extracts broken characters from scanned pages.
    """
    text = text.strip()
    if not text:
        return True

    words = text.split()
    if len(words) < 10:
        return True

    # WHY: scanned pages often produce lots of single-character "words"
    one_letter_words = sum(1 for word in words if len(word) == 1)
    if one_letter_words / len(words) > 0.10:
        return True

    # WHY: very short average word length = character splitting (OCR artifact)
    avg_word_len = sum(len(word) for word in words) / len(words)
    if avg_word_len < 3:
        return True

    return False

def ocr_pdf_page(pdf_path: str, page_number: int) -> str:
    """
    WHAT: Converts a single scanned PDF page to text using OCR.
    WHY:  Scanned pages have no extractable text — we convert to image first.
    """
    images = convert_from_path(
        pdf_path,
        dpi=OCR_DPI,
        first_page=page_number,
        last_page=page_number,
    )

    if not images:
        return ""

    text = pytesseract.image_to_string(images[0], lang="eng")

    # WHY: clean up common OCR artifacts
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    return text

def apply_ocr_to_low_quality_pages(pdf_path: str, documents: list) -> list:
    """
    WHAT: Loops through all pages — replaces low quality text with OCR text.
    WHY:  Auto-handles mixed PDFs (some pages normal, some scanned).
    """
    improved_documents = []

    for doc in documents:
        if is_low_quality_text(doc.page_content):
            page_number = doc.metadata.get("page", 1)
            print(f"   OCR applied → page {page_number} of {pdf_path}")

            ocr_text = ocr_pdf_page(pdf_path, page_number)

            if ocr_text:
                doc.page_content = ocr_text  # WHY: replace bad text with OCR text

        improved_documents.append(doc)

    return improved_documents

# ================================================================
# HELPER — EXTRACT TABLES FROM A PDF PAGE AS MARKDOWN
# ================================================================

def extract_tables_from_pdf(pdf_path: str) -> list:
    """
    WHAT: Extracts tables from a PDF and converts them to markdown chunks.
    WHY:  PyPDFLoader destroys table structure — pdfplumber preserves rows/columns.
    RETURNS: list of LangChain Document objects, one per table found
    """
    from langchain_core.documents import Document

    table_documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()

            for table in tables:
                if not table:
                    continue

                # Convert table rows to markdown format
                markdown_rows = []
                for i, row in enumerate(table):
                    # Clean None values
                    cleaned_row = [cell if cell else "" for cell in row]
                    markdown_rows.append("| " + " | ".join(cleaned_row) + " |")
                    # Add header separator after first row
                    if i == 0:
                        markdown_rows.append("| " + " | ".join(["---"] * len(cleaned_row)) + " |")

                markdown_table = "\n".join(markdown_rows)

                # Store as a Document chunk with metadata
                doc = Document(
                    page_content=markdown_table,
                    metadata={
                        "source": pdf_path,
                        "page": page_num + 1,
                        "type": "table"
                    }
                )
                table_documents.append(doc)

    return table_documents

# ================================================================
# STEP 1 — LOAD PDFs
# ================================================================
 
def load_pdfs() -> list:
    """
    WHAT: Loads all PDF files from data/pdfs/ folder.
    WHY:  Each PDF (course catalog, syllabus, policy doc) is loaded
          as a list of LangChain Document objects (one per page).
 
    RETURNS: list of Document objects from all PDFs combined
    """
 
    # WHY: get all .pdf files from the pdfs directory
    pdf_files = [
        f for f in os.listdir(PDF_DIR)
        if f.endswith(".pdf")
    ]
 
    if not pdf_files:
        print("WARNING: No PDFs found in data/pdfs/ — add your university PDFs!")
        return []
 
    all_documents = []
 
    for pdf_file in pdf_files:
        # WHY: build full path to the PDF file
        pdf_path = os.path.join(PDF_DIR, pdf_file)
 
        # WHY: PyPDFLoader reads each page as a separate Document object
        # NOTE: each Document has .page_content (text) and .metadata (source, page)
        
        # WHY: PyPDFLoader extracts regular text content & apply ocr for scanned pages
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"OK Loaded: {pdf_file} — {len(documents)} pages")

        # WHY: auto-detects scanned pages and applies OCR where needed
        documents = apply_ocr_to_low_quality_pages(pdf_path, documents)

        all_documents.extend(documents)

        # WHY: pdfplumber separately extracts tables as markdown chunks
        table_docs = extract_tables_from_pdf(pdf_path)
        if table_docs:
            print(f"   Tables found: {len(table_docs)} table(s) extracted as markdown")
            all_documents.extend(table_docs)
 
    print(f"\nTotal pages loaded: {len(all_documents)}")
    return all_documents
 
 
# ================================================================
# STEP 2 — SPLIT INTO CHUNKS
# ================================================================
 
def split_documents(documents: list) -> list:
    """
    WHAT: Splits large PDF pages into smaller overlapping chunks.
    WHY:  LLMs have context limits — we can't pass entire PDFs.
          Smaller chunks = more precise retrieval.
          Overlap = no meaning lost at chunk boundaries.
 
    RETURNS: list of smaller Document chunks
    """
 
    # WHY: RecursiveCharacterTextSplitter splits on:
    #      paragraphs (\n\n) → sentences (\n) → words ( ) → characters
    #      in that priority order — most natural split possible
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # WHY: 1000 chars = ~250 tokens, good for context
        chunk_overlap=CHUNK_OVERLAP, # WHY: 200 char overlap = safety net at boundaries
        length_function=len,         # WHY: measure chunk size by character count
    )
 
    chunks = splitter.split_documents(documents)
 
    print(f"OK Split into {len(chunks)} chunks")
    print(f"   Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
 
    return chunks
 
 
# ================================================================
# STEP 3 — INGEST: PDF → CHUNKS → EMBEDDINGS → FAISS
# ================================================================

def ingest_pdfs():
    print("\nStarting PDF ingestion...\n")
 
    documents = load_pdfs()
    if not documents:
        return
 
    chunks = split_documents(documents)
 
    print("\nCreating embeddings...")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=_rag_embeddings)
 
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
 
    print(f"\nOK FAISS index saved to: {VECTORSTORE_DIR}")
    print(f"SUCCESS Ingestion complete — {len(chunks)} chunks indexed.")

# def ingest_pdfs():
#     """
#     WHAT: Full ingestion pipeline — PDF → chunks → embeddings → FAISS.
#     WHY:  Run this ONCE when you add new PDFs.
#           After this, FAISS index is saved locally and reused every time.
 
#     SAVES:
#         vectorstore/index.faiss  ← vector index (numbers)
#         vectorstore/index.pkl    ← chunk texts + metadata
#     """
 
#     print("\nPROCESSING Starting PDF ingestion...\n")
 
#     # STEP 1: Load PDFs
#     documents = load_pdfs()
#     if not documents:
#         return
 
#     # STEP 2: Split into chunks
#     chunks = split_documents(documents)

#     print("\nCreating embeddings...")
#     vectorstore = FAISS.from_documents(documents=chunks, embedding=_rag_embeddings)
 
#     # STEP 3: Create embeddings
#     # WHY: OpenAIEmbeddings converts each chunk into a vector (list of numbers)
#     # WHY: text-embedding-3-small is fast, cheap, and accurate enough for our use
#     print("\nPROCESSING Creating embeddings (this may take a moment)...")
#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-small"
#     )
 
#     # STEP 4: Store in FAISS
#     # WHY: FAISS.from_documents embeds all chunks and builds the searchable index
#     print("PROCESSING Building FAISS index...")
#     vectorstore = FAISS.from_documents(
#         documents=chunks,
#         embedding=embeddings
#     )
 
#     # STEP 5: Save FAISS index to disk
#     # WHY: so we don't re-embed every time the app restarts (saves time + API cost)
#     os.makedirs(VECTORSTORE_DIR, exist_ok=True)
#     vectorstore.save_local(VECTORSTORE_DIR)
 
#     print(f"\nOK FAISS index saved to: {VECTORSTORE_DIR}")
#     print(f"   Files created:")
#     print(f"   → vectorstore/index.faiss  (vector index)")
#     print(f"   → vectorstore/index.pkl    (chunk metadata)")
#     print(f"\nSUCCESS Ingestion complete! {len(chunks)} chunks indexed.")
 
 
# ================================================================
# STEP 4 — LOAD FAISS INDEX (at query time)
# ================================================================
def load_vectorstore() -> FAISS:
    global _rag_vectorstore
 
    if _rag_vectorstore is not None:
        return _rag_vectorstore
 
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            "FAISS index not found! Run ingest_pdfs() first.\n"
            f"Expected at: {FAISS_INDEX_PATH}"
        )
 
    _rag_vectorstore = FAISS.load_local(
        folder_path=VECTORSTORE_DIR,
        embeddings=_rag_embeddings,
        allow_dangerous_deserialization=True
    )
 
    print(f"OK Vectorstore loaded — {_rag_vectorstore.index.ntotal} chunks indexed")
    return _rag_vectorstore
 
# def load_vectorstore() -> FAISS:
#     """
#     WHAT: Loads the saved FAISS index from disk.
#     WHY:  Instead of re-embedding every time, we load the saved index.
#           This makes every query fast — no API call for embeddings at runtime.
 
#     RETURNS: FAISS vectorstore object ready for similarity search
#     """
 
#     global _rag_vectorstore
#     if _rag_vectorstore is not None:
#         return _rag_vectorstore
 
#     # WHY: check if FAISS index exists before trying to load
#     if not os.path.exists(FAISS_INDEX_PATH):
#         raise FileNotFoundError(
#             "FAISS index not found!\n"
#             "Run ingest_pdfs() first to create the index.\n"
#             f"Expected at: {FAISS_INDEX_PATH}"
#         )
 
#     # WHY: allow_dangerous_deserialization=True needed for loading pkl files
#     # NOTE: safe here because WE created these files ourselves
#     _rag_vectorstore = FAISS.load_local(
#         folder_path=VECTORSTORE_DIR,
#         embeddings=_rag_embeddings,
#         allow_dangerous_deserialization=True
#     )
 
#     return _rag_vectorstore


# ================================================================
# STEP 5 — RUN RAG AGENT (called by LangGraph node)
# ================================================================
 

def run_rag_agent(question: str) -> dict:
    """
    WHAT: Main function — takes student question, returns RAG answer.
    WHY:  This is what LangGraph calls when router decides RAG is needed.

    ARGS:
        question (str): student's natural language question

    RETURNS:
        dict:
            answer  (str)  → GPT's answer based on retrieved chunks
            sources (list) → list of PDF filenames used to answer
            chunks  (list) → actual retrieved text chunks (for debug)
    """

    print(f"\nRAG Agent received: {question}")
 
    # STEP 1: Load FAISS index from disk
    vectorstore = load_vectorstore()
 
    # STEP 2: Create retriever
    # WHY: retriever searches FAISS for top-k most similar chunks
    # WHY: search_type="similarity" = cosine similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_CHUNKS}  # WHY: fetch top 4 most relevant chunks
    )
 
    # STEP 3: Reuse the global LLM for faster responses
    llm = _rag_llm
 
    # STEP 4: Build RetrievalQA chain
    # NEW ✅ — LCEL style chain
    # WHY: modern LangChain way to build retrieval chain
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    # WHY: formats retrieved docs into one string for the prompt
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # WHY: chain = retrieve → format → prompt → LLM → parse
    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
         | RAG_PROMPT
         | llm
         | StrOutputParser()
)

    answer      = chain.invoke(question)
    source_docs = retriever.invoke(question)
 
    # WHY: extract actual chunk texts for debug/transparency mode
    chunks = [doc.page_content for doc in source_docs]
 
    print(f"RAG answer generated")
    print(f"   Sources used: {len(source_docs)} documents")
 
    return {
        "answer":  answer,
        "chunks":  chunks
    }
 
 
# ================================================================
# QUICK TEST — run this file directly to test RAG
# ================================================================
 
if __name__ == "__main__":
    # WHY: lets you test RAG independently without running the full app
    # HOW: run → python agents/rag_agent.py
 
    print("=" * 50)
    print("  RAG AGENT — Quick Test")
    print("=" * 50)
 
    # Step 1: Ingest PDFs first
    ingest_pdfs()
 
    # Step 2: Test a question
    result = run_rag_agent("What are the prerequisites for Machine Learning?")
 
    print("\n📝 ANSWER:")
    print(result["answer"])
 
    print("\n📎 SOURCES:")
    for src in result["sources"]:
        print(f"  → {src}")
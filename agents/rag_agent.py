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
import pickle
import re

# WHY: OCR support for scanned PDF pages
from pdf2image import convert_from_path
import pytesseract

# WHY: loads our API keys from .env file automatically
from dotenv import load_dotenv

# WHY: LangChain's PDF loader — reads PDF files page by page
from langchain_community.document_loaders import PyPDFLoader

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

# WHY: cache objects to avoid repeated expensive reloads
_rag_vectorstore = None
_rag_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
 
# WHY: chunk size = 1000 chars gives enough context per chunk
# WHY: overlap = 200 chars prevents losing meaning at chunk boundaries
# NOTE: tune these if answers feel incomplete or too broad
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
 
# WHY: top 8 chunks retrieved per question — balance of context vs noise
TOP_K_CHUNKS = 4

# WHY: dpi used for converting scanned PDF pages to images for OCR
OCR_DPI = 300
 
 
# ── PROMPT TEMPLATE ─────────────────────────────────────────────
 
# WHY: custom prompt makes GPT answer like a university advisor
# WHY: {context} = retrieved chunks, {question} = student's question
# NOTE: "If not in context, say you don't know" prevents hallucination
RAG_PROMPT_TEMPLATE = """
You are a helpful and professional University Academic Advisor.
Use the following context from university documents to answer the student's question.
If the answer is not found in the context, say:
"I don't have that information in the current university documents."
 
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
 

def ocr_pdf_page(pdf_path: str, page_number: int) -> str:
    """
    WHAT: Converts a scanned PDF page into text using OCR.
    WHY:  Some uploaded PDFs are scanned images and have no extractable text.
    """

    images = convert_from_path(
        pdf_path,
        dpi=OCR_DPI,
        first_page=page_number,
        last_page=page_number,
    )

    if not images:
        return ""

    page_image = images[0]
    text = pytesseract.image_to_string(page_image, lang="eng")
    
    # Clean up common OCR artifacts
    text = text.strip()
    # Remove excessive spaces between characters (common OCR issue)
    text = re.sub(r'(?<=\w)\s(?=\w)', '', text)
    # Fix common OCR mistakes
    text = re.sub(r'\s+', ' ', text)
    
    return text

def is_low_quality_text(text: str) -> bool:
    """
    WHAT: Detects text that is likely broken or garbled from scanned PDFs.
    WHY:  Some scanned pages have a text layer with poor OCR output in the PDF.
    """

    text = text.strip()
    if not text:
        return True

    words = text.split()
    if len(words) < 10:  # Lower threshold
        return True

    one_letter_words = sum(1 for word in words if len(word) == 1)
    if one_letter_words / len(words) > 0.10:  # Lower threshold
        return True

    # Check for excessive spaces between characters (OCR artifact)
    if re.search(r'(?:\b[a-zA-Z]\b[\s\n]+){5,}', text):
        return True
    
    # Check for very short average word length (indicates character splitting)
    avg_word_len = sum(len(word) for word in words) / len(words)
    if avg_word_len < 3:
        return True

    return False


def apply_ocr_to_low_quality_pages(pdf_path: str, documents: list) -> list:
    """
    WHAT: OCR is currently disabled due to missing dependencies.
    WHY:  Poppler and Tesseract need to be installed for OCR to work.
    TODO: Install OCR dependencies and re-enable this function.
    """
    
    print(f"   Skipping OCR for {pdf_path} (dependencies not available)")
    return documents


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
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # WHY: if the PDF is scanned, low-quality pages will be detected and OCR will extract text
        documents = apply_ocr_to_low_quality_pages(pdf_path, documents)

        print(f"OK Loaded: {pdf_file} — {len(documents)} pages")
        all_documents.extend(documents)
 
    print(f"\nDOCS Total pages loaded: {len(all_documents)}")
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
# STEP 3 — CREATE EMBEDDINGS & SAVE TO FAISS
# ================================================================
 
def ingest_pdfs():
    """
    WHAT: Full ingestion pipeline — PDF → chunks → embeddings → FAISS.
    WHY:  Run this ONCE when you add new PDFs.
          After this, FAISS index is saved locally and reused every time.
 
    SAVES:
        vectorstore/index.faiss  ← vector index (numbers)
        vectorstore/index.pkl    ← chunk texts + metadata
    """
 
    print("\nPROCESSING Starting PDF ingestion...\n")
 
    # STEP 1: Load PDFs
    documents = load_pdfs()
    if not documents:
        return
 
    # STEP 2: Split into chunks
    chunks = split_documents(documents)
 
    # STEP 3: Create embeddings
    # WHY: OpenAIEmbeddings converts each chunk into a vector (list of numbers)
    # WHY: text-embedding-3-small is fast, cheap, and accurate enough for our use
    print("\nPROCESSING Creating embeddings (this may take a moment)...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
 
    # STEP 4: Store in FAISS
    # WHY: FAISS.from_documents embeds all chunks and builds the searchable index
    print("PROCESSING Building FAISS index...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
 
    # STEP 5: Save FAISS index to disk
    # WHY: so we don't re-embed every time the app restarts (saves time + API cost)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
 
    print(f"\nOK FAISS index saved to: {VECTORSTORE_DIR}")
    print(f"   Files created:")
    print(f"   → vectorstore/index.faiss  (vector index)")
    print(f"   → vectorstore/index.pkl    (chunk metadata)")
    print(f"\nSUCCESS Ingestion complete! {len(chunks)} chunks indexed.")
 
 
# ================================================================
# STEP 4 — LOAD FAISS INDEX (at query time)
# ================================================================
 
def load_vectorstore() -> FAISS:
    """
    WHAT: Loads the saved FAISS index from disk.
    WHY:  Instead of re-embedding every time, we load the saved index.
          This makes every query fast — no API call for embeddings at runtime.
 
    RETURNS: FAISS vectorstore object ready for similarity search
    """
 
    global _rag_vectorstore
    if _rag_vectorstore is not None:
        return _rag_vectorstore
 
    # WHY: check if FAISS index exists before trying to load
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            "FAISS index not found!\n"
            "Run ingest_pdfs() first to create the index.\n"
            f"Expected at: {FAISS_INDEX_PATH}"
        )
 
    # WHY: allow_dangerous_deserialization=True needed for loading pkl files
    # NOTE: safe here because WE created these files ourselves
    _rag_vectorstore = FAISS.load_local(
        folder_path=VECTORSTORE_DIR,
        embeddings=_rag_embeddings,
        allow_dangerous_deserialization=True
    )
 
    return _rag_vectorstore


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
        "sources": source_docs,
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
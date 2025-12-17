from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.documents import Document
from typing import List, Optional, Union
import pandas as pd

load_dotenv()

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

location = "/home/abhidharsh-fgil/FGIL Projects/FAISS/documents/tele_issues.xlsx"


def process_docx(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Process DOCX file and return chunks."""
    try:
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        print(f"📄 Loaded {len(documents)} document(s)")

        # Add metadata
        filename = os.path.basename(file_path)
        for doc in documents:
            doc.metadata["filename"] = filename
            doc.metadata["source"] = file_path
            doc.metadata["file_type"] = "docx"
            doc.metadata["indexed_at"] = datetime.now().isoformat()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✂️ Split into {len(chunks)} chunks")

        return chunks
    except Exception as e:
        print(f"❌ Error processing DOCX: {str(e)}")
        return []


def process_csv(
    file_path: str,
    text_columns: Optional[List[str]] = None,
    row_format: str = "structured",
) -> List[Document]:
    """
    Process CSV file - each row becomes a separate document.

    Args:
        file_path: Path to CSV file
        text_columns: Specific columns to include (None = all columns)
        row_format: 'structured' (key: value) or 'natural' (sentence-like)

    Returns:
        List of Document objects
    """
    try:
        df = pd.read_csv(file_path)
        print(f"📊 Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

        # Select columns
        if text_columns:
            df = df[text_columns]

        filename = os.path.basename(file_path)
        documents = []

        for idx, row in df.iterrows():
            # Convert row to text
            if row_format == "structured":
                # Format: "Column1: Value1 | Column2: Value2 | ..."
                text_content = " | ".join(
                    [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
                )
            else:
                # Natural format: "The Column1 is Value1, Column2 is Value2..."
                text_content = ", ".join(
                    [f"{col} is {val}" for col, val in row.items() if pd.notna(val)]
                )

            # Create document with rich metadata
            doc = Document(
                page_content=text_content,
                metadata={
                    "filename": filename,
                    "source": file_path,
                    "file_type": "csv",
                    "row_index": idx,
                    "indexed_at": datetime.now().isoformat(),
                    # Store original row data as metadata for reference
                    **{
                        f"col_{col}": str(val)
                        for col, val in row.items()
                        if pd.notna(val)
                    },
                },
            )
            documents.append(doc)

        print(f"📝 Created {len(documents)} documents from CSV rows")
        return documents
    except Exception as e:
        print(f"❌ Error processing CSV: {str(e)}")
        return []


def process_excel(
    file_path: str,
    sheet_name: Optional[Union[str, int, List]] = None,
    text_columns: Optional[List[str]] = None,
    row_format: str = "structured",
) -> List[Document]:
    """
    Process Excel file - each row becomes a separate document.

    Args:
        file_path: Path to Excel file
        sheet_name: Specific sheet(s) to process (None = all sheets)
        text_columns: Specific columns to include (None = all columns)
        row_format: 'structured' (key: value) or 'natural' (sentence-like)

    Returns:
        List of Document objects
    """
    # Read Excel file
    try:
        # Read Excel file
        excel_file = pd.ExcelFile(file_path)
        if sheet_name is None:
            sheet_names = excel_file.sheet_names
            print(f"📗 Found {len(sheet_names)} sheet(s): {sheet_names}")
        else:
            sheet_names = (
                [sheet_name] if isinstance(sheet_name, (str, int)) else sheet_name
            )

        filename = os.path.basename(file_path)
        documents = []

        for sheet in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            print(f"📊 Sheet '{sheet}': {len(df)} rows, {len(df.columns)} columns")

            # Select columns
            if text_columns:
                available_cols = [col for col in text_columns if col in df.columns]
                df = df[available_cols]

            for idx, row in df.iterrows():
                # Convert row to text
                if row_format == "structured":
                    text_content = " | ".join(
                        [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
                    )
                else:
                    text_content = ", ".join(
                        [f"{col} is {val}" for col, val in row.items() if pd.notna(val)]
                    )

                # Skip empty rows
                if not text_content.strip():
                    continue

                # Create document with rich metadata
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "filename": filename,
                        "source": file_path,
                        "file_type": "excel",
                        "sheet_name": str(sheet),
                        "row_index": idx,
                        "indexed_at": datetime.now().isoformat(),
                        **{
                            f"col_{col}": str(val)
                            for col, val in row.items()
                            if pd.notna(val)
                        },
                    },
                )
                documents.append(doc)

        print(f"📝 Created {len(documents)} documents from Excel rows")
        return documents
    except Exception as e:
        print(f"❌ Error processing Excel: {str(e)}")
        return []


def process_pdf(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Process PDF file and return chunks.

    Args:
        file_path: Path to PDF file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        List of Document objects (chunks)
    """
    try:
        # Load PDF - PyPDFLoader loads each page as a separate document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"📄 Loaded PDF with {len(documents)} page(s)")

        # Add/enhance metadata
        filename = os.path.basename(file_path)
        for doc in documents:
            doc.metadata["filename"] = filename
            doc.metadata["source"] = file_path
            doc.metadata["file_type"] = "pdf"
            doc.metadata["indexed_at"] = datetime.now().isoformat()
            # PyPDFLoader already adds 'page' metadata, but ensure it exists
            if "page" not in doc.metadata:
                doc.metadata["page"] = 0

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✂️ Split into {len(chunks)} chunks")

        return chunks
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        return []


def store_document_embeddings(
    file_path: str,
    index_path: str = "faiss_index",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    # ✅ ADDED: Parameters for CSV/Excel
    text_columns: Optional[List[str]] = None,
    sheet_name: Optional[Union[str, int, List]] = None,
    row_format: str = "structured",
):
    """
    Load a document, create embeddings, and store in FAISS.
    Supports .docx, .csv, .xlsx, and .xls files.

    Args:
        file_path: Path to the file (.docx, .csv, .xlsx, .xls)
        index_path: Path to save FAISS index
        chunk_size: Size of text chunks (for docx only)
        chunk_overlap: Overlap between chunks (for docx only)
        text_columns: Columns to include for CSV/Excel (None = all)
        sheet_name: Sheet(s) to process for Excel (None = all)
        row_format: 'structured' or 'natural' for CSV/Excel
        pdf_loader_type: Loader type for PDF ('pypdf', 'pymupdf', 'pdfminer', 'unstructured')

    Returns:
        vector_store: The FAISS vector store object, or None if failed
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    # Detect file type
    file_extension = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    print(f"🔍 Processing file: {filename} (type: {file_extension})")

    # ========================================
    # Process based on file type
    # ========================================

    if file_extension == ".docx":
        chunks = process_docx(file_path, chunk_size, chunk_overlap)

    elif file_extension == ".csv":
        chunks = process_csv(
            file_path, text_columns=text_columns, row_format=row_format
        )

    elif file_extension in [".xlsx", ".xls"]:
        chunks = process_excel(
            file_path,
            sheet_name=sheet_name,
            text_columns=text_columns,
            row_format=row_format,
        )

    elif file_extension == ".pdf":
        chunks = process_pdf(file_path, chunk_size, chunk_overlap)

    else:
        print(f"❌ Unsupported file format: {file_extension}")
        print("   Supported formats: .docx, .csv, .xlsx, .xls")
        return None

    if not chunks:
        print("❌ No content extracted from file")
        return None

    print(f"📦 Total chunks/documents to embed: {len(chunks)}")

    # Create embeddings and store
    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

        # ========================================
        # KEY CHANGE: Check if index exists
        # ========================================
        if os.path.exists(index_path):
            # Load existing index and ADD new documents
            print(f"📂 Found existing index at '{index_path}'")
            vector_store = FAISS.load_local(
                index_path, embeddings_model, allow_dangerous_deserialization=True
            )

            # Add new documents to existing index
            vector_store.add_documents(chunks)
            print(f"➕ Added '{filename}' to existing index")
        else:
            # Create new index
            print(f"🆕 Creating new index at '{index_path}'")
            vector_store = FAISS.from_documents(chunks, embeddings_model)

        # Save index
        vector_store.save_local(index_path)
        print(f"✅ FAISS index saved to '{index_path}'")

        return vector_store
    except Exception as e:
        print(f"❌ Error creating embeddings: {str(e)}")
        return None


def load_vector_store(index_path: str = "faiss_index"):
    """Load existing FAISS index from disk."""
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.load_local(
        index_path, embeddings_model, allow_dangerous_deserialization=True
    )
    print(f"📂 Loaded FAISS index from '{index_path}'")
    return vector_store


def list_indexed_files(vector_store):
    """List all files that have been indexed."""
    if vector_store is None:
        print("❌ No vector store available!")
        return []

    # Get sample of documents to extract filenames
    all_docs = vector_store.similarity_search("", k=10000)
    files = set()

    for doc in all_docs:
        filename = doc.metadata.get("filename", "Unknown")
        indexed_at = doc.metadata.get("indexed_at", "Unknown")
        files.add((filename, indexed_at))

    print("\n📂 Indexed Files:")
    print("-" * 50)
    for filename, indexed_at in sorted(files):
        print(f"  • {filename}")
        print(f"    Indexed at: {indexed_at}")

    return list(files)


# print(f"Generated {len(embeddings)} embeddings")
# print(f"Each embedding has {len(embeddings[0])} dimensions")

# ========================================
# 🔍 SEARCH OPERATIONS START HERE
# ========================================


def search(vector_store, query: str, k: int = 3):
    """Perform similarity search."""
    print(f"\n🔍 Searching for: '{query}'")
    print("=" * 60)

    results = vector_store.similarity_search_with_score(query, k=k)

    # Group results by file
    files_found = set()

    for i, (doc, score) in enumerate(results, 1):
        filename = doc.metadata.get("filename", "N/A")
        files_found.add(filename)

        print(f"\n📄 Result {i} (Score: {score:.4f})")
        print(f"📁 File: {filename}")
        print("-" * 50)
        print(doc.page_content)

    print(f"\n📊 Results from {len(files_found)} file(s): {', '.join(files_found)}")

    return results


# ========================================
# 🤖 NEW: ANSWER QUESTION FUNCTION
# ========================================


def answer_question(
    vector_store,
    question: str,
    k: int = 3,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
):
    """
    Answer a question using retrieved context from FAISS.

    Args:
        vector_store: FAISS vector store
        question: User's question
        k: Number of chunks to retrieve
        model: OpenAI model to use
        temperature: LLM temperature (0 = deterministic)

    Returns:
        dict: Contains answer, sources, and context used
    """
    print(f"\n❓ Question: {question}")
    print("=" * 60)

    # Step 1: Retrieve relevant chunks
    print("\n📚 Retrieving relevant context...")
    retrieved_docs = vector_store.similarity_search(question, k=k)

    # Step 2: Build context from retrieved chunks
    context_parts = []
    sources = []
    files_used = set()

    for i, doc in enumerate(retrieved_docs, 1):
        filename = doc.metadata.get("filename", "Unknown")
        files_used.add(filename)

        context_parts.append(f"[Source: {filename}]\n{doc.page_content}")
        sources.append(
            {
                "chunk": i,
                "filename": filename,
                "content_preview": doc.page_content[:100] + "...",
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Create prompt
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the answer is not found in the context, say "I couldn't find the answer in the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    # Step 4: Call LLM
    print("🤖 Generating answer...")
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    answer = response.content

    # Step 5: Display results
    print("\n" + "=" * 60)
    print("💡 ANSWER:")
    print("=" * 60)
    print(answer)

    print("\n📚 SOURCES USED:")
    print("-" * 40)
    for source in sources:
        print(f"  📄 Chunk {source['chunk']}: {source['filename']}")
        print(f"     Preview: {source['content_preview']}")

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "context": context,
    }


# Example searches
# search("Agent visa limit")
# search("How to create a new user?")
# search("password reset process")
# search("admin permissions")

# location = "/home/abhidharsh-fgil/FGIL Projects/FAISS/documents/67ee1cb7-d927-4ee6-9cbd-3f82e1ea0c0aFG_English_Telecom Technician  - IOT DevicesSystem_TELQ6210_4.0"

store_document_embeddings(location)

# Store only ONCE
# vector_store = load_vector_store()

# Search MULTIPLE times using the same vector_store
# search(vector_store, "How to create a new user?")
# search(vector_store, "password reset process")
# search(vector_store, "admin permissions")
# result = answer_question(vector_store, "steps for user creation")
# print(result)

from langchain_community.document_loaders import Docx2txtLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

location = "/home/abhidharsh-fgil/FGIL Projects/FAISS/documents/AdminUserManual.docx"


def store_document_embeddings(
    file_path: str,
    index_path: str = "faiss_index",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Load a document, create embeddings, and store in FAISS.

    Args:
        file_path: Path to the .docx file
        index_path: Path to save FAISS index
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        vector_store: The FAISS vector store object
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    # Load document
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    print(f"📄 Loaded {len(documents)} document(s)")

    # Add metadata
    filename = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["filename"] = filename
        doc.metadata["source"] = file_path
        doc.metadata["indexed_at"] = datetime.now().isoformat()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks")

    # Create embeddings and store
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

location = "/home/abhidharsh-fgil/FGIL Projects/FAISS/documents/Staff portal Grouping functionality - User Manual.docx"

# store_document_embeddings(location)

# Store only ONCE
vector_store = load_vector_store()

# Search MULTIPLE times using the same vector_store
# search(vector_store, "How to create a new user?")
# search(vector_store, "password reset process")
# search(vector_store, "admin permissions")
result = answer_question(vector_store, "steps for user creation")
print(result)

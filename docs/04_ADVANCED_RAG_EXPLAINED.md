# Advanced RAG System Explained - Step by Step

## **What Makes This "Advanced"?**

This goes beyond basic RAG by adding:

1. **Multiple document formats** (text, CSV, JSON)
2. **Metadata tracking** (source type, topic)
3. **Contextual compression** (filters irrelevant info)
4. **Advanced chunking** (smarter text splitting)
5. **Metadata filtering** (query specific document types)

Think of it as building a **data catalog with AI search** - familiar to data engineers!

---

## **Step-by-Step Breakdown:**

### **1. Create Diverse Documents**

```python
def create_diverse_documents():
```

Creates **3 different document types:**

**A. Text File** (`data_eng_concepts.txt`)

- Contains concepts: ETL vs ELT, Data Lakehouse, Streaming vs Batch, Data Quality
- **Why:** Unstructured knowledge, like documentation

**B. CSV File** (`de_tools.csv`)

- Columns: tool, category, use_case, popularity
- Contains: Airflow, dbt, Kafka, Snowflake, Databricks
- **Why:** Structured data, like tool inventories

**C. JSON File** (`architectures.json`)

- Nested structure with patterns array
- Contains: Lambda, Kappa, Medallion architectures
- **Why:** Semi-structured data, like configuration files

**Real-world parallel:** Just like you work with multiple data sources (databases, APIs, files), RAG systems need to handle various formats!

---

### **2. Load Documents with Specialized Loaders**

```python
text_loader = TextLoader("data_eng_concepts.txt")
csv_loader = CSVLoader("de_tools.csv")
json_loader = JSONLoader(
    file_path="architectures.json",
    jq_schema=".patterns[]",
    text_content=False
)
```

**TextLoader:**

- Simple file reader
- Treats entire file as one document

**CSVLoader:**

- Converts each row into a document
- Columns become part of the content
- Result: 5 documents (one per tool)

**JSONLoader:**

- `jq_schema=".patterns[]"` - uses JQ query language (like SQL for JSON)
- Extracts each pattern as a separate document
- `text_content=False` - preserves JSON structure
- Result: 3 documents (Lambda, Kappa, Medallion)

**Why different loaders?** Each format needs specific parsing logic - like using different connectors in data pipelines!

---

### **3. Add Metadata Tagging**

```python
for doc in text_docs:
    doc.metadata["source_type"] = "text"
    doc.metadata["topic"] = "concepts"

for doc in csv_docs:
    doc.metadata["source_type"] = "csv"
    doc.metadata["topic"] = "tools"

for doc in json_docs:
    doc.metadata["source_type"] = "json"
    doc.metadata["topic"] = "architecture"
```

**Metadata is like table columns in a database:**

- Allows filtering and grouping
- Tracks data lineage
- Enables targeted retrieval

**Example metadata:**

```python
{
    "source_type": "json",
    "topic": "architecture",
    "source": "architectures.json"
}
```

**Data engineering parallel:** This is like adding partition keys or tags to your datasets!

---

### **4. Advanced Text Splitting**

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    length_function=len,
)
```

**Breaking it down:**

**`chunk_size=500`**

- Maximum 500 characters per chunk
- Larger than basic example (was 200) for more context

**`chunk_overlap=50`**

- 50 characters overlap between chunks
- Prevents breaking sentences mid-context

**`separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]`**

- **Tries to split in order:**
  1. Double newline (paragraph breaks) ← Best
  2. Single newline (line breaks)
  3. Period (sentence end)
  4. Exclamation/Question marks
  5. Comma
  6. Space
  7. Character-by-character ← Last resort

**Why "Recursive"?**

```
Text too long?
  ↓
Try splitting by "\n\n"
  ↓ Still too long?
Try splitting by "\n"
  ↓ Still too long?
Try splitting by "."
  ↓ ... and so on
```

**Result:** Smart chunks that respect document structure!

---

### **5. Create Vector Store**

```python
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="data_eng_knowledge"
)
```

**What happens:**

1. Each chunk → OpenAI embeddings → vector (1536 numbers)
2. Vectors stored in ChromaDB
3. `collection_name` = like a database table name

**Current state:**

- 11 chunks stored
- Each with content + metadata + vector

---

### **6. Create Base Retriever**

```python
base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

**Parameters:**

- `search_type="similarity"` - cosine similarity search
- `k=4` - retrieve top 4 most relevant chunks

**But wait... there's more!**

---

### **7. Add Contextual Compression (The Magic!)**

```python
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

**This is HUGE! Here's what it does:**

**Without compression:**

```
Question: "What is Lambda Architecture?"

Retrieved chunk 1: [Full 500 characters including irrelevant info]
Retrieved chunk 2: [Full 500 characters including irrelevant info]
Retrieved chunk 3: [Full 500 characters including irrelevant info]
Retrieved chunk 4: [Full 500 characters including irrelevant info]
```

**With compression:**

```
Question: "What is Lambda Architecture?"

Retrieved chunk 1: [Only extracts: "Lambda Architecture combines batch
and streaming layers... Pros: handles real-time data"]
Retrieved chunk 2: [Only relevant sentences extracted]
...
```

**How it works:**

```
1. Base retriever finds 4 chunks
   ↓
2. LLM reads each chunk
   ↓
3. LLM extracts ONLY relevant parts to the question
   ↓
4. Compressed, focused context sent to final LLM
```

**Benefits:**

- ✅ Less noise in context
- ✅ More accurate answers
- ✅ Lower token costs
- ✅ Fits more relevant info in context window

**Data engineering parallel:** Like applying WHERE clauses and projections to get only what you need!

---

### **8. Format Documents Function**

```python
def format_docs(docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source_type', 'unknown')} | "
        f"Topic: {doc.metadata.get('topic', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )
```

**Transforms retrieved docs into readable format:**

**Before:**

```python
[
    Document(page_content="...", metadata={"source_type": "json", "topic": "architecture"}),
    Document(page_content="...", metadata={"source_type": "text", "topic": "concepts"})
]
```

**After:**

```
Source: json | Topic: architecture
[content here]

Source: text | Topic: concepts
[content here]
```

This gives the LLM context about WHERE the information came from!

---

### **9. Build the RAG Chain**

```python
rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**Let's break down this pipeline:**

```python
{"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
```

**This creates a dictionary:**

- `context`: Question → compression_retriever → format_docs
- `question`: Question passes through unchanged

**Then the pipeline continues:**

```
→ prompt (inserts context + question into template)
→ llm (generates answer)
→ StrOutputParser() (extracts text)
```

---

## **Complete Query Flow:**

**Question:** "What's the difference between ETL and ELT?"

```
Step 1: Question Embedding
"What's the difference between ETL and ELT?"
  → Vector [0.123, -0.456, ...]

Step 2: Base Retrieval (k=4)
Search vectorstore
  → Find 4 most similar chunks

Retrieved chunks:
1. "ETL vs ELT: ETL (Extract, Transform, Load)..." [500 chars]
2. "Data Lakehouse: Combines data lake..." [500 chars]
3. "Streaming vs Batch: Batch processing..." [500 chars]
4. "Apache Airflow, Orchestration..." [500 chars]

Step 3: Contextual Compression 🔥
LLM reads each chunk with the question

Chunk 1: ✅ HIGHLY RELEVANT - Extract full content
Chunk 2: ❌ Not relevant - Discard
Chunk 3: ⚠️ Partially relevant - Extract only "Batch processing handles data..."
Chunk 4: ❌ Not relevant - Discard

Compressed context:
"ETL vs ELT: ETL (Extract, Transform, Load) processes data before loading.
ELT (Extract, Load, Transform) loads raw data first..."

Step 4: Format with Metadata
"Source: text | Topic: concepts
ETL vs ELT: ETL (Extract, Transform, Load)..."

Step 5: Create Prompt
"You are an AI assistant helping data engineers...
Context:
Source: text | Topic: concepts
ETL vs ELT: ETL...

Question: What's the difference between ETL and ELT?

Answer (be specific and reference the context):"

Step 6: LLM Generation
Receives prompt → Generates answer

Step 7: Parse Output
Extract text from LLM response

Final Answer:
"The main difference between ETL and ELT lies in the order of operations.
In ETL, data is extracted, transformed, then loaded. In ELT, data is
extracted, loaded as raw data, then transformed using the target system's
compute. Modern cloud warehouses prefer ELT..."
```

---

## **Visual Comparison:**

### **Basic RAG Flow:**

```
Question
  ↓
Embed Question
  ↓
Find 2 Similar Chunks
  ↓
Insert ALL Chunks into Prompt
  ↓
Generate Answer
```

### **Advanced RAG Flow:**

```
Question
  ↓
Embed Question
  ↓
Find 4 Similar Chunks (more candidates)
  ↓
Contextual Compression (filter irrelevant parts) 🔥
  ↓
Add Source Metadata
  ↓
Insert ONLY Relevant Info into Prompt
  ↓
Generate More Accurate Answer
```

---

## **Metadata Filtering Example**

```python
architecture_docs = vectorstore.similarity_search(
    "architecture patterns",
    k=2,
    filter={"topic": "architecture"}
)
```

**This is like SQL:**

```sql
SELECT TOP 2 *
FROM documents
WHERE topic = 'architecture'
  AND similarity_to('architecture patterns') > threshold
ORDER BY similarity DESC
```

**Result:** Only gets documents with `topic: "architecture"` metadata!

**Real-world example:**

```python
# Search only in production docs
prod_docs = vectorstore.similarity_search(
    query="deployment steps",
    filter={"environment": "production", "status": "approved"}
)

# Search only team-specific docs
team_docs = vectorstore.similarity_search(
    query="team processes",
    filter={"owner": "team_analytics"}
)

# Search by date range (if you add timestamp metadata)
recent_docs = vectorstore.similarity_search(
    query="latest features",
    filter={"year": 2024, "month": {"$gte": 11}}  # Nov 2024 or later
)
```

---

## **Key Comparisons:**

### **Basic RAG vs Advanced RAG:**

| Feature          | Basic RAG             | Advanced RAG                       |
| ---------------- | --------------------- | ---------------------------------- |
| Document Types   | Single (text)         | Multiple (text, CSV, JSON)         |
| Text Splitting   | Simple, fixed         | Recursive, intelligent             |
| Metadata         | None                  | Rich tagging (source, topic, etc.) |
| Retrieval        | Retrieves full chunks | Contextual compression             |
| Filtering        | No filtering          | Metadata-based filtering           |
| Prompt Context   | Generic               | Context-aware with source info     |
| Chunks Retrieved | 2                     | 4 (then compressed)                |
| Context Quality  | ⭐⭐⭐                | ⭐⭐⭐⭐⭐                         |
| Production Ready | 🟡 Basic              | ✅ Production-grade                |

### **Data Engineering Parallel:**

| Data Engineering Concept | RAG Equivalent                |
| ------------------------ | ----------------------------- |
| Multiple data sources    | Multiple document loaders     |
| Data catalog metadata    | Document metadata             |
| ETL pipeline             | Document processing pipeline  |
| Query optimization       | Contextual compression        |
| Partitioning/indexing    | Metadata filtering            |
| Data quality checks      | Chunk validation              |
| Data lineage tracking    | Source metadata in results    |
| Column-level security    | Metadata-based access control |

---

## **Production Considerations:**

### **1. Document Management**

```python
# Add rich metadata for production
doc.metadata.update({
    "ingestion_date": "2024-12-11",
    "version": "1.0",
    "author": "team_data",
    "last_modified": "2024-12-10",
    "classification": "internal",
    "department": "engineering",
    "tags": ["architecture", "design-patterns"]
})
```

### **2. Incremental Updates**

```python
# Only process new/modified documents
def update_vectorstore(vectorstore, docs_directory):
    # Track last ingestion time
    last_sync = load_last_sync_time()

    # Find new/modified documents
    new_docs = find_documents_modified_since(docs_directory, last_sync)

    if new_docs:
        print(f"Processing {len(new_docs)} new/modified documents")
        splits = text_splitter.split_documents(new_docs)
        vectorstore.add_documents(splits)

    # Update sync time
    save_last_sync_time(datetime.now())
```

### **3. Chunking Strategy by Content Type**

```python
def get_text_splitter(doc_type):
    """Different chunking strategies for different content"""
    if doc_type == "code":
        # Larger chunks for code to preserve context
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " "]
        )
    elif doc_type == "documentation":
        # Medium chunks for documentation
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". "]
        )
    elif doc_type == "chat_logs":
        # Smaller chunks for chat messages
        return RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n", ": ", " "]
        )
    else:
        # Default
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
```

### **4. Monitoring & Observability**

```python
def log_retrieval_metrics(question, retrieved_docs, compressed_docs):
    """Track retrieval quality"""
    metrics = {
        "question": question,
        "num_retrieved": len(retrieved_docs),
        "num_after_compression": len(compressed_docs),
        "compression_ratio": len(compressed_docs) / len(retrieved_docs),
        "avg_similarity": calculate_avg_similarity(retrieved_docs),
        "source_distribution": {
            source: count
            for source, count in count_sources(retrieved_docs).items()
        },
        "topics_covered": list(set(doc.metadata.get("topic") for doc in compressed_docs))
    }

    # Log to monitoring system
    logger.info("Retrieval metrics", extra=metrics)

    # Send to metrics service (e.g., DataDog, CloudWatch)
    metrics_client.increment("rag.retrieval.count")
    metrics_client.gauge("rag.compression_ratio", metrics["compression_ratio"])

    return metrics
```

### **5. Caching for Performance**

```python
from functools import lru_cache
import hashlib

def hash_query(query: str) -> str:
    """Create cache key from query"""
    return hashlib.md5(query.encode()).hexdigest()

# Cache embeddings
@lru_cache(maxsize=1000)
def get_cached_embedding(text: str):
    return embeddings.embed_query(text)

# Cache retrieval results
from cachetools import TTLCache
retrieval_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour TTL

def cached_retrieve(query: str):
    cache_key = hash_query(query)

    if cache_key in retrieval_cache:
        print(f"Cache hit for: {query}")
        return retrieval_cache[cache_key]

    # Perform retrieval
    results = compression_retriever.get_relevant_documents(query)
    retrieval_cache[cache_key] = results

    return results
```

### **6. Error Handling**

```python
def robust_rag_query(question: str, max_retries=3):
    """RAG query with error handling and retries"""
    for attempt in range(max_retries):
        try:
            # Try retrieval
            response = rag_chain.invoke(question)
            return response

        except Exception as e:
            logger.error(f"RAG query failed (attempt {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                # Exponential backoff
                time.sleep(2 ** attempt)
            else:
                # Final fallback
                return "I'm sorry, I'm having trouble retrieving information right now. Please try again later."
```

---

## **Real-World Use Cases:**

### **1. Internal Documentation Search**

```python
# Setup
for doc in company_docs:
    doc.metadata.update({
        "department": "engineering",
        "access_level": "internal",
        "last_updated": "2024-12-11",
        "author": "john.doe@company.com",
        "version": "2.1"
    })

# Search with permissions
def search_docs(query, user):
    docs = vectorstore.similarity_search(
        query=query,
        filter={
            "department": user.department,
            "access_level": {"$in": user.allowed_access_levels}
        }
    )
    return docs

# Usage
user = User(department="engineering", allowed_access_levels=["public", "internal"])
results = search_docs("deployment process", user)
```

### **2. Data Pipeline Documentation**

```python
# Index pipeline definitions
pipeline_docs = []
for pipeline in get_all_pipelines():
    doc = Document(
        page_content=pipeline.get_documentation(),
        metadata={
            "pipeline_name": pipeline.name,
            "owner": pipeline.owner,
            "status": pipeline.status,
            "schedule": pipeline.schedule,
            "data_sources": pipeline.sources,
            "last_run": pipeline.last_run_date.isoformat()
        }
    )
    pipeline_docs.append(doc)

vectorstore.add_documents(pipeline_docs)

# Natural language queries
answer = rag_chain.invoke("Show me all production pipelines that run daily")
answer = rag_chain.invoke("Which pipelines use PostgreSQL as a source?")
answer = rag_chain.invoke("What pipelines are owned by team_analytics?")
```

### **3. Code Search & Documentation**

```python
# Index codebase
for file_path in find_python_files("src/"):
    with open(file_path, "r") as f:
        code = f.read()

    doc = Document(
        page_content=code,
        metadata={
            "language": "python",
            "file_path": file_path,
            "file_type": "source_code",
            "last_modified": get_file_mtime(file_path),
            "module": extract_module_name(file_path),
            "functions": extract_function_names(code),
            "classes": extract_class_names(code)
        }
    )
    code_docs.append(doc)

# Code-aware chunking
code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n"]
)

splits = code_splitter.split_documents(code_docs)
vectorstore.add_documents(splits)

# Ask coding questions
answer = rag_chain.invoke("How do we handle errors in the ETL pipeline?")
answer = rag_chain.invoke("Show me examples of retry logic")
answer = rag_chain.invoke("Where is the database connection configured?")
```

### **4. Support Ticket Analysis**

```python
# Index support tickets
for ticket in get_support_tickets():
    doc = Document(
        page_content=f"Issue: {ticket.title}\n\n{ticket.description}\n\nResolution: {ticket.resolution}",
        metadata={
            "ticket_id": ticket.id,
            "category": ticket.category,
            "priority": ticket.priority,
            "status": ticket.status,
            "created_date": ticket.created_at.isoformat(),
            "resolved_date": ticket.resolved_at.isoformat() if ticket.resolved_at else None,
            "tags": ticket.tags
        }
    )
    ticket_docs.append(doc)

vectorstore.add_documents(ticket_docs)

# Search for similar issues
answer = rag_chain.invoke("Have we seen connection timeout issues before?")
answer = rag_chain.invoke("How did we resolve data quality problems in the past?")

# Filter by category
similar_tickets = vectorstore.similarity_search(
    query="pipeline failure",
    filter={"category": "pipeline_issues", "status": "resolved"}
)
```

### **5. Research Paper Database**

```python
# Index research papers
for paper in research_papers:
    doc = Document(
        page_content=f"Title: {paper.title}\n\nAbstract: {paper.abstract}\n\nKey Findings: {paper.findings}",
        metadata={
            "title": paper.title,
            "authors": paper.authors,
            "publication_date": paper.date.isoformat(),
            "journal": paper.journal,
            "doi": paper.doi,
            "keywords": paper.keywords,
            "citation_count": paper.citations
        }
    )
    paper_docs.append(doc)

# Search by topic and date
recent_ml_papers = vectorstore.similarity_search(
    query="machine learning data quality",
    filter={
        "keywords": {"$in": ["machine-learning", "data-quality"]},
        "publication_date": {"$gte": "2023-01-01"}
    }
)
```

---

## **Advanced Features:**

### **1. Multi-Query Retrieval**

Generate multiple variations of the question for better coverage:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# This will automatically:
# 1. Generate 3-5 variations of your question
# 2. Retrieve docs for each variation
# 3. Deduplicate results
# 4. Return unique relevant docs

# Example: "What is ETL?" becomes:
# - "What is ETL?"
# - "Explain Extract Transform Load process"
# - "How does ETL work?"
# - "What are the steps in ETL?"
```

### **2. Ensemble Retrieval**

Combine multiple retrieval methods:

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# Keyword-based retrieval
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 4

# Semantic retrieval
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Combine both (70% semantic, 30% keyword)
ensemble_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# Now you get best of both worlds!
```

### **3. Parent Document Retrieval**

Retrieve small chunks for matching, but return larger parent chunks for context:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Store for parent documents
store = InMemoryStore()

# Small chunks for retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# Large chunks for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Retrieves: Small chunks (precise matching)
# Returns: Large chunks (better context)
```

### **4. Self-Query Retrieval**

Let the LLM construct metadata filters from natural language:

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Define metadata schema
metadata_field_info = [
    AttributeInfo(
        name="source_type",
        description="The type of document source",
        type="string"
    ),
    AttributeInfo(
        name="topic",
        description="The topic of the document",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="The year the document was created",
        type="integer"
    )
]

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Data engineering documentation",
    metadata_field_info=metadata_field_info
)

# User asks: "Show me architecture documents from 2024"
# LLM automatically creates: filter={"topic": "architecture", "year": 2024}
```

---

## **Performance Optimization:**

### **1. Batch Processing**

```python
# Process documents in batches
def ingest_documents_in_batches(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        splits = text_splitter.split_documents(batch)
        vectorstore.add_documents(splits)
        print(f"Processed batch {i//batch_size + 1}")
```

### **2. Async Operations**

```python
import asyncio

async def process_documents_async(file_paths):
    tasks = [load_and_process_document(path) for path in file_paths]
    results = await asyncio.gather(*tasks)
    return results

# Use for faster ingestion
documents = asyncio.run(process_documents_async(file_paths))
```

### **3. Embedding Cache**

```python
# Avoid re-embedding same text
embedding_cache = {}

def get_embedding_with_cache(text):
    text_hash = hashlib.md5(text.encode()).hexdigest()

    if text_hash not in embedding_cache:
        embedding_cache[text_hash] = embeddings.embed_query(text)

    return embedding_cache[text_hash]
```

---

## **Testing & Validation:**

### **1. Retrieval Quality Tests**

```python
def test_retrieval_quality():
    """Test if retriever finds relevant documents"""
    test_cases = [
        {
            "query": "What is ETL?",
            "expected_topics": ["concepts"],
            "expected_keywords": ["extract", "transform", "load"]
        },
        {
            "query": "Which tools for orchestration?",
            "expected_topics": ["tools"],
            "expected_keywords": ["airflow"]
        }
    ]

    for test in test_cases:
        docs = compression_retriever.get_relevant_documents(test["query"])

        # Check topics
        topics = [doc.metadata.get("topic") for doc in docs]
        assert any(t in topics for t in test["expected_topics"]), \
            f"Expected topics {test['expected_topics']} not found in {topics}"

        # Check keywords
        content = " ".join(doc.page_content.lower() for doc in docs)
        for keyword in test["expected_keywords"]:
            assert keyword in content, \
                f"Expected keyword '{keyword}' not found in retrieved content"

    print("✅ All retrieval quality tests passed!")
```

### **2. Answer Quality Tests**

```python
def test_answer_quality():
    """Test if RAG provides accurate answers"""
    test_cases = [
        {
            "question": "What's the difference between ETL and ELT?",
            "required_phrases": ["extract", "transform", "load", "order"]
        }
    ]

    for test in test_cases:
        answer = rag_chain.invoke(test["question"])

        for phrase in test["required_phrases"]:
            assert phrase.lower() in answer.lower(), \
                f"Required phrase '{phrase}' not in answer"

    print("✅ All answer quality tests passed!")
```

---

## **The Power of This System:**

**You can now:**

1. ✅ Search across multiple document types (text, CSV, JSON)
2. ✅ Filter by metadata (like SQL WHERE clauses)
3. ✅ Get compressed, relevant answers (no noise)
4. ✅ Track data lineage (source metadata)
5. ✅ Build a self-service knowledge base
6. ✅ Handle production-scale document ingestion
7. ✅ Monitor and optimize retrieval quality
8. ✅ Implement advanced retrieval patterns

**This is production-ready!** It combines:

- Data engineering principles (metadata, partitioning, lineage)
- AI capabilities (embeddings, retrieval, generation)
- Best practices (compression, filtering, formatting, monitoring)

---

## **Next Steps:**

### **1. Add More Document Types**

```python
# Add PDF support
from langchain_community.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader("manual.pdf")

# Add Word documents
from langchain_community.document_loaders import Docx2txtLoader
docx_loader = Docx2txtLoader("specs.docx")

# Add web pages
from langchain_community.document_loaders import WebBaseLoader
web_loader = WebBaseLoader("https://docs.example.com")
```

### **2. Build a Web Interface**

```python
# Use Streamlit for a chat interface
import streamlit as st

st.title("📚 Knowledge Base Search")

query = st.text_input("Ask a question:")
if query:
    with st.spinner("Searching..."):
        answer = rag_chain.invoke(query)
    st.write(answer)

    # Show sources
    docs = compression_retriever.get_relevant_documents(query)
    with st.expander("View Sources"):
        for doc in docs:
            st.write(f"**Source:** {doc.metadata['source_type']}")
            st.write(doc.page_content)
            st.divider()
```

### **3. Add Authentication & Access Control**

```python
# Implement row-level security
def get_user_filter(user):
    return {
        "access_level": {"$in": user.access_levels},
        "department": {"$in": [user.department, "public"]}
    }

# Apply during search
docs = vectorstore.similarity_search(
    query=query,
    filter=get_user_filter(current_user)
)
```

### **4. Schedule Automated Updates**

```python
# Use Apache Airflow or similar
from airflow import DAG
from airflow.operators.python import PythonOperator

def sync_documents():
    new_docs = fetch_new_documents()
    ingest_documents_in_batches(new_docs)

dag = DAG('rag_document_sync', schedule_interval='@daily')
sync_task = PythonOperator(
    task_id='sync_documents',
    python_callable=sync_documents,
    dag=dag
)
```

### **5. Monitor in Production**

```python
# Add observability
from opentelemetry import trace
from opentelemetry.instrumentation.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument()

# Now you get automatic tracing of:
# - Retrieval latency
# - LLM calls
# - Token usage
# - Error rates
```

You now have the foundation to build enterprise-grade RAG systems that handle multiple document types, scale to production, and provide accurate, contextualized answers!

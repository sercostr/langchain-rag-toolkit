# LangChain Quick Reference for Data Engineers

## 🎯 Core Concepts

### 1. LLM (Large Language Model)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
response = llm.invoke("Your question here")
```

**Data Engineer Translation**: LLM is like a function that takes text input and returns text output. Think of it as a very smart API.

### 2. Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Explain {concept} for a {role}"
)
```

**Data Engineer Translation**: Like SQL parameterized queries. Prevents injection, enables reuse.

### 3. Chains (LCEL - LangChain Expression Language)

```python
chain = prompt | llm | output_parser
result = chain.invoke({"concept": "embeddings", "role": "data engineer"})
```

**Data Engineer Translation**: Like Unix pipes or Spark transformations. Data flows through each step.

### 4. Vector Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("Some text")  # Returns list of ~1536 floats
```

**Data Engineer Translation**:

- Text → Numbers (like hashing, but preserves meaning)
- Similar text → Similar numbers
- Enables "semantic search" instead of keyword search

### 5. Vector Store

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)
```

**Data Engineer Translation**:

- Like a database, but stores vectors
- Query: "Find documents similar to X"
- Uses: ANN (Approximate Nearest Neighbor) search
- Similar to: Elasticsearch for text, but for meaning

### 6. RAG (Retrieval Augmented Generation)

```python
# 1. Retrieve relevant docs
retriever = vectorstore.as_retriever(k=3)

# 2. Generate answer using retrieved context
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

**Data Engineer Translation**:

```
Traditional:  Query → Database → Result
RAG:          Question → Vector DB → Relevant Docs → LLM → Answer
```

## 🔧 Common Patterns

### Pattern 1: Simple Q&A

```python
llm = ChatOpenAI(model="gpt-4o-mini")
answer = llm.invoke("What is ETL?")
```

### Pattern 2: Structured Output

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
```

### Pattern 3: RAG Pipeline

```python
# Load → Split → Embed → Store → Retrieve → Generate
loader = TextLoader("file.txt")
docs = loader.load()
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
```

### Pattern 4: Agent with Tools

```python
from langchain.agents import create_sql_agent

agent = create_sql_agent(llm, db, verbose=True)
agent.invoke("How many users signed up last week?")
```

## 📊 Comparison Table

| Concept    | Data Engineering Equivalent | Example                             |
| ---------- | --------------------------- | ----------------------------------- |
| LLM        | API/Service                 | GPT-4, Claude                       |
| Prompt     | SQL Query                   | "Find all users..."                 |
| Chain      | Data Pipeline               | Extract → Transform → Load          |
| Embeddings | Hash/Index                  | Text → Vector                       |
| Vector DB  | Database                    | Chroma, Pinecone, pgvector          |
| RAG        | Query + Context             | Search docs, generate answer        |
| Agent      | Orchestrator                | Airflow DAG that decides next steps |
| Tool       | External Service            | Database, API, Calculator           |

## 💡 Mental Models

### 1. LLM as a Function

```python
# Traditional function
def analyze_data(data: DataFrame) -> Report:
    return processed_report

# LLM function
def llm(prompt: str) -> str:
    return generated_text
```

### 2. Embeddings as Coordinates

```
"Data Engineering" → [0.2, 0.8, 0.3, ...]  # Point in 1536-dim space
"Data Science"     → [0.3, 0.7, 0.4, ...]  # Nearby point
"Cooking"          → [0.9, 0.1, 0.2, ...]  # Far away point
```

### 3. RAG as JOIN

```sql
-- Traditional
SELECT answer
FROM knowledge_base
WHERE topic = 'ETL'

-- RAG (conceptually)
SELECT llm_generate(question, context)
FROM (
    SELECT * FROM vector_store
    WHERE similarity(embedding, question_embedding) > threshold
    LIMIT 3
) as context
```

## 🚀 Quick Start Commands

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API key
```

### Run Examples

```bash
python 01_hello_langchain.py
python 02_simple_rag.py
python 03_data_engineer_sql_agent.py
python 04_document_rag_advanced.py
```

### Interactive Testing

```python
# Start Python REPL
python

# Test LLM
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
print(llm.invoke("Hello!").content)
```

## ⚡ Performance Tips (Data Engineer Mindset)

### 1. Batch Processing

```python
# Bad: One at a time
for doc in docs:
    embedding = embeddings.embed_query(doc)

# Good: Batch
embeddings.embed_documents(docs)  # Processes in batches
```

### 2. Caching

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())  # Cache repeated queries
```

### 3. Cost Optimization

```python
# Use cheaper models for simple tasks
cheap_llm = ChatOpenAI(model="gpt-4o-mini")  # ~$0.10 per 1M tokens
expensive_llm = ChatOpenAI(model="gpt-4")    # ~$30 per 1M tokens

# Monitor token usage
response = llm.invoke("Question")
print(response.usage_metadata)  # {'input_tokens': 10, 'output_tokens': 50}
```

### 4. Parallel Processing

```python
# Use async for multiple queries
import asyncio

async def process_batch(questions):
    tasks = [llm.ainvoke(q) for q in questions]
    return await asyncio.gather(*tasks)
```

## 🐛 Debugging

### 1. Enable Verbose Mode

```python
agent = create_sql_agent(llm, db, verbose=True)  # Shows reasoning
```

### 2. Use LangSmith Tracing

```python
# In .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key

# Automatically traces all chains
# View at: https://smith.langchain.com/
```

### 3. Print Intermediate Steps

```python
chain = prompt | llm | StrOutputParser()

# Debug each step
print(prompt.format(concept="RAG"))  # See formatted prompt
result = llm.invoke(prompt.format(concept="RAG"))
print(result)  # See raw LLM response
```

## 📐 Architecture Patterns

### Pattern 1: Simple Chatbot

```
User Input → LLM → Response
```

### Pattern 2: RAG System

```
User Question → Embed → Vector Search → Retrieve Docs → LLM (with context) → Answer
```

### Pattern 3: Agent System

```
User Request → Agent (LLM) → Decides Tool → Executes → Observes → Decides Next → ...
```

### Pattern 4: Multi-Agent

```
User → Coordinator Agent → Task Distribution → [Agent 1, Agent 2, Agent 3] → Aggregation → Response
```

## 💰 Cost Estimation

### Models (as of Dec 2024)

- **GPT-4o-mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output
- **GPT-4**: ~$30 per 1M input tokens, ~$60 per 1M output
- **Embeddings**: ~$0.02 per 1M tokens

### Typical Costs

- **Learning (this repo)**: $1-5 total
- **Small RAG app**: $10-50/month
- **Production RAG**: $100-1000/month (depends on usage)

### Cost Control

```python
# 1. Use cheaper models
llm = ChatOpenAI(model="gpt-4o-mini")

# 2. Limit output tokens
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=200)

# 3. Cache results
# 4. Optimize chunk size in RAG
# 5. Monitor with LangSmith
```

## 🔗 Essential Links

- [LangChain Docs](https://python.langchain.com/)
- [OpenAI Pricing](https://openai.com/pricing)
- [LangSmith](https://smith.langchain.com/)
- [Vector DB Guide](https://www.pinecone.io/learn/)

## 🎯 Next Steps

1. ✅ Master these concepts by running the 4 Python files
2. ✅ Build something at work (start with SQL agent!)
3. ✅ Join LangChain Discord for help
4. ✅ Read RECOMMENDATIONS.md for complete learning path

---

**Pro Tip**: As a data engineer, you already understand pipelines, optimization, and production systems. AI engineering is 80% the same - just different components! 🚀

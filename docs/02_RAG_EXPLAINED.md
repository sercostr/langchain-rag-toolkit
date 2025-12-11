# RAG System Explained

## **RAG Overview**

RAG = **Retrieval Augmented Generation**

- **Retrieval**: Find relevant information from documents
- **Generation**: Use LLM to generate answer based on retrieved info

## **Step-by-Step Breakdown:**

### **1. Create Sample Documents**

```python
def create_sample_documents():
    docs = [
        "LangChain is a framework...",
        "RAG combines retrieval...",
        # ... more documents
    ]
```

- Creates a text file with knowledge about AI/LangChain
- This is your **knowledge base**

### **2. Load Documents**

```python
loader = TextLoader("sample_docs.txt")
documents = loader.load()
```

- Reads the text file into LangChain's document format

### **3. Split Documents into Chunks**

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splits = text_splitter.split_documents(documents)
```

**Why split?**

- LLMs have context limits
- Smaller chunks = more precise retrieval
- `chunk_overlap=20` means chunks share 20 characters (prevents losing context at boundaries)

**Result:** 5 chunks from your documents

### **4. Create Embeddings & Vector Store**

```python
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="ai_learning"
)
```

**What happens:**

1. **Embeddings**: Each chunk is converted to a vector (list of numbers)
   - Similar meaning = similar vectors
   - Example: "dog" and "puppy" have similar vectors
2. **Vector Store (ChromaDB)**: Stores these vectors for fast similarity search
   - Like a database but optimized for finding similar items

### **5. Create Retriever**

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
```

- `k=2` means "retrieve the 2 most relevant chunks" for each question

### **6. Build the RAG Chain**

```python
template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer: """

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**This is the magic! Let me break it down:**

```python
{"context": retriever, "question": RunnablePassthrough()}
```

- **Input**: User's question
- **Output**: Dictionary with:
  - `context`: Top 2 relevant chunks (from retriever)
  - `question`: The original question (passed through)

Then the pipeline continues:

```
→ prompt (formats the template with context + question)
→ llm (generates answer)
→ StrOutputParser() (extracts text)
```

### **7. How a Query Works**

When you ask: **"What is RAG?"**

**Step-by-step flow:**

1. **Question Embedding**

   - "What is RAG?" → converted to vector

2. **Similarity Search** (retriever)

   - Compares question vector with all document vectors
   - Finds 2 most similar chunks:
     ```
     "RAG combines retrieval and generation..."
     "LangChain is a framework..."
     ```

3. **Prompt Construction**

   ```
   Answer the question based only on the following context:

   RAG combines retrieval and generation...
   LangChain is a framework...

   Question: What is RAG?

   Answer:
   ```

4. **LLM Generation**
   - Receives the prompt with context
   - Generates answer based on the retrieved documents
   - Output: "RAG (Retrieval Augmented Generation) combines retrieval and generation..."

## **Visual Flow:**

```
User Question: "What is RAG?"
       ↓
Convert to embedding (vector)
       ↓
Search vector database
       ↓
Find 2 most similar chunks
       ↓
Insert chunks into prompt template
       ↓
Send to LLM
       ↓
Generate answer based on context
       ↓
Return answer to user
```

## **Key Benefits of RAG:**

1. **Accurate answers** - Based on your specific documents, not just LLM training
2. **Transparency** - You know where the answer came from (the retrieved chunks)
3. **Up-to-date** - Add new documents without retraining the LLM
4. **Domain-specific** - Works with your company/personal knowledge base

## **Comparison:**

| Without RAG                    | With RAG                         |
| ------------------------------ | -------------------------------- |
| LLM uses only training data    | LLM uses your specific documents |
| May hallucinate or be outdated | Grounded in your knowledge base  |
| Generic answers                | Context-specific answers         |

## **Real-World Applications:**

- **Chatbots** - Answer questions about your product/documentation
- **Customer Support** - Query support tickets and documentation
- **Research Assistants** - Search through papers and documents
- **Code Assistants** - Find relevant code examples from your codebase
- **Legal/Medical** - Query domain-specific knowledge bases

This is the foundation of most AI applications today!

## **Next Steps:**

- Try adding your own documents to `sample_docs.txt`
- Experiment with different `chunk_size` and `k` values
- Try different vector databases (Pinecone, pgvector)
- Add memory to make it conversational
- Implement filters and metadata search

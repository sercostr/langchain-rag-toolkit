"""
Project 4: Advanced RAG - Document Processing for Data Engineers

Goal: Build a production-quality RAG system with:
- Multiple document types (PDF, CSV, JSON)
- Advanced chunking strategies
- Metadata filtering
- Re-ranking
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
import json
import csv
import os

load_dotenv()


def create_diverse_documents():
    """Create sample documents in different formats"""
    
    # 1. Text document - Data Engineering concepts
    with open("data_eng_concepts.txt", "w") as f:
        f.write("""
Data Engineering Concepts:

ETL vs ELT: ETL (Extract, Transform, Load) processes data before loading. 
ELT (Extract, Load, Transform) loads raw data first, then transforms using the target system's compute.
Modern cloud warehouses prefer ELT because of their powerful processing capabilities.

Data Lakehouse: Combines data lake flexibility with data warehouse performance.
Examples include Databricks Delta Lake, Apache Iceberg, and AWS Lake Formation.
They provide ACID transactions, schema enforcement, and time travel on data lakes.

Streaming vs Batch: Batch processing handles data in scheduled intervals (hourly, daily).
Streaming processing handles data in real-time as it arrives.
Tools: Kafka, Spark Streaming, Flink for streaming; Airflow, dbt for batch.

Data Quality: Ensuring accuracy, completeness, consistency, timeliness.
Use tools like Great Expectations, dbt tests, and Soda for data quality checks.
""")
    
    # 2. CSV document - Tool comparison
    with open("de_tools.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tool", "category", "use_case", "popularity"])
        writer.writeheader()
        writer.writerows([
            {"tool": "Apache Airflow", "category": "Orchestration", "use_case": "Workflow scheduling", "popularity": "High"},
            {"tool": "dbt", "category": "Transformation", "use_case": "SQL transformations", "popularity": "High"},
            {"tool": "Apache Kafka", "category": "Streaming", "use_case": "Real-time data pipelines", "popularity": "High"},
            {"tool": "Snowflake", "category": "Warehouse", "use_case": "Cloud data warehousing", "popularity": "High"},
            {"tool": "Databricks", "category": "Platform", "use_case": "Unified analytics", "popularity": "High"},
        ])
    
    # 3. JSON document - Architecture patterns
    with open("architectures.json", "w") as f:
        json.dump({
            "patterns": [
                {
                    "name": "Lambda Architecture",
                    "description": "Combines batch and streaming layers for comprehensive data processing",
                    "components": ["Batch Layer", "Speed Layer", "Serving Layer"],
                    "pros": "Handles both historical and real-time data",
                    "cons": "Complex to maintain two separate pipelines"
                },
                {
                    "name": "Kappa Architecture",
                    "description": "Simplified architecture using only streaming",
                    "components": ["Stream Processing", "Serving Layer"],
                    "pros": "Simpler than Lambda, single pipeline",
                    "cons": "Requires reprocessing for historical changes"
                },
                {
                    "name": "Medallion Architecture",
                    "description": "Bronze (raw) -> Silver (cleaned) -> Gold (aggregated)",
                    "components": ["Bronze Layer", "Silver Layer", "Gold Layer"],
                    "pros": "Clear data quality progression, easy to debug",
                    "cons": "Can be over-engineered for simple use cases"
                }
            ]
        }, f, indent=2)
    
    print("✅ Created diverse sample documents")


def build_advanced_rag():
    """Build an advanced RAG system with multiple document types"""
    
    # Load different document types
    text_loader = TextLoader("data_eng_concepts.txt")
    csv_loader = CSVLoader("de_tools.csv")
    
    # Custom loader for JSON
    json_loader = JSONLoader(
        file_path="architectures.json",
        jq_schema=".patterns[]",
        text_content=False
    )
    
    # Load all documents
    text_docs = text_loader.load()
    csv_docs = csv_loader.load()
    json_docs = json_loader.load()
    
    # Add metadata
    for doc in text_docs:
        doc.metadata["source_type"] = "text"
        doc.metadata["topic"] = "concepts"
    
    for doc in csv_docs:
        doc.metadata["source_type"] = "csv"
        doc.metadata["topic"] = "tools"
    
    for doc in json_docs:
        doc.metadata["source_type"] = "json"
        doc.metadata["topic"] = "architecture"
    
    all_docs = text_docs + csv_docs + json_docs
    print(f"✅ Loaded {len(all_docs)} documents from various sources")
    
    # Advanced text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )
    
    splits = text_splitter.split_documents(all_docs)
    print(f"✅ Split into {len(splits)} chunks")
    
    # Create vector store with metadata
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="data_eng_knowledge"
    )
    
    # Create retriever with metadata filtering
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Add contextual compression for better results
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    print("✅ Created advanced retriever with compression")
    
    # Create RAG chain
    template = """You are an AI assistant helping data engineers learn new concepts.
Use the following context to answer the question. If you're not sure, say so.

Context:
{context}

Question: {question}

Answer (be specific and reference the context):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(
            f"Source: {doc.metadata.get('source_type', 'unknown')} | "
            f"Topic: {doc.metadata.get('topic', 'unknown')}\n{doc.page_content}"
            for doc in docs
        )
    
    rag_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, vectorstore


def test_advanced_rag(chain):
    """Test with data engineering questions"""
    
    questions = [
        "What's the difference between ETL and ELT?",
        "Which tools should I use for orchestration?",
        "Explain the Medallion Architecture",
        "What are the pros and cons of Lambda Architecture?",
        "How do I ensure data quality in my pipelines?",
    ]
    
    for question in questions:
        print(f"\n{'='*70}")
        print(f"❓ Question: {question}")
        print('='*70)
        response = chain.invoke(question)
        print(f"💬 Answer:\n{response}")


def demonstrate_metadata_filtering(vectorstore):
    """Show how to filter by metadata"""
    
    print(f"\n{'='*70}")
    print("🔍 Metadata Filtering Examples")
    print('='*70)
    
    # Filter by source type
    architecture_docs = vectorstore.similarity_search(
        "architecture patterns",
        k=2,
        filter={"topic": "architecture"}
    )
    
    print("\n📐 Architecture-related documents:")
    for doc in architecture_docs:
        print(f"- {doc.metadata}: {doc.page_content[:100]}...")


if __name__ == "__main__":
    print("="*70)
    print("Advanced RAG for Data Engineers")
    print("="*70)
    
    # Create documents
    create_diverse_documents()
    
    # Build RAG system
    print("\n🤖 Building advanced RAG system...")
    rag_chain, vectorstore = build_advanced_rag()
    
    # Test it
    print("\n🧪 Testing RAG system...")
    test_advanced_rag(rag_chain)
    
    # Demonstrate metadata filtering
    demonstrate_metadata_filtering(vectorstore)
    
    print("\n✅ Done! You now have a production-quality RAG system!")
    print("💡 Key learnings:")
    print("   - Multiple document formats (text, CSV, JSON)")
    print("   - Metadata tagging and filtering")
    print("   - Contextual compression for better retrieval")
    print("   - This is similar to building data catalogs!")

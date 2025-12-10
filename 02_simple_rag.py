"""
Project 2: Simple RAG System

Goal: Build your first RAG (Retrieval Augmented Generation) system
- Document loading
- Text splitting
- Embeddings
- Vector store
- Retrieval & generation
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def create_sample_documents():
    """Create sample documents for testing"""
    docs = [
        "LangChain is a framework for developing applications powered by language models. It enables applications to be context-aware and reason.",
        "RAG (Retrieval Augmented Generation) combines retrieval and generation. It retrieves relevant documents and uses them to generate better responses.",
        "Vector embeddings are numerical representations of text. Similar texts have similar embeddings, enabling semantic search.",
        "A vector database stores embeddings and enables fast similarity search. Popular options include Pinecone, Chroma, and pgvector.",
        "Data pipelines in AI systems handle data ingestion, transformation, and loading - similar to traditional ETL but optimized for ML workflows."
    ]
    
    with open("sample_docs.txt", "w") as f:
        for doc in docs:
            f.write(doc + "\n\n")
    
    print("✅ Created sample_docs.txt")


def build_simple_rag():
    """Build a simple RAG system"""
    
    # 1. Load documents
    loader = TextLoader("sample_docs.txt")
    documents = loader.load()
    
    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splits = text_splitter.split_documents(documents)
    print(f"✅ Split documents into {len(splits)} chunks")
    
    # 3. Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="ai_learning"
    )
    print("✅ Created vector store")
    
    # 4. Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 5. Create RAG chain
    template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer: """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Build the chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def test_rag_system(chain):
    """Test the RAG system with questions"""
    questions = [
        "What is RAG?",
        "How do vector embeddings work?",
        "What are vector databases?",
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        response = chain.invoke(question)
        print(f"💬 Answer: {response}")
        print("-" * 50)


if __name__ == "__main__":
    print("="*50)
    print("Building Simple RAG System")
    print("="*50)
    
    # Step 1: Create sample documents
    create_sample_documents()
    
    # Step 2: Build RAG system
    # rag_chain = build_simple_rag()
    
    # Step 3: Test it
    # test_rag_system(rag_chain)
    
    print("\n✅ Uncomment the code above to run the RAG system!")
    print("💡 Make sure you have OPENAI_API_KEY in your .env file")

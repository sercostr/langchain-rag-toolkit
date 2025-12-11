"""
Project 1: Hello LangChain - Your First LLM Application

Goal: Get familiar with basic LangChain concepts
- LLM invocation
- Prompt templates
- Simple chains
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def basic_llm_call():
    """Basic LLM invocation"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.invoke("What are the key differences between data engineering and AI engineering?")
    print(response.content)


def prompt_template_example():
    """Using prompt templates"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI tutor teaching data engineers about AI engineering."),
        ("user", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "question": "Explain what embeddings are in simple terms with an example from data engineering."
    })
    
    print(response)


def simple_chain_example():
    """Chain multiple operations"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Create a chain that explains a concept and then provides code example
    explain_prompt = ChatPromptTemplate.from_template(
        "Explain {concept} in 2-3 sentences for a data engineer."
    )
    
    code_prompt = ChatPromptTemplate.from_template(
        "Given this explanation: {explanation}\n\n"
        "Provide a simple Python code example."
    )
    
    # Chain 1: Get explanation
    explain_chain = explain_prompt | llm | StrOutputParser()
    
    # Chain 2: Get code example
    code_chain = code_prompt | llm | StrOutputParser()
    
    # Execute
    explanation = explain_chain.invoke({"concept": "vector embeddings"})
    print("Explanation:")
    print(explanation)
    print("\n" + "="*50 + "\n")
    
    code = code_chain.invoke({"explanation": explanation})
    print("Code Example:")
    print(code)


if __name__ == "__main__":
    print("="*50)
    print("1. Basic LLM Call")
    print("="*50)
    basic_llm_call()
    
    print("\n" + "="*50)
    print("2. Prompt Template Example")
    print("="*50)
    prompt_template_example()
    
    print("\n" + "="*50)
    print("3. Simple Chain Example")
    print("="*50)
    simple_chain_example()
    
    print("\n✅ All examples completed!")

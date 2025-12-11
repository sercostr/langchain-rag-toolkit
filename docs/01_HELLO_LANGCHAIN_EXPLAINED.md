# Hello LangChain Explained - Step by Step

## **What is This?**

Your first LangChain application! This introduces the core building blocks:

1. **LLM invocation** - Calling an AI model
2. **Prompt templates** - Structured prompts with variables
3. **Chains** - Combining operations in sequence

Think of it as "Hello World" for AI engineering!

---

## **Step-by-Step Breakdown:**

### **1. Basic LLM Call**

```python
def basic_llm_call():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.invoke("What are the key differences between data engineering and AI engineering?")
    print(response.content)
```

**Breaking it down:**

**`ChatOpenAI`**

- Interface to OpenAI's chat models
- Connects to GPT models (GPT-4, GPT-4o-mini, etc.)

**`model="gpt-4o-mini"`**

- Specifies which model to use
- "gpt-4o-mini" = faster, cheaper version of GPT-4
- Other options: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"

**`temperature=0.7`**

- Controls randomness/creativity
- Scale: 0.0 to 2.0
  - **0.0** = Deterministic, consistent (good for production)
  - **0.7** = Balanced creativity (good for general use)
  - **1.5+** = Very creative, unpredictable

**`llm.invoke()`**

- Sends request to the model
- **Synchronous** - waits for response
- Alternative: `llm.ainvoke()` for async

**`response.content`**

- Extracts the text from the response
- Response also contains: metadata, token usage, model info

**What you get:**

```
Question: "What are the key differences between data engineering and AI engineering?"
Response: "Data engineering focuses on building pipelines to move and transform data..."
```

---

### **2. Prompt Templates**

```python
def prompt_template_example():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI tutor teaching data engineers about AI engineering."),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "question": "Explain what embeddings are in simple terms with an example from data engineering."
    })
```

**Why use templates?**

**Without templates (hard-coded):**

```python
response = llm.invoke("Explain embeddings")
```

- No context
- No reusability
- Hard to maintain

**With templates (structured):**

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor..."),
    ("user", "{question}")
])
```

- ✅ Reusable with different questions
- ✅ Consistent system context
- ✅ Variables for dynamic content

---

#### **Understanding Message Types:**

```python
ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor..."),
    ("user", "{question}")
])
```

**`system` message:**

- Sets the AI's role/personality/context
- Like setting up a consultant's background
- Examples:
  - "You are a Python expert"
  - "You are a data engineer helping with SQL"
  - "You are a code reviewer"

**`user` message:**

- The actual question/request
- Can use variables with `{variable_name}`
- Example: `"Explain {concept} in simple terms"`

**Other message types:**

- `assistant` - Previous AI responses (for conversation history)
- `function` - Function call results (advanced)

**Real conversation flow:**

```
System: "You are a helpful AI tutor teaching data engineers about AI engineering."
User: "Explain what embeddings are in simple terms with an example from data engineering."
Assistant: "Embeddings are numerical representations of data..."
```

---

#### **The Pipe Operator (`|`)**

```python
chain = prompt | llm | StrOutputParser()
```

This is **chain composition** - like Unix pipes!

**What it means:**

```
Input
  ↓
prompt (format the template)
  ↓
llm (send to model)
  ↓
StrOutputParser (extract text)
  ↓
Output
```

**Step-by-step execution:**

**Step 1: Input**

```python
{"question": "Explain what embeddings are..."}
```

**Step 2: Prompt formatting**

```python
prompt | ...
# Creates:
[
  SystemMessage(content="You are a helpful AI tutor..."),
  HumanMessage(content="Explain what embeddings are...")
]
```

**Step 3: LLM invocation**

```python
... | llm | ...
# Sends to OpenAI, receives:
AIMessage(content="Embeddings are numerical representations...")
```

**Step 4: Parse output**

```python
... | StrOutputParser()
# Extracts:
"Embeddings are numerical representations..."
```

**Data engineering parallel:**

```python
# Like a data pipeline:
raw_data | transform | filter | aggregate | output
```

---

#### **StrOutputParser**

```python
StrOutputParser()
```

**What it does:**

- Extracts plain text from LLM response
- Removes metadata, formatting, etc.

**Without parser:**

```python
AIMessage(
    content="Embeddings are...",
    additional_kwargs={},
    response_metadata={'token_usage': {...}, 'model_name': 'gpt-4o-mini'}
)
```

**With parser:**

```python
"Embeddings are..."
```

**Other parsers available:**

- `JsonOutputParser()` - Parse JSON responses
- `PydanticOutputParser()` - Parse into Python objects
- `CommaSeparatedListOutputParser()` - Parse lists
- `DatetimeOutputParser()` - Parse dates

---

### **3. Simple Chain Example (Sequential Chains)**

```python
def simple_chain_example():
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
    code = code_chain.invoke({"explanation": explanation})
```

**This demonstrates chaining LLM calls!**

---

#### **Flow Visualization:**

````
User Input: "vector embeddings"
        ↓
┌─────────────────────────────────────┐
│  Chain 1: Get Explanation           │
│                                     │
│  explain_prompt                     │
│  "Explain vector embeddings..."     │
│         ↓                           │
│  llm.invoke()                       │
│         ↓                           │
│  Output: "Vector embeddings are     │
│  numerical representations that     │
│  capture semantic meaning..."       │
└─────────────────────────────────────┘
        ↓
explanation = "Vector embeddings are numerical..."
        ↓
┌─────────────────────────────────────┐
│  Chain 2: Get Code Example          │
│                                     │
│  code_prompt                        │
│  "Given this explanation:           │
│  {explanation}                      │
│  Provide a Python code example."    │
│         ↓                           │
│  llm.invoke()                       │
│         ↓                           │
│  Output: "```python                 │
│  from sentence_transformers ...     │
│  ```"                               │
└─────────────────────────────────────┘
        ↓
Final Output: Code example
````

---

#### **Why Use Sequential Chains?**

**Without chaining (all in one prompt):**

```python
response = llm.invoke("Explain vector embeddings and provide code example")
```

- ❌ Less control over each step
- ❌ Can't reuse parts
- ❌ Harder to debug
- ❌ Can't modify intermediate results

**With chaining (step by step):**

```python
explanation = explain_chain.invoke({"concept": "vector embeddings"})
code = code_chain.invoke({"explanation": explanation})
```

- ✅ Clear separation of concerns
- ✅ Each step is reusable
- ✅ Easy to debug each step
- ✅ Can inspect/modify intermediate results
- ✅ Can add validation between steps

---

#### **Real-World Use Cases:**

**1. Documentation Generator**

```python
# Chain 1: Summarize code
summary_chain = summary_prompt | llm | StrOutputParser()

# Chain 2: Generate API docs
docs_chain = docs_prompt | llm | StrOutputParser()

# Chain 3: Generate examples
example_chain = example_prompt | llm | StrOutputParser()

summary = summary_chain.invoke({"code": code})
docs = docs_chain.invoke({"summary": summary})
examples = example_chain.invoke({"docs": docs})
```

**2. Data Pipeline Generator**

```python
# Chain 1: Analyze requirements
analysis_chain = analyze_prompt | llm | StrOutputParser()

# Chain 2: Generate SQL
sql_chain = sql_prompt | llm | StrOutputParser()

# Chain 3: Generate tests
test_chain = test_prompt | llm | StrOutputParser()

analysis = analysis_chain.invoke({"requirements": requirements})
sql = sql_chain.invoke({"analysis": analysis})
tests = test_chain.invoke({"sql": sql})
```

**3. Code Review Assistant**

```python
# Chain 1: Find issues
issues_chain = issues_prompt | llm | JsonOutputParser()

# Chain 2: Suggest fixes
fixes_chain = fixes_prompt | llm | StrOutputParser()

# Chain 3: Generate improved code
improved_chain = improved_prompt | llm | StrOutputParser()

issues = issues_chain.invoke({"code": code})
fixes = fixes_chain.invoke({"issues": issues})
improved_code = improved_chain.invoke({"fixes": fixes, "original": code})
```

---

## **Key Concepts Summary:**

### **1. LLM (Large Language Model)**

- The AI model that generates text
- In this case: GPT-4o-mini from OpenAI
- Like a function: `input text → process → output text`

### **2. Temperature**

| Value   | Behavior      | Use Case                        |
| ------- | ------------- | ------------------------------- |
| 0.0     | Deterministic | Production code, factual Q&A    |
| 0.3-0.7 | Balanced      | General use, tutorials          |
| 1.0-1.5 | Creative      | Brainstorming, creative writing |
| 2.0     | Very random   | Experimental, artistic          |

### **3. Prompt Templates**

- Reusable prompt structures with variables
- Support system/user/assistant messages
- Variables: `{variable_name}`

### **4. Chains**

- Sequence of operations connected with `|`
- Data flows through each step
- Like data pipelines: `extract | transform | load`

### **5. Output Parsers**

- Extract specific data from LLM responses
- `StrOutputParser()` = plain text
- Other parsers for JSON, structured data, etc.

---

## **Comparison:**

### **Simple Function Call vs LangChain:**

**Traditional approach:**

```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a tutor"},
        {"role": "user", "content": "Explain embeddings"}
    ]
)

text = response['choices'][0]['message']['content']
```

**LangChain approach:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a tutor"),
    ("user", "Explain {concept}")
])

chain = prompt | llm | StrOutputParser()
text = chain.invoke({"concept": "embeddings"})
```

**Why LangChain?**

- ✅ **Reusable**: Same prompt with different inputs
- ✅ **Composable**: Chain multiple operations
- ✅ **Maintainable**: Clear structure
- ✅ **Testable**: Test each component separately
- ✅ **Scalable**: Easy to add more steps
- ✅ **Observable**: Built-in logging and tracing

---

## **Visual Examples:**

### **Example 1: Basic LLM Call**

```
┌─────────────────────────────────┐
│  Question:                      │
│  "What are the key differences  │
│  between data engineering and   │
│  AI engineering?"               │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│  ChatOpenAI                     │
│  (gpt-4o-mini, temp=0.7)        │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│  Answer:                        │
│  "Data engineering focuses on   │
│  building pipelines to move and │
│  transform data at scale..."    │
└─────────────────────────────────┘
```

### **Example 2: Prompt Template**

```
Input: {"question": "Explain embeddings"}
              ↓
┌─────────────────────────────────┐
│  ChatPromptTemplate             │
│                                 │
│  System: "You are a tutor..."   │
│  User: "Explain embeddings"     │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│  ChatOpenAI                     │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│  StrOutputParser                │
└─────────────────────────────────┘
              ↓
Output: "Embeddings are..."
```

### **Example 3: Sequential Chains**

````
Input: {"concept": "vector embeddings"}
              ↓
┌─────────────────────────────────┐
│  Chain 1: Explanation           │
│                                 │
│  Prompt → LLM → Parser          │
└─────────────────────────────────┘
              ↓
intermediate: "Vector embeddings are..."
              ↓
┌─────────────────────────────────┐
│  Chain 2: Code Example          │
│                                 │
│  Prompt → LLM → Parser          │
└─────────────────────────────────┘
              ↓
Output: "```python\n..."
````

---

## **Common Patterns:**

### **1. Single LLM Call**

```python
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Your question")
```

**Use when:** Simple, one-off questions

### **2. Template + LLM**

```python
prompt = ChatPromptTemplate.from_messages([...])
chain = prompt | llm | StrOutputParser()
response = chain.invoke({"variable": "value"})
```

**Use when:** Reusable prompts, consistent formatting

### **3. Sequential Chains**

```python
chain1 = prompt1 | llm | StrOutputParser()
chain2 = prompt2 | llm | StrOutputParser()

result1 = chain1.invoke(input1)
result2 = chain2.invoke({"context": result1})
```

**Use when:** Multi-step processing, dependent operations

### **4. Parallel Chains (Advanced)**

```python
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "keywords": keywords_chain,
    "sentiment": sentiment_chain
})

results = parallel_chain.invoke({"text": text})
# Returns: {"summary": "...", "keywords": [...], "sentiment": "positive"}
```

**Use when:** Independent operations, save time

---

## **Debugging Tips:**

### **1. Print Intermediate Results**

```python
# Instead of:
chain = prompt | llm | StrOutputParser()

# Do:
formatted_prompt = prompt.invoke({"question": "test"})
print("Formatted prompt:", formatted_prompt)

llm_response = llm.invoke(formatted_prompt)
print("LLM response:", llm_response)

parsed = StrOutputParser().invoke(llm_response)
print("Parsed:", parsed)
```

### **2. Use Verbose Mode**

```python
chain = prompt | llm.with_config(verbose=True) | StrOutputParser()
```

### **3. Check Token Usage**

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = chain.invoke({"question": "test"})
    print(f"Tokens used: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost}")
```

---

## **Best Practices:**

### **1. Always Set Temperature**

```python
# Bad: Uses default (usually 0.7)
llm = ChatOpenAI(model="gpt-4o-mini")

# Good: Explicit temperature
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # For consistency
```

### **2. Use System Messages**

```python
# Bad: No context
prompt = ChatPromptTemplate.from_template("{question}")

# Good: Clear context
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data engineering expert"),
    ("user", "{question}")
])
```

### **3. Load API Keys from Environment**

```python
# Bad: Hard-coded
llm = ChatOpenAI(api_key="sk-...")

# Good: From environment
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI()  # Automatically uses OPENAI_API_KEY
```

### **4. Handle Errors**

```python
try:
    response = chain.invoke({"question": question})
except Exception as e:
    print(f"Error: {e}")
    response = "Sorry, I couldn't process that request."
```

---

## **What You Learned:**

1. ✅ **Basic LLM calls** - How to invoke an AI model
2. ✅ **Prompt templates** - Structured, reusable prompts
3. ✅ **Message types** - System vs user messages
4. ✅ **Chains** - Composing operations with `|`
5. ✅ **Output parsing** - Extracting clean text
6. ✅ **Sequential processing** - Multi-step LLM workflows

---

## **Next Steps:**

### **From Here to Production:**

**01_hello_langchain.py** (You are here)

- Basic concepts: LLM, prompts, chains

↓

**02_simple_rag.py**

- Add document retrieval
- Combine retrieval + generation

↓

**03_data_engineer_sql_agent.py**

- Add tools and agents
- Dynamic decision making

↓

**04_document_rag_advanced.py**

- Production features
- Multiple formats, metadata, compression

↓

**Production System**

- Authentication
- Monitoring
- Scaling
- Error handling

---

## **Practice Exercises:**

### **Exercise 1: Modify Temperature**

Try different temperatures and see how responses change:

```python
for temp in [0.0, 0.5, 1.0, 1.5]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temp)
    response = llm.invoke("Tell me a creative story about data pipelines")
    print(f"\nTemperature {temp}:")
    print(response.content)
```

### **Exercise 2: Create Your Own Template**

```python
# Create a template for explaining technical concepts
your_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical writer who explains complex topics simply"),
    ("user", "Explain {topic} using an analogy from {domain}")
])

chain = your_prompt | llm | StrOutputParser()
response = chain.invoke({"topic": "embeddings", "domain": "cooking"})
```

### **Exercise 3: Build a Three-Step Chain**

```python
# Step 1: Analyze a data problem
# Step 2: Suggest a solution approach
# Step 3: Generate implementation code

analyze_prompt = ChatPromptTemplate.from_template(
    "Analyze this data problem: {problem}"
)
solution_prompt = ChatPromptTemplate.from_template(
    "Given this analysis: {analysis}\nSuggest a solution approach"
)
code_prompt = ChatPromptTemplate.from_template(
    "Given this solution: {solution}\nProvide Python implementation"
)

# Build the chains
analyze_chain = analyze_prompt | llm | StrOutputParser()
solution_chain = solution_prompt | llm | StrOutputParser()
code_chain = code_prompt | llm | StrOutputParser()

# Execute
problem = "I need to deduplicate customer records from multiple sources"
analysis = analyze_chain.invoke({"problem": problem})
solution = solution_chain.invoke({"analysis": analysis})
code = code_chain.invoke({"solution": solution})
```

---

## **The Foundation:**

This is **foundational** - everything in LangChain builds on these concepts:

- LLM invocation
- Prompts
- Chains
- Parsing

Master these, and you're ready for:

- RAG systems
- Agents
- Complex workflows
- Production applications

You've taken your first step into AI engineering! 🚀

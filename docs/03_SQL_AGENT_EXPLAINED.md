# SQL Agent Explained - Step by Step

## **What is a SQL Agent?**

An **AI agent** that can:

1. Understand natural language questions
2. Generate SQL queries automatically
3. Execute those queries on a database
4. Return human-readable answers

Perfect for data engineers - it bridges SQL expertise with AI!

---

## **Step-by-Step Breakdown:**

### **1. Create Sample Database**

```python
def create_sample_database():
    conn = sqlite3.connect("data_pipeline_metrics.db")
```

**Creates two tables:**

**Table 1: `pipelines`**

- Stores metadata about data pipelines
- Columns: `pipeline_id`, `pipeline_name`, `source_system`, `target_system`, `owner`

**Table 2: `pipeline_runs`**

- Stores execution history
- Columns: `run_id`, `pipeline_id`, `run_date`, `status`, `records_processed`, `duration_seconds`, `error_message`

**Sample data:**

- 3 pipelines (customer_etl, product_sync, sales_aggregation)
- 6 pipeline runs with various statuses

This simulates a real data engineering monitoring system!

---

### **2. Connect to Database**

```python
db = SQLDatabase.from_uri("sqlite:///data_pipeline_metrics.db")
```

**What happens:**

- `SQLDatabase` is LangChain's database wrapper
- Connects to the SQLite database
- Provides methods to:
  - Get table names
  - Get table schemas
  - Execute SQL queries
  - Get sample rows

This gives the agent "eyes" into the database structure.

---

### **3. Initialize LLM**

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

- Uses GPT-4o-mini (faster and cheaper than GPT-4)
- `temperature=0` → deterministic, consistent SQL generation
- This LLM will generate SQL queries and interpret results

---

### **4. Create SQL Agent**

```python
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
```

**Breaking down parameters:**

**`AgentType.ZERO_SHOT_REACT_DESCRIPTION`**

- **Zero-shot**: No examples needed, works from scratch
- **ReAct**: Reasoning + Acting pattern
  - Agent **Reasons** about what to do
  - Agent **Acts** by using tools
  - Agent **Observes** results
  - Repeats until answer is found

**`verbose=True`**

- Shows the agent's thinking process
- You saw this in the output: "Action: sql_db_list_tables", "Thought: I should check schema..."

**`handle_parsing_errors=True`**

- If agent makes a mistake, it can retry
- Makes it more robust

---

### **5. Agent Tools (Automatic)**

The agent automatically gets these tools:

| Tool                   | What it does                                     |
| ---------------------- | ------------------------------------------------ |
| `sql_db_list_tables`   | Lists all tables in database                     |
| `sql_db_schema`        | Gets schema (columns, types) for specific tables |
| `sql_db_query_checker` | Validates SQL syntax before executing            |
| `sql_db_query`         | Executes SQL queries                             |

The agent decides which tools to use and when!

---

## **How a Query Works - Example**

**Question:** "How many pipelines are in the database?"

### **Agent's Thought Process:**

**Step 1: List Tables**

```
Action: sql_db_list_tables
Output: pipeline_runs, pipelines
```

_"I need to know what tables exist"_

**Step 2: Check Schema**

```
Action: sql_db_schema
Input: "pipelines"
Output:
CREATE TABLE pipelines (
    pipeline_id INTEGER PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    ...
)
```

_"Let me see the structure of the pipelines table"_

**Step 3: Validate Query**

```
Action: sql_db_query_checker
Input: "SELECT COUNT(*) AS pipeline_count FROM pipelines;"
Output: ✓ Valid SQL
```

_"Is my SQL syntax correct?"_

**Step 4: Execute Query**

```
Action: sql_db_query
Input: "SELECT COUNT(*) AS pipeline_count FROM pipelines;"
Output: [(3,)]
```

_"Run the query and get results"_

**Step 5: Generate Answer**

```
Final Answer: There are 3 pipelines in the database.
```

---

## **Visual Flow:**

```
User Question
    ↓
Agent receives question
    ↓
┌─────────────────────────────────────┐
│  Agent uses ReAct Pattern:         │
│                                     │
│  1. THINK: What do I need to know? │
│     → List tables                   │
│                                     │
│  2. ACT: Use tool                   │
│     → sql_db_list_tables            │
│                                     │
│  3. OBSERVE: Got table names        │
│                                     │
│  4. THINK: Need table structure     │
│     → Check schema                  │
│                                     │
│  5. ACT: Use tool                   │
│     → sql_db_schema                 │
│                                     │
│  6. OBSERVE: Got column info        │
│                                     │
│  7. THINK: Can write SQL now        │
│     → SELECT COUNT(*) FROM...       │
│                                     │
│  8. ACT: Validate query             │
│     → sql_db_query_checker          │
│                                     │
│  9. ACT: Execute query              │
│     → sql_db_query                  │
│                                     │
│ 10. OBSERVE: Got result [(3,)]      │
│                                     │
│ 11. THINK: Format answer            │
│     → "There are 3 pipelines..."    │
└─────────────────────────────────────┘
    ↓
Human-readable answer
```

---

## **More Complex Example**

**Question:** "What's the average number of records processed by each pipeline?"

**Agent's SQL:**

```sql
SELECT
    p.pipeline_id,
    p.pipeline_name,
    AVG(pr.records_processed) AS average_records_processed
FROM pipelines p
JOIN pipeline_runs pr ON p.pipeline_id = pr.pipeline_id
GROUP BY p.pipeline_id
ORDER BY average_records_processed DESC
LIMIT 10;
```

**Result:**

- customer_etl: 151,000 records
- sales_aggregation: 46,000 records
- product_sync: 44,500 records

The agent:

1. ✅ Joined two tables correctly
2. ✅ Used aggregation (AVG)
3. ✅ Grouped by pipeline
4. ✅ Ordered results
5. ✅ Formatted output nicely

**All from natural language!**

---

## **Key Concepts:**

### **1. Agent vs Chain**

- **Chain**: Fixed sequence (A → B → C)
- **Agent**: Decides dynamically what to do next
  - Can skip steps
  - Can retry
  - Can use different tools based on question

### **2. ReAct Pattern**

```
Thought → Action → Observation → Thought → Action → ...
```

- Mimics human problem-solving
- Can correct mistakes
- Shows reasoning

### **3. Tools**

- Functions the agent can call
- Agent learns what each tool does
- Decides which to use and in what order

---

## **Why This Matters for Data Engineers:**

### **Traditional Approach:**

```
User: "Which pipelines failed?"
You: Write SQL query
You: Run query
You: Format results
You: Send to user
```

### **With SQL Agent:**

```
User: "Which pipelines failed?"
Agent: [Does everything automatically]
User: Gets answer in seconds
```

---

## **Real-World Use Cases:**

1. **Pipeline Monitoring Dashboard**

   - "Show me failures in the last 24 hours"
   - "Which pipelines are running slower than usual?"

2. **Ad-hoc Analysis**

   - "What's the failure rate by team?"
   - "Which source systems have the most issues?"

3. **Self-Service Analytics**

   - Non-technical users can query data
   - No SQL knowledge needed

4. **Automated Alerts**

   - Agent checks metrics periodically
   - Sends alerts when anomalies detected

5. **Data Quality Monitoring**
   - "Show me pipelines with 0 records processed"
   - "Which runs took longer than 5 minutes?"

---

## **Important Notes:**

### **Security Considerations:**

- Agent has full database access
- Use **read-only** database connections in production
- Implement query limits to prevent expensive operations
- Example:
  ```python
  # Read-only connection
  db = SQLDatabase.from_uri(
      "sqlite:///data.db",
      sample_rows_in_table_info=3,
      max_string_length=100
  )
  ```

### **Limitations:**

- Can make mistakes with complex queries
- May not understand domain-specific terminology without guidance
- Should validate critical queries before execution
- Performance depends on LLM quality

### **Improvements:**

```python
# Add few-shot examples for better SQL generation
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

# Create toolkit with custom configuration
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Add examples in the prompt
examples = """
Example 1:
Question: Show failed runs
SQL: SELECT * FROM pipeline_runs WHERE status = 'failed'

Example 2:
Question: Average duration by pipeline
SQL: SELECT pipeline_id, AVG(duration_seconds)
     FROM pipeline_runs GROUP BY pipeline_id
"""
```

---

## **Comparison:**

| Traditional SQL                    | SQL Agent                  |
| ---------------------------------- | -------------------------- |
| Manual query writing               | Natural language questions |
| Requires SQL expertise             | Anyone can query           |
| Time-consuming for complex queries | Fast, automatic            |
| Static dashboards                  | Dynamic exploration        |
| Schema knowledge required          | Agent discovers schema     |
| Fixed queries                      | Flexible, adaptive queries |

---

## **Advanced Features:**

### **1. Custom Tools**

Add domain-specific tools:

```python
from langchain.tools import Tool

def check_pipeline_health(pipeline_name: str) -> str:
    """Check if a pipeline is healthy based on recent runs"""
    # Your custom logic
    return f"Pipeline {pipeline_name} is healthy"

custom_tool = Tool(
    name="check_pipeline_health",
    func=check_pipeline_health,
    description="Check pipeline health status"
)

# Add to agent
agent = create_sql_agent(
    llm=llm,
    db=db,
    extra_tools=[custom_tool],
    verbose=True
)
```

### **2. Memory**

Make the agent remember previous questions:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

# Now it can handle follow-up questions:
# User: "Show me all pipelines"
# Agent: [Shows pipelines]
# User: "Which ones failed?" (remembers context)
```

### **3. Prompt Customization**

Guide the agent's behavior:

```python
from langchain.agents import create_sql_agent

prefix = """You are a data pipeline monitoring assistant.
When analyzing failures, always check:
1. Error messages
2. Duration compared to successful runs
3. Number of records processed

Be concise but informative."""

agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_kwargs={"prefix": prefix},
    verbose=True
)
```

---

## **Production Considerations:**

### **1. Query Limits**

```python
# Prevent expensive queries
db = SQLDatabase.from_uri(
    "sqlite:///data.db",
    sample_rows_in_table_info=3,  # Limit sample data
    max_string_length=100,  # Truncate long strings
)
```

### **2. Read-Only Access**

```python
# For PostgreSQL
db = SQLDatabase.from_uri(
    "postgresql://readonly_user:pass@host:5432/db"
)
```

### **3. Timeout Protection**

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Query took too long")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    result = agent.invoke({"input": question})
finally:
    signal.alarm(0)  # Cancel timeout
```

### **4. Cost Monitoring**

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = agent.invoke({"input": question})
    print(f"Tokens used: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost}")
```

---

## **The Power:**

You asked: "How many pipelines are in the database?"

The agent:

1. ✅ Explored the database schema
2. ✅ Generated correct SQL
3. ✅ Validated the query
4. ✅ Executed it
5. ✅ Formatted the answer

**All automatically, in seconds!**

This is AI enhancing your data engineering workflows - you can focus on pipeline logic while AI handles the query interface.

---

## **Next Steps:**

1. **Connect to Real Databases**

   - Replace SQLite with PostgreSQL/MySQL/Snowflake
   - Use connection strings from your environment

2. **Add Custom Tools**

   - Pipeline restart tool
   - Alert notification tool
   - Slack integration

3. **Build a Dashboard**

   - Use Streamlit to create a UI
   - Let users ask questions through a chat interface

4. **Integrate with Monitoring**

   - Schedule periodic checks
   - Send alerts when anomalies detected
   - Create automated reports

5. **Advanced Queries**
   - Time-series analysis
   - Trend detection
   - Performance optimization suggestions

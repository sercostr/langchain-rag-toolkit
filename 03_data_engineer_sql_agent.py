"""
Project 3: SQL Agent - Leverage Your Data Engineering Skills!

Goal: Build an agent that can query databases using natural language
This bridges your existing SQL expertise with AI capabilities.
"""

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import sqlite3

load_dotenv()


def create_sample_database():
    """Create a sample database with data pipeline metrics"""
    conn = sqlite3.connect("data_pipeline_metrics.db")
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipelines (
            pipeline_id INTEGER PRIMARY KEY,
            pipeline_name TEXT NOT NULL,
            source_system TEXT,
            target_system TEXT,
            owner TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id INTEGER PRIMARY KEY,
            pipeline_id INTEGER,
            run_date DATE,
            status TEXT,
            records_processed INTEGER,
            duration_seconds INTEGER,
            error_message TEXT,
            FOREIGN KEY (pipeline_id) REFERENCES pipelines(pipeline_id)
        )
    """)
    
    # Insert sample data
    pipelines_data = [
        (1, "customer_etl", "PostgreSQL", "Snowflake", "team_analytics"),
        (2, "product_sync", "MongoDB", "BigQuery", "team_data"),
        (3, "sales_aggregation", "MySQL", "Redshift", "team_bi"),
    ]
    
    runs_data = [
        (1, 1, "2024-12-09", "success", 150000, 320, None),
        (2, 1, "2024-12-10", "success", 152000, 310, None),
        (3, 2, "2024-12-09", "failed", 0, 45, "Connection timeout"),
        (4, 2, "2024-12-10", "success", 89000, 180, None),
        (5, 3, "2024-12-09", "success", 45000, 120, None),
        (6, 3, "2024-12-10", "success", 47000, 125, None),
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO pipelines VALUES (?, ?, ?, ?, ?)", pipelines_data)
    cursor.executemany("INSERT OR REPLACE INTO pipeline_runs VALUES (?, ?, ?, ?, ?, ?, ?)", runs_data)
    
    conn.commit()
    conn.close()
    
    print("✅ Created sample database: data_pipeline_metrics.db")


def create_sql_agent_simple():
    """Create a simple SQL agent"""
    
    # Connect to database
    db = SQLDatabase.from_uri("sqlite:///data_pipeline_metrics.db")
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create SQL agent
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


def test_sql_agent(agent):
    """Test SQL agent with natural language queries"""
    
    questions = [
        "How many pipelines are in the database?",
        "Which pipeline had failures yesterday?",
        "What's the average number of records processed by each pipeline?",
        "Show me the pipeline with the longest average duration",
        "Which team owns the most pipelines?",
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print('='*60)
        
        try:
            response = agent.invoke({"input": question})
            print(f"\n💬 Answer: {response['output']}")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("="*60)
    print("SQL Agent - Natural Language to SQL")
    print("="*60)
    
    # Create sample database
    create_sample_database()
    
    # Create and test agent
    print("\n🤖 Creating SQL Agent...")
    agent = create_sql_agent_simple()
    
    print("\n🧪 Testing SQL Agent...")
    test_sql_agent(agent)
    
    print("\n✅ Done! This demonstrates how AI can interact with your data pipelines.")
    print("💡 As a data engineer, you can extend this to monitor real pipelines!")

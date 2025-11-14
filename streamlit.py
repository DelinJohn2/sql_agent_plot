import logging
import streamlit as st
from typing import Literal
from langchain_community.utilities import SQLDatabase
from config import build_llm,sql_connection
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langgraph.prebuilt import ToolNode
from langchain.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph

# Set up the page configuration and title for the Streamlit app
st.set_page_config(page_title="Interactive SQL Agent", layout="wide")
st.title("Interactive SQL Agent with LangChain")

# Custom logging filter to ensure 'step' exists in the log record
class StepFilter(logging.Filter):
    def filter(self, record):
        if 'step' not in record.__dict__:
            record.step = 'N/A'  # Assign a default value if 'step' is missing
        return True

# Configure logging to include timestamps and step numbers for clarity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - STEP %(step)d - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app_log.txt'), logging.StreamHandler()]  # Logs to both file and console
)

# Add the custom filter to the logger
logger = logging.getLogger(__name__)
logger.addFilter(StepFilter())

# Step 0: Initialize the SQL engine connection with error handling
try:
    engine = sql_connection()
    logger.info("Database connection successful.", extra={'step': 0})
except Exception as e:
    logger.error(f"Database connection failed: {e}", extra={'step': 0})
    raise e

# Step 1: Initialize the language model (LLM) using your build_llm function with error handling
try:
    llm = build_llm()
    logger.info("LLM initialization successful.", extra={'step': 1})
except Exception as e:
    logger.error(f"LLM initialization failed: {e}", extra={'step': 1})
    raise e

# Define the SQLAgent class as you already have it
class SQLAgent:
    def __init__(self, engine, llm):
        self.db = SQLDatabase(engine=engine)
        logger.info("SQLDatabase object created.", extra={'step': 2})
        self.llm = llm
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()
        logger.info(f"Toolkit initialized with {len(self.tools)} tools.", extra={'step': 2})

        # Extract key tools by name for easy reference
        self.get_schema_tool = next(tool for tool in self.tools if tool.name == "sql_db_schema")
        self.run_query_tool = next(tool for tool in self.tools if tool.name == "sql_db_query")
        self.list_tables_tool = next(tool for tool in self.tools if tool.name == "sql_db_list_tables")
        logger.info("Key tools mapped.", extra={'step': 2})

        # Wrap important tools into ToolNodes for graph-based usage
        self.get_schema_node = ToolNode([self.get_schema_tool], name="get_schema")
        self.run_query_node = ToolNode([self.run_query_tool], name="run_query")
        logger.info("ToolNodes created for schema and query tools.", extra={'step': 2})

    def list_available_tools(self):
        logger.info("Listing available tools:", extra={'step': 3})
        for tool in self.tools:
            logger.info(f"{tool.name}: {tool.description}", extra={'step': 3})

    def list_tables(self, state: MessagesState):
        tool_call = {"name": "sql_db_list_tables", "args": {}, "id": "abc123", "type": "tool_call"}
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        tool_message = self.list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")
        return {"messages": [tool_call_message, tool_message, response]}

    def generate_query(self, state: MessagesState):
        prompt = f"""You are an agent designed to interact with a SQL database. Given an input question, create a syntactically correct {self.db.dialect} query to run..."""
        system_message = {"role": "system", "content": prompt}
        llm_with_tools = self.llm.bind_tools([self.run_query_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

# Initialize the SQL agent
agent_obj = SQLAgent(engine=engine, llm=llm)

# Streamlit interactive session for user input
def run_agent_interactive():
    st.header("Ask a Question")
    user_input = st.text_input("Enter your question for the SQL Agent:")

    if user_input:
        messages = [{"role": "user", "content": user_input}]
        
        # Build state graph and execute sequential steps
        builder = StateGraph(MessagesState)
        builder.add_node(agent_obj.list_tables)
        builder.add_node(agent_obj.generate_query)
        builder.add_edge(START, "list_tables")
        builder.add_edge("list_tables", "generate_query")
        
        # Compile the agent and start the process
        agent = builder.compile()
        
        # Run the agent sequentially
        state = {"messages": messages}
        
        # Execute list_tables step
        for step in agent.stream(state):
            if step and "messages" in step and step["messages"]:
                agent_response = step['messages'][-1].content
                st.write(f"Agent response: {agent_response}")
                
                # Log the response to the console and file
                logger.info(f"Agent response: {agent_response}", extra={'step': 4})

# Run the interactive session in the Streamlit app
if __name__ == "__main__":
    run_agent_interactive()

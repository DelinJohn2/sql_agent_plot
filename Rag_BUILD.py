import logging
from typing import Literal
from langchain_community.utilities import SQLDatabase
# from my_Sql_connection import sql_connection
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from Openi_ai import build_llm
from langgraph.prebuilt import ToolNode
from langchain.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from KPL_SQL_CON import sql_connection as kpl_sql_connection

# Custom logging filter to ensure 'step' exists in the log record
class StepFilter(logging.Filter):
    def filter(self, record):
        if 'step' not in record.__dict__:
            record.step = 'N/A'  # Assign a default value if 'step' is missing
        return True

# Configure logging to include timestamps and step numbers for clarity and save logs to a file
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
    engine = kpl_sql_connection()
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

# Step 2: Define the SQLAgent class that will handle toolkit creation and tool methods
class SQLAgent:
    def __init__(self, engine, llm):
        # Step 2.1: Wrap SQLAlchemy engine with LangChain SQLDatabase utility
        self.db = SQLDatabase(engine=engine)
        logger.info("SQLDatabase object created.", extra={'step': 2})

        # Step 2.2: Store the initialized LLM and create the SQL toolkit using both
        self.llm = llm
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()
        logger.info(f"Toolkit initialized with {len(self.tools)} tools.", extra={'step': 2})

        # Step 2.3: Extract key tools by name for easy reference
        self.get_schema_tool = next(tool for tool in self.tools if tool.name == "sql_db_schema")
        self.run_query_tool = next(tool for tool in self.tools if tool.name == "sql_db_query")
        self.list_tables_tool = next(tool for tool in self.tools if tool.name == "sql_db_list_tables")
        logger.info("Key tools mapped.", extra={'step': 2})

        # Step 2.4: Wrap important tools into ToolNodes for graph-based usage
        self.get_schema_node = ToolNode([self.get_schema_tool], name="get_schema")
        self.run_query_node = ToolNode([self.run_query_tool], name="run_query")
        logger.info("ToolNodes created for schema and query tools.", extra={'step': 2})

    # Step 3: Method to log all available tools and their descriptions
    def list_available_tools(self):
        logger.info("Listing available tools:", extra={'step': 3})
        for tool in self.tools:
            logger.info(f"{tool.name}: {tool.description}", extra={'step': 3})

    # Step 4: Method invoking tool to list database tables
    def list_tables(self, state: MessagesState):
        logger.info("Executing list_tables tool.", extra={'step': 4})
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "abc123",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        tool_message = self.list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")
        return {"messages": [tool_call_message, tool_message, response]}

    # Step 5: Method calling schema retrieval tool via bound LLM
    def call_get_schema(self, state: MessagesState):
        logger.info("Executing call_get_schema tool.", extra={'step': 5})
        llm_with_tools = self.llm.bind_tools([self.get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # Step 6: Method for generating SQL queries based on input questions
    def generate_query(self, state: MessagesState):
        logger.info("Executing generate_query.", extra={'step': 6})
        prompt = f"""
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {self.db.dialect} query to run,
        then look at the results of the query and return the answer. Unless specified,
        always limit your query to at most 5 results.

        Order the results by a relevant column to return the most interesting examples.
        Only request relevant columns—never all columns.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        """
        system_message = {"role": "system", "content": prompt}
        llm_with_tools = self.llm.bind_tools([self.run_query_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    # Step 7: Method to check and validate SQL queries for common mistakes
    def check_query(self, state: MessagesState):
        logger.info("Executing check_query.", extra={'step': 7})
        prompt = f"""
        You are a SQL expert with strong attention to detail.
        Double-check the {self.db.dialect} query for common mistakes, such as:
        - Using NOT IN with NULL values
        - UNION vs UNION ALL confusion
        - BETWEEN for ranges
        - Data type mismatches
        - Proper quoting of identifiers
        - Correct function arguments
        - Proper casting
        - Correct columns in joins

        If errors exist, rewrite the query; otherwise, reproduce it.
        You will execute the query after this validation.
        """
        system_message = {"role": "system", "content": prompt}
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = self.llm.bind_tools([self.run_query_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message, user_message])
        response.id = state["messages"][-1].id
        return {"messages": [response]}

# Step 8: Instantiate the agent class
agent_obj = SQLAgent(engine=engine, llm=llm)

# Step 9: Define the function controlling flow between nodes
def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    last_message = state["messages"][-1]
    # Continue if tool call exists in last message; else end
    if not last_message.tool_calls:
        logger.info("No tool calls in last message, ending flow.", extra={'step': 9})
        return END
    else:
        logger.info("Tool calls detected, continuing to check_query.", extra={'step': 9})
        return "check_query"

# Step 10: Build the state graph that defines the conversation flow
builder = StateGraph(MessagesState)

builder.add_node(agent_obj.list_tables)
builder.add_node(agent_obj.call_get_schema)
builder.add_node(agent_obj.get_schema_node, "get_schema")
builder.add_node(agent_obj.generate_query)
builder.add_node(agent_obj.check_query)
builder.add_node(agent_obj.run_query_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

# Compile the agent from the graph
agent = builder.compile()
# logger.info("Agent compiled successfully.", extra={'step': 10})

# Step 11: Define the interactive session to keep agent alive for multiple queries
def interactive_agent(agent):
    logger.info("Starting interactive session.")
    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("exit", "quit"):
            logger.info("Exiting. Goodbye!")
            break

        # Prepare initial user message to agent
        messages = [{"role": "user", "content": user_input}]

        # Stream agent's response stepwise, printing each message prettily
        for step in agent.stream({"messages": messages}, stream_mode="values"):
            if step and "messages" in step and step["messages"]:
                # logger.info(f"Agent response: {step['messages'][-1].content}")
                print(f"\nAgent: {step['messages'][-1].content}")
                

if __name__ == "__main__":
    # Step 12: Log available tools and start interactive session
    # agent_obj.list_available_tools()
    interactive_agent(agent)

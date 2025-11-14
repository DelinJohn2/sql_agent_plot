import logging
from typing import Literal
from langchain_community.utilities import SQLDatabase
from my_Sql_connection import sql_connection
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from Openi_ai import build_llm
from langgraph.prebuilt import ToolNode
from langchain.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph






engine = sql_connection()  # Establish database connection


# Step 1: Initialize the language model (LLM) using your build_llm function with error handling

llm = build_llm()  # Initialize language model

# Re-raise exception to stop execution if LLM initialization fails

# Step 2: Define the SQLAgent class that will handle toolkit creation and tool methods
class SQLAgent:
	def __init__(self, engine, llm):
		# Step 2.1: Wrap SQLAlchemy engine with LangChain SQLDatabase utility
		self.db = SQLDatabase(engine=engine)

		# Step 2.2: Store the initialized LLM and create the SQL toolkit using both
		self.llm = llm
		self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
		self.tools = self.toolkit.get_tools()

		# Step 2.3: Extract key tools by name for easy reference
		self.get_schema_tool = next(tool for tool in self.tools if tool.name == "sql_db_schema")
		self.run_query_tool = next(tool for tool in self.tools if tool.name == "sql_db_query")
		self.list_tables_tool = next(tool for tool in self.tools if tool.name == "sql_db_list_tables")

		# Step 2.4: Wrap important tools into ToolNodes for graph-based usage
		self.get_schema_node = ToolNode([self.get_schema_tool], name="get_schema")
		self.run_query_node = ToolNode([self.run_query_tool], name="run_query")

	# Step 3: Method to log all available tools and their descriptions


	# Step 4: Method invoking tool to list database tables
	def list_tables(self, state: MessagesState):
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

		llm_with_tools = self.llm.bind_tools([self.get_schema_tool], tool_choice="any")
		response = llm_with_tools.invoke(state["messages"])
		return {"messages": [response]}

	# Step 6: Method for generating SQL queries based on input questions
	def generate_query(self, state: MessagesState):
		# ---- Step 1: SQL Generation Prompt ----
		prompt = f"""
		You are an agent designed to interact with a SQL database.
		Given an input question, create a syntactically correct {self.db.dialect} query to run,
		then look at the results of the query and return the answer. Unless specified,
		always limit your query to at most 5 results.

		The result should be a ananlysis of the data

		DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
		"""
		system_message = {"role": "system", "content": prompt}

		# ---- Step 2: Generate and Run Query ----
		llm_with_tools = self.llm.bind_tools([self.run_query_tool])
		response = llm_with_tools.invoke([system_message] + state["messages"])

		# ---- Step 3: Extract the query result ----
		# The query result is usually embedded in response.content or a ToolMessage
		query_output = getattr(response, "content", None)

		# ---- Step 4: Generate Plotly JSON from Query Result ----
		if query_output:
			plotly_prompt = f"""
						You are a data visualization expert. 
						Your job is to convert the SQL result below into **valid Plotly JSON**.

						STRICT RULES (MUST FOLLOW):
						1. All line charts MUST use: 
						"type": "scatter"
						and 
						"mode": "lines" or "lines+markers"
						(NEVER use "type": "line")
						2. Allowed trace types ONLY:
						"scatter", "bar", "pie", "histogram"
						(NO other types)
						3. Every figure MUST follow this exact structure:
						{{
							"name": "<title>",
							"figure": {{
								"data": [...],
								"layout": {{
									"title": "<title>",
									"xaxis": {{}},
									"yaxis": {{}}
								}}
							}}
						}}

						4. Output must be a SINGLE valid JSON object with this format:
						{{
							"plots": [
								{{ ... }},
								{{ ... }},
								{{ ... }}
							]
						}}

						5. JSON must be valid and must work with:
						json.loads(...)
						go.Figure(...)

						6. DO NOT add backticks, markdown, comments, or text before/after the JSON.

						7. If you create a line chart, ALWAYS use:
						"type": "scatter"

						SQL Result:
						{query_output}

						Now generate 1–3 Plotly charts (different perspectives).
						Important:Make sure that the plots are very much in coherence with the data no hallucinations or assumptions should be given
						Return ONLY the JSON.
						"""



			plotly_response = self.llm.invoke([{"role": "system", "content": plotly_prompt}])

		else:
			plotly_response = None

		# ---- Step 5: Return both messages ----
		return {"messages": [response, plotly_response] if plotly_response else [response]}


	# Step 7: Method to check and validate SQL queries for common mistakes
	def check_query(self, state: MessagesState):

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
        return END
    else:
        return "check_query"

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

agent = builder.compile()
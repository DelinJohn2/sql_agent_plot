import logging
from typing import Literal
from langchain_community.utilities import SQLDatabase


from langchain_community.agent_toolkits import SQLDatabaseToolkit
from config import build_llm,sql_connection
from langgraph.prebuilt import ToolNode
from langchain.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
import json


from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

def load_faiss_index(path="faiss_index"):
    if not os.path.exists(path):
        return None

    embeddings = OpenAIEmbeddings()

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )



engine = sql_connection()  # Establish database connection


# Step 1: Initialize the language model (LLM) using your build_llm function with error handling

llm = build_llm()  # Initialize language model


def  vector_store_call(question:str):
    classifier_prompt = f"""
            You are a SQL planning assistant.

            Given this user question:
            {question}

            Split it into two parts:

            1. schema_needs:
            - which tables, fields, or structural elements might be needed

            2. logic_needs:
            - what business logic, formulas, or domain rules might matter later

            Return JSON ONLY:
            {{
            "schema_needs": [...],
            "logic_needs": [...]
            }}"""

    raw_plan = llm.invoke(classifier_prompt).content
    plan = json.loads(raw_plan)

    schema_query = " ".join(plan.get("schema_needs", []))
    logic_query = " ".join(plan.get("logic_needs", []))

    return schema_query, logic_query




class SQLAgent:
    def __init__(self, engine, llm,schema_query,
                 logic_query):
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

        #
        self.schema_query=schema_query
        self.logic_query=logic_query



        




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
        v_db=load_faiss_index()
        context=v_db.similarity_search(state['messages'][0].content)
        print(context)
        # ---- Step 1: SQL Generation Prompt ----
        prompt = f"""
        You are an intelligent SQL agent with access to a SQL execution tool.

    Your job is to:
    1. Understand the user's question
    2. Use the vector memory context (FAISS) to identify:
    - Relevant tables
    - Field meanings
    - Past query logic
    3. Generate a **valid, executable {self.db.dialect} SELECT query**
    {self.schema_query}
    using the SQL tool bound to you.

    =====================================================
    📌 USER QUESTION
    =====================================================
    {state['messages'][0].content}

    =====================================================
    📌 CONTEXT FROM VECTOR MEMORY (FAISS)
    (Helps you understand table purposes, relationships,
    and prior SQL outputs. Use it if relevant.)
    =====================================================
    {context}


        """
        system_message = {"role": "system", "content": prompt}

        # ---- Step 2: Generate and Run Query ----
        llm_with_tools = self.llm.bind_tools([self.run_query_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages":  [response]}

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
        - make sure there is a limit of 10

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
    def final_output(self, state: MessagesState):

        # Extract SQL tool result
        last_message = state["messages"][-1]
        sql_result = last_message.content

        if not sql_result:
            return {"messages":AIMessage(content="there is no sufficent data")}

        # ------------------------------------
        # LLM CALL #1: Data Analysis + Charts
        # ------------------------------------
        analysis_prompt = f"""
You are a senior data analyst.

Here is the logic context (domain logic):
{self.logic_query}

Here is the SQL result:
{sql_result}

Using BOTH the logic context and the SQL output,
provide a clear, insightful analysis with meaningful
interpretations, trends, and business insights.
"""

        analysis_output = self.llm.invoke(analysis_prompt).content


        plot_prompt = f"""
You are a data visualization expert.
Here is the logic context (domain logic):
{self.logic_query}

Given the following SQL query result, create a Plotly-compatible JSON
(as a Python dict) to visualize the data effectively. Make sure you use only the last sql query result that is where the weightage is

VERY IMPORTANT RULES (follow strictly):
1. Use only valid Plotly schema.
2. Do NOT use `marker.color` with non-numeric values when using a colorscale.
3. If categories (strings) are present, do NOT use colorscales.
4. If you want to color categories, you MUST generate valid HEX color codes for each category.
5. Do NOT invent invalid color formats (like "S001", "Store1", etc.).
6. Use simple Plotly figures: bar, line, scatter, pie — whichever fits the data best.
7. Every field in the JSON must be valid and must not include comments.

The JSON must follow this structure exactly:
{{
    "data": [...],
    "layout": {{}}
}}

SQL Result:
{sql_result}

Output ONLY valid JSON — no explanations, no markdown.
"""


        raw_plots = self.llm.invoke(plot_prompt).content
        return {
            "messages":[raw_plots]+[analysis_output]
        }




agent_obj = SQLAgent(engine=engine, llm=llm)



builder = StateGraph(MessagesState)

builder.add_node(agent_obj.list_tables)
builder.add_node(agent_obj.call_get_schema)
builder.add_node(agent_obj.get_schema_node, "get_schema")
builder.add_node(agent_obj.generate_query)
builder.add_node(agent_obj.check_query)
builder.add_node(agent_obj.run_query_node, "run_query")
builder.add_node(agent_obj.final_output, "final_output")

# Graph edges
builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_edge("generate_query", "check_query")
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "final_output")
builder.add_edge("final_output",END)
agent = builder.compile()



# import logging
# from typing import Literal
# from langchain_community.utilities import SQLDatabase

# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from config import build_llm,sql_connection
# from langgraph.prebuilt import ToolNode
# from langchain.messages import AIMessage
# from langchain_core.runnables import RunnableConfig
# from langgraph.graph import END, START, MessagesState, StateGraph
# import json





# engine = sql_connection()  # Establish database connection


# # Step 1: Initialize the language model (LLM) using your build_llm function with error handling

# llm = build_llm()  # Initialize language model



# class SQLAgent:
#     def __init__(self, engine, llm):
#         # Step 2.1: Wrap SQLAlchemy engine with LangChain SQLDatabase utility
#         self.db = SQLDatabase(engine=engine)

#         # Step 2.2: Store the initialized LLM and create the SQL toolkit using both
#         self.llm = llm
#         self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
#         self.tools = self.toolkit.get_tools()

#         # Step 2.3: Extract key tools by name for easy reference
#         self.get_schema_tool = next(tool for tool in self.tools if tool.name == "sql_db_schema")
#         self.run_query_tool = next(tool for tool in self.tools if tool.name == "sql_db_query")
#         self.list_tables_tool = next(tool for tool in self.tools if tool.name == "sql_db_list_tables")

#         # Step 2.4: Wrap important tools into ToolNodes for graph-based usage
#         self.get_schema_node = ToolNode([self.get_schema_tool], name="get_schema")
#         self.run_query_node = ToolNode([self.run_query_tool], name="run_query")

#     # Step 3: Method to log all available tools and their descriptions


#     # Step 4: Method invoking tool to list database tables
#     def list_tables(self, state: MessagesState):
#         tool_call = {
#             "name": "sql_db_list_tables",
#             "args": {},
#             "id": "abc123",
#             "type": "tool_call",
#         }
#         tool_call_message = AIMessage(content="", tool_calls=[tool_call])
#         tool_message = self.list_tables_tool.invoke(tool_call)
#         response = AIMessage(f"Available tables: {tool_message.content}")

#         return {"messages": [tool_call_message, tool_message, response]}

#     # Step 5: Method calling schema retrieval tool via bound LLM
#     def call_get_schema(self, state: MessagesState):

#         llm_with_tools = self.llm.bind_tools([self.get_schema_tool], tool_choice="any")
#         response = llm_with_tools.invoke(state["messages"])
#         return {"messages": [response]}

#     # Step 6: Method for generating SQL queries based on input questions
#     def generate_query(self, state: MessagesState):
#         # ---- Step 1: SQL Generation Prompt ----
#         prompt = f"""
#         You are an agent designed to interact with a SQL database.
#         Given an input question, create a syntactically correct {self.db.dialect} query to run,
#         make sure the column names adhere to the names given in the shema to ensure .

#         DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
#         """
#         system_message = {"role": "system", "content": prompt}

#         # ---- Step 2: Generate and Run Query ----
#         llm_with_tools = self.llm.bind_tools([self.run_query_tool])
#         response = llm_with_tools.invoke([system_message] + state["messages"])
#         return {"messages":  [response]}

#     # Step 7: Method to check and validate SQL queries for common mistakes
#     def check_query(self, state: MessagesState):

#         prompt = f"""
#         You are a SQL expert with strong attention to detail.
#         Double-check the {self.db.dialect} query for common mistakes, such as:
#         - Using NOT IN with NULL values
#         - UNION vs UNION ALL confusion
#         - BETWEEN for ranges
#         - Data type mismatches
#         - Proper quoting of identifiers
#         - Correct function arguments
#         - Proper casting
#         - Correct columns in joins
#         - make sure there is a limit of 10

#         If errors exist, rewrite the query; otherwise, reproduce it.
#         You will execute the query after this validation.
#         """
#         system_message = {"role": "system", "content": prompt}
#         tool_call = state["messages"][-1].tool_calls[0]
#         user_message = {"role": "user", "content": tool_call["args"]["query"]}
#         llm_with_tools = self.llm.bind_tools([self.run_query_tool], tool_choice="any")
#         response = llm_with_tools.invoke([system_message, user_message])
#         response.id = state["messages"][-1].id
#         return {"messages": [response]}

#     # Step 8: Instantiate the agent class
#     def final_output(self, state: MessagesState):

#         # Extract SQL tool result
#         last_message = state["messages"][-1]
#         sql_result = last_message.content

#         if not sql_result:
#             return {"messages":AIMessage(content="there is no sufficent data")}

#         # ------------------------------------
#         # LLM CALL #1: Data Analysis + Charts
#         # ------------------------------------
#         analysis_prompt = f"""
#         You are a senior data analyst.

#         Given this SQL result:

#         {sql_result}

#         give an analysis of the output and provide insig
#         """

#         analysis_output = self.llm.invoke(analysis_prompt).content


#         plot_prompt = f"""
# You are a data visualization expert.

# Given the following SQL query result, create a Plotly-compatible JSON
# (as a Python dict) to visualize the data effectively.

# VERY IMPORTANT RULES (follow strictly):
# 1. Use only valid Plotly schema.
# 2. Do NOT use `marker.color` with non-numeric values when using a colorscale.
# 3. If categories (strings) are present, do NOT use colorscales.
# 4. If you want to color categories, you MUST generate valid HEX color codes for each category.
# 5. Do NOT invent invalid color formats (like "S001", "Store1", etc.).
# 6. Use simple Plotly figures: bar, line, scatter, pie — whichever fits the data best.
# 7. Every field in the JSON must be valid and must not include comments.

# The JSON must follow this structure exactly:
# {{
#     "data": [...],
#     "layout": {{}}
# }}

# SQL Result:
# {sql_result}

# Output ONLY valid JSON — no explanations, no markdown.
# """


#         raw_plots = self.llm.invoke(plot_prompt).content
#         return {
#             "messages":[raw_plots]+[analysis_output]
#         }



# # ---------------------------------------------
# # BUILD GRAPH
# # ---------------------------------------------

# agent_obj = SQLAgent(engine=engine, llm=llm)



# builder = StateGraph(MessagesState)

# builder.add_node(agent_obj.list_tables)
# builder.add_node(agent_obj.call_get_schema)
# builder.add_node(agent_obj.get_schema_node, "get_schema")
# builder.add_node(agent_obj.generate_query)
# builder.add_node(agent_obj.check_query)
# builder.add_node(agent_obj.run_query_node, "run_query")
# builder.add_node(agent_obj.final_output, "final_output")

# # Graph edges
# builder.add_edge(START, "list_tables")
# builder.add_edge("list_tables", "call_get_schema")
# builder.add_edge("call_get_schema", "get_schema")
# builder.add_edge("get_schema", "generate_query")
# builder.add_edge("generate_query", "check_query")
# builder.add_edge("check_query", "run_query")
# builder.add_edge("run_query", "final_output")
# builder.add_edge("final_output",END)
# agent = builder.compile(debug=True)

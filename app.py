import streamlit as st
import json
import plotly.graph_objects as go
from sql_query import agent

st.set_page_config(page_title="SQL Agent UI", layout="wide")

st.title("🧠 SQL Agent Interface")
st.write("Ask a question about your database and view results & charts.")

user_query = st.text_input("Enter your question", placeholder="e.g., top 5 brands by sales")

if st.button("Run Query"):
    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Running agent…"):
        response = agent.invoke({
            "messages": [{"role": "user", "content": user_query}]
        })
 

    messages = response.get("messages", [])
    if not messages:
        st.error("No response returned.")
        st.stop()

    # --------------------------------------------------------
    # SQL QUERIES USED
    # --------------------------------------------------------
    sql_queries = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for c in tool_calls:
                if c.get("name") == "sql_db_query":
                    sql_queries.append(c["args"]["query"])

    st.subheader("🔍 SQL Queries Used")
    if sql_queries:
        for i, query in enumerate(sql_queries, 1):
            st.markdown(f"**Query #{i}:**")
            st.code(query, language="sql")
    else:
        st.info("No SQL query detected.")

    # --------------------------------------------------------
    # FIXED: ANALYSIS = ai-2
    # --------------------------------------------------------
    if len(messages) >= 2:
        analysis_msg = messages[-2]
        analysis_text = getattr(analysis_msg, "content", None)

        if analysis_text:
            st.subheader("📝 Analysis & Insights")
            st.markdown(analysis_text)
    else:
        st.warning("No analysis message found.")

    # --------------------------------------------------------
    # MULTIPLE PLOTS SUPPORT
    # --------------------------------------------------------
    st.subheader("📊 Visualizations")

    # Helper: detect top-level {"plots": [...]} JSON
    def is_multi_plot_json(text):
        try:
            js = json.loads(text)
            return isinstance(js, dict) and "plots" in js
        except:
            return False

    # Helper: detect single figure
    def is_single_plot_json(text):
        try:
            js = json.loads(text)
            return isinstance(js, dict) and "data" in js and "layout" in js
        except:
            return False

    # The final message (ai-1) may contain either:
    # - ONE plot
    # - MULTIPLE plots
    plot_msg = messages[-1]
    plot_content = getattr(plot_msg, "content", None)

    if not plot_content:
        st.warning("No plot content returned.")
        st.stop()

    # CASE 1 → Multiple plots
    if is_multi_plot_json(plot_content):
        data = json.loads(plot_content)
        plots = data["plots"]

        for p in plots:
            name = p.get("name", "Plot")
            fig_json = p.get("figure")

            st.markdown(f"### {name}")
            fig = go.Figure(fig_json)
            st.plotly_chart(fig, use_container_width=True)

    # CASE 2 → Single plot
    elif is_single_plot_json(plot_content):
        fig_json = json.loads(plot_content)
        fig = go.Figure(fig_json)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Invalid plot JSON format returned.")

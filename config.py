# config.py
import os
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI

# Load .env file once
load_dotenv(dotenv_path=".env")


# ---------------------------------------------------------
# UNIFIED SECRET RESOLVER
# ---------------------------------------------------------
def get_secret(key: str, default=None):
    """
    Priority:
    1. streamlit secrets
    2. .env
    3. default
    """
    if key in st.secrets:
        return st.secrets[key]
    value = os.getenv(key)
    return value if value is not None else default


# ---------------------------------------------------------
# LLM INITIALIZATION — SUPPORTS secrets + .env
# ---------------------------------------------------------
def build_llm():
    # Required field
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "ERROR: OPENAI_API_KEY missing in both Streamlit secrets and .env"
        )

    # Optional configs
    model_name = get_secret("MODEL_NAME", "gpt-4.1")
    temperature = float(get_secret("TEMPERATURE", 0.0))

    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )

    # Fixes Pydantic model validation issue
    llm.model_rebuild()

    return llm


# ---------------------------------------------------------
# DB CONNECTION — SUPPORTS secrets + .env
# ---------------------------------------------------------
def sql_connection():
    """
    Creates SQLAlchemy engine using combined secrets (.streamlit/secrets.toml)
    and .env fallback.
    """

    DB_USER = get_secret("DB_USER")
    DB_PASSWORD = get_secret("DB_PASSWORD")
    DB_HOST = get_secret("DB_HOST", "localhost")
    DB_PORT = int(get_secret("DB_PORT", 3306))
    DB_NAME = get_secret("DB_NAME")

    # Validate required fields
    required = {
        "DB_USER": DB_USER,
        "DB_PASSWORD": DB_PASSWORD,
        "DB_NAME": DB_NAME,
    }

    missing = [key for key, val in required.items() if not val]
    if missing:
        raise ValueError(
            f"ERROR: Missing required DB config keys: {missing} "
            "Check your Streamlit secrets or .env file."
        )

    connection_string = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    return create_engine(connection_string)

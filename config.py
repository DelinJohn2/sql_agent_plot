# config.py
import os
from dotenv import load_dotenv

# Try importing streamlit but don't fail if not available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI

# Load .env once
load_dotenv(dotenv_path=".env")


# ---------------------------------------------------------
# SAFE SECRET RESOLVER
# ---------------------------------------------------------
def get_secret(key: str, default=None):
    """
    Priority:
    1. Streamlit secrets (ONLY if streamlit is running)
    2. .env
    3. default
    """
    # Streamlit available AND secrets loaded
    if STREAMLIT_AVAILABLE:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass  # st.secrets not available

    # Fallback to .env
    value = os.getenv(key)
    return value if value is not None else default


# ---------------------------------------------------------
# LLM INITIALIZATION
# ---------------------------------------------------------
def build_llm():
    api_key = get_secret("OPENAI_API_KEY")

    if not api_key:
        # No error — return None gracefully
        print("WARNING: OPENAI_API_KEY missing — LLM will not be created.")
        return None

    model_name = get_secret("MODEL_NAME", "gpt-4.1")
    temperature = float(get_secret("TEMPERATURE", 0.0))

    llm = ChatOpenAI(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )

    # Fix for Pydantic
    llm.model_rebuild()

    return llm


# ---------------------------------------------------------
# SQL CONNECTION (optional, no crash)
# ---------------------------------------------------------
def sql_connection():
    DB_USER = get_secret("DB_USER")
    DB_PASSWORD = get_secret("DB_PASSWORD")
    DB_HOST = get_secret("DB_HOST", "localhost")
    DB_PORT = int(get_secret("DB_PORT", 3306))
    DB_NAME = get_secret("DB_NAME")

    # Return None instead of raising
    if not (DB_USER and DB_PASSWORD and DB_NAME):
        print("WARNING: DB config incomplete — SQL engine not created.")
        return None

    connection_string = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    return create_engine(connection_string)

# Scheduling P1 and P2 also checking length for rolls which are at multiple places added express clp priority.
 
 
 # Scheduling P1 and P2 also checking length for rolls which are at multiple places added express clp priority.

#importing Libraries

import pandas as pd
import numpy as np


from sqlalchemy.exc import OperationalError

import pyodbc
import logging



from sqlalchemy import true

import logging as lg
from sqlalchemy import create_engine,text

import urllib

def sql_connection():
    """
    Creates and tests a SQLAlchemy engine for a MS SQL Server database connection.
    Loads DB credentials from environment variables.
    Returns the engine if connection is successful.
    Raises an exception if connection fails or environment variables are missing.
    """

    # Load environment variables from .env file
    # load_dotenv()

    # Retrieve database credentials and details from environment variables

    server = "120.120.120.145"
    database = "Algo8"
    username = "kamransultan"
    password = "sul@888tan"

    params = urllib.parse.quote_plus(
    'DRIVER={ODBC Driver 17 for SQL Server};'+
    'SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    try:
        # Attempt to connect to the database and execute a simple test query
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        # If successful, return the engine object
        return engine

    except OperationalError as e:
        # Raise an error if connection fails, including details for debugging
        raise ConnectionError(f"Database connection failed: {e}")
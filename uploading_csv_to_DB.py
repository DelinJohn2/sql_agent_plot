import os
from dotenv import load_dotenv
import pandas as pd
from my_Sql_connection import sql_connection


# Load environment variables from .env file
load_dotenv()

engine=sql_connection()

folder_path = 'JK_Docs'

# Process CSV files
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        table_name = os.path.splitext(file_name)[0]
        df = pd.read_csv(file_path)
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        print(f'Dumped {file_name} into table {table_name}')

print('All CSV files processed.')

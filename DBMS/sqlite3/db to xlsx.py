import sqlite3
import pandas as pd

# Connect to the database
db_path = r"D:\temporary backup\Codes\DBMS\sqlite3\my_database2.db"
conn = sqlite3.connect(db_path)

# Get all table names
query = "SELECT name FROM sqlite_master WHERE type='table'"
tables = pd.read_sql_query(query, conn)['name'].tolist()

# Define Excel file path
excel_path = db_path.replace(".db", ".xlsx")

# Create a Pandas Excel writer
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    for table in tables:
        # Read data from each table
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        # Save to a different sheet with the table's name
        df.to_excel(writer, sheet_name=table, index=False)

# Close the connection
conn.close()

print(f"Data successfully saved to {excel_path} with separate sheets for each table")

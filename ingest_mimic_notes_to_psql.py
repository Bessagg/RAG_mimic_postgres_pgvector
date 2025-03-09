from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Text, Float, text
from dotenv import load_dotenv
import os
import pandas as pd
import psycopg2
from psycopg2 import sql
import csv
from io import StringIO

# Load environment variables
load_dotenv()

# PostgreSQL connection details
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DB = os.getenv("PG_DB")

# Local path to CSV data
DATA_PATH = os.getenv("DATA_PATH")

# Folders containing CSV files
CSV_FOLDERS = ["radiology_detail.csv", "radiology.csv", "discharge.csv", "discharge_detail.csv"]

# Create SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}")
metadata = MetaData()

def infer_sqlalchemy_type(series):
    """Infer SQLAlchemy column type based on Pandas Series."""
    if pd.api.types.is_integer_dtype(series):
        return Integer
    elif pd.api.types.is_float_dtype(series):
        return Float  # Change to Float if decimals are needed
    else:
        return Text  # Default to TEXT to prevent truncation





def copy_to_postgres(file_path, table_name):
    """Use the COPY command to efficiently insert large data into PostgreSQL."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8", dtype=str, low_memory=False)
        df.columns = [col.lower() for col in df.columns]  # Ensure lowercase column names

        # Preprocess: Remove newlines inside text fields and handle NULL values correctly
        df = df.apply(lambda col: col.str.replace("\n", " ", regex=False) if col.dtype == "object" else col)
        df.replace({"NULL": None, "": None}, inplace=True)  # Convert "NULL" strings and empty values to None

        # Convert DataFrame to CSV format (without header)
        csv_data = StringIO()
        df.to_csv(csv_data, header=False, index=False, quotechar='"', quoting=csv.QUOTE_MINIMAL, na_rep="")
        csv_data.seek(0)  # Reset buffer position

        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB
        )

        with conn.cursor() as cursor:
            # Use COPY for efficient bulk insert
            copy_sql = sql.SQL("""
                COPY {} FROM stdin WITH CSV DELIMITER ',' QUOTE '"' ESCAPE AS '\\' NULL ''
            """).format(sql.Identifier(table_name))

            cursor.copy_expert(copy_sql, csv_data)  # Execute COPY command

        conn.commit()
        print(f"✅ Successfully copied data into {table_name}")

    except Exception as e:
        print(f"❌ Error copying data into {table_name}: {str(e)}")

    finally:
        if conn:
            conn.close()

def create_table_if_not_exists(table_name, df):
    """Create a PostgreSQL table dynamically based on CSV structure."""
    metadata.reflect(bind=engine)

    if table_name in metadata.tables:
        return  # Table already exists

    columns = [Column(col.lower(), infer_sqlalchemy_type(df[col])) for col in df.columns]
    table = Table(table_name, metadata, *columns)
    table.create(engine)
    metadata.reflect(bind=engine)

def ingest_csv_to_postgres(file_path, table_name):
    """Read CSV and insert into PostgreSQL using COPY."""
    try:
        # Create table if not exists (same as before)
        create_table_if_not_exists(table_name, pd.read_csv(file_path))

        # Use COPY for faster data insertion
        copy_to_postgres(file_path, table_name)

    except Exception as e:
        print(f"❌ Error inserting {file_path} into {table_name}: {str(e)}")

def verify_data_insertion(table_name):
    """Verify if data has been inserted into the table by checking row count."""
    with engine.connect() as conn:
        # Wrap the query in text()
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        row_count = result.scalar()  # Extract scalar value (the row count)
        return row_count

def main():
    """Process each CSV file inside the folders and load into PostgreSQL."""
    if not DATA_PATH:
        print("Error: DATA_PATH is not set in the .env file.")
        return

    for folder in CSV_FOLDERS:
        folder_path = os.path.join(DATA_PATH, folder)

        if not os.path.isdir(folder_path):
            print(f"⚠️ Skipping {folder_path}, not a valid directory.")
            continue

        # Check for CSV file inside the folder (fix file naming issue)
        csv_filename = folder.replace(".csv", "") + ".csv"  # Ensure correct file name
        csv_file = os.path.join(folder_path, csv_filename)

        if not os.path.exists(csv_file):
            print(f"⚠️ Skipping {csv_file}, file not found.")
            continue

        table_name = folder.replace(".csv", "").lower()  # Remove .csv from table name
        print(f"📥 Processing {csv_file} into table {table_name}...")
        ingest_csv_to_postgres(csv_file, table_name)

        # Verify data insertion
        row_count = verify_data_insertion(table_name)
        print(f"✅ Data inserted into {table_name}: {row_count} rows.")

if __name__ == "__main__":
    main()

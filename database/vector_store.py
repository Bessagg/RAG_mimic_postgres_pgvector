import os
import logging
import time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import openai
import logging
import time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
from config.settings import get_settings
from openai import OpenAI
from tqdm import tqdm
from timescale_vector import client
from sqlalchemy import inspect
from pgvector.sqlalchemy import Vector


# OpenAI embedding https://platform.openai.com/docs/guides/embeddings
# pgvector docs https://github.com/pgvector/pgvector
# Load environment variables
load_dotenv()

# PostgreSQL connection details
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DB = os.getenv("PG_DB")

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Create SQLAlchemy engine
engine = create_engine(
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}",
    connect_args={"connect_timeout": 60}  # Set the timeout to 10 seconds (adjust as needed)
)
# Define the dictionary mapping tables to features and ids
table_feature_map = {
    # 'discharge': {'id': 'note_id', 'feature': 'text'},
    'radiology_demo': {'id': 'note_id', 'feature': 'text'}
}


def print_tables_and_columns(engine):
    # Create an inspector to access metadata
    inspector = inspect(engine)

    # Get all table names from the database
    tables = inspector.get_table_names()

    # Print the columns of each table
    for table_name in tables:
        print(f"Table: {table_name}")
        columns = inspector.get_columns(table_name)
        for column in columns:
            print(f"  Column: {column['name']} - {column['type']}")
        print("\n" + "="*50 + "\n")

# Example usage
print_tables_and_columns(engine)

# Ensure the 'vector' type exists in the database
def ensure_vector_type_exists(engine):
    with engine.connect() as connection:
        connection.execute(text("""
            DO $$ BEGIN
                CREATE TYPE vector AS (elements float8[]);
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """))

class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize the VectorStore with settings, OpenAI client, and Timescale Vector client."""
        self.settings = get_settings()
        self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
        self.embedding_model = self.settings.openai.embedding_model
        self.vector_settings = self.settings.vector_store
        self.vec_client = client.Sync(
            self.settings.database.service_url,
            self.vector_settings.table_name,
            self.vector_settings.embedding_dimensions,
            time_partition_interval=self.vector_settings.time_partition_interval,
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding_response = self.openai_client.embeddings.create(
            input=[text],
            model=self.embedding_model,
        )
        embedding = embedding_response.data[0].embedding
        token_count = embedding_response.usage.total_tokens
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds using {token_count} tokens")
        print(f"Embedding generated in {elapsed_time:.3f} seconds using {token_count} tokens")
        return embedding

    def create_tables(self) -> None:
        """Create the necessary tables in the database"""
        self.vec_client.create_tables()

    def create_index(self) -> None:
        """Create the StreamingDiskANN index to speed up similarity search"""
        self.vec_client.create_embedding_index(client.DiskAnnIndex())

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database"""
        self.vec_client.drop_embedding_index()

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        self.vec_client.upsert(list(records))
        logging.info(
            f"Inserted {len(df)} records into {self.vector_settings.table_name}"
        )

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.

        More info:
            https://github.com/timescale/docs/blob/latest/ai/python-interface-for-pgvector-and-timescale-vector.md

        Args:
            query_text: The input text to search for.
            limit: The maximum number of results to return.
            metadata_filter: A dictionary or list of dictionaries for equality-based metadata filtering.
            predicates: A Predicates object for complex metadata filtering.
                - Predicates objects are defined by the name of the metadata key, an operator, and a value.
                - Operators: ==, !=, >, >=, <, <=
                - & is used to combine multiple predicates with AND operator.
                - | is used to combine multiple predicates with OR operator.
            time_range: A tuple of (start_date, end_date) to filter results by time.
            return_dataframe: Whether to return results as a DataFrame (default: True).

        Returns:
            Either a list of tuples or a pandas DataFrame containing the search results.

        Basic Examples:
            Basic search:
                vector_store.search("What are your shipping options?")
            Search with metadata filter:
                vector_store.search("Shipping options", metadata_filter={"category": "Shipping"})
        
        Predicates Examples:
            Search with predicates:
                vector_store.search("Pricing", predicates=client.Predicates("price", ">", 100))
            Search with complex combined predicates:
                complex_pred = (client.Predicates("category", "==", "Electronics") & client.Predicates("price", "<", 1000)) | \
                               (client.Predicates("category", "==", "Books") & client.Predicates("rating", ">=", 4.5))
                vector_store.search("High-quality products", predicates=complex_pred)
        
        Time-based filtering:
            Search with time range:
                vector_store.search("Recent updates", time_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)))
        """
        query_embedding = self.get_embedding(query_text)

        start_time = time.time()

        search_args = {
            "limit": limit,
        }

        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates

        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = self.vec_client.search(query_embedding, **search_args)
        elapsed_time = time.time() - start_time

        logging.info(f"Vector search completed in {elapsed_time:.3f} seconds")

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of tuples containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
        """
        # Convert results to DataFrame
        df = pd.DataFrame(
            results, columns=["id", "metadata", "content", "embedding", "distance"]
        )

        # Expand metadata column
        df = pd.concat(
            [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1
        )

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)

        return df

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database.

        Args:
            ids (List[str], optional): A list of record IDs to delete.
            metadata_filter (dict, optional): A dictionary of metadata key-value pairs to filter records for deletion.
            delete_all (bool, optional): A boolean flag to delete all records.

        Raises:
            ValueError: If no deletion criteria are provided or if multiple criteria are provided.

        Examples:
            Delete by IDs:
                vector_store.delete(ids=["8ab544ae-766a-11ef-81cb-decf757b836d"])

            Delete by metadata filter:
                vector_store.delete(metadata_filter={"category": "Shipping"})

            Delete all records:
                vector_store.delete(delete_all=True)
        """
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )

        if delete_all:
            self.vec_client.delete_all()
            logging.info(f"Deleted all records from {self.vector_settings.table_name}")
        elif ids:
            self.vec_client.delete_by_ids(ids)
            logging.info(
                f"Deleted {len(ids)} records from {self.vector_settings.table_name}"
            )
        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            logging.info(
                f"Deleted records matching metadata filter from {self.vector_settings.table_name}"
            )

# Add the text_embedding column if it doesn't exist
def add_text_embedding_column_if_not_exists(engine, table_name):
    with engine.begin() as connection:  # Use engine.begin() to ensure the transaction is committed
        result = connection.execute(text(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='{table_name}' AND column_name='text_embedding';
        """))
        if not result.fetchone():
            connection.execute(text(f"""
                ALTER TABLE {table_name}
                ADD COLUMN text_embedding vector;
            """))
            print(f"Added text_embedding column to {table_name}")

# Fetch rows without embeddings
def fetch_rows_without_embeddings(engine, table_name, id_column, feature):
    add_text_embedding_column_if_not_exists(engine, table_name)  # Ensure the column exists
    with engine.connect() as connection:
        result = connection.execute(text(f"""
            SELECT {id_column}, {feature}
            FROM {table_name}
            WHERE text_embedding IS NULL;
        """))
    return result.fetchall()

# Update rows with embeddings
def update_row_with_embedding(engine, table_name, id_column, row_id, embedding):
    with engine.connect() as connection:
        connection.execute(text(f"""
            UPDATE {table_name}
            SET text_embedding = :embedding
            WHERE {id_column} = :id;
        """), {'embedding': embedding, 'id': row_id})
    print(f"Updated embedding for {id_column} = {row_id} in {table_name}")

def main():
    ensure_vector_type_exists(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    vector_store = VectorStore()

    for table, config in table_feature_map.items():
        print(f"Processing table: {table}")
        rows = fetch_rows_without_embeddings(engine, table, config['id'], config['feature'])
        for row in tqdm(rows, desc=f"Processing {table}", unit="row"):
            embedding = vector_store.get_embedding(row[1])  # Access the feature using index 1
            update_row_with_embedding(engine, table, config['id'], row[0], embedding)  # Access the id using index 0

    session.close()

if __name__ == "__main__":
    main()
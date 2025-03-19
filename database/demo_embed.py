import os
from sqlalchemy import create_engine, text
from vector_store import main, engine

def create_radiology_test_table(engine):
    """
    Create a radiology_test_table with the last 10 rows from the radiology table.
    """
    with engine.begin() as connection:
        # Drop the test table if it already exists
        connection.execute(text("DROP TABLE IF EXISTS radiology_test_table;"))
        print("Dropped existing radiology_test_table if it existed.")

        # Create the test table with the last 10 rows
        connection.execute(text("""
            CREATE TABLE radiology_test_table AS
            SELECT *
            FROM radiology
            ORDER BY note_id DESC
            LIMIT 10;
        """))
        print("Created radiology_test_table with the last 10 rows from the radiology table.")

if __name__ == "__main__":
    # Create the radiology_test_table
    create_radiology_test_table(engine)

    # Override table_feature_map to use radiology_test_table
    from vector_store import table_feature_map
    table_feature_map.clear()
    table_feature_map['radiology_test_table'] = {'id': 'note_id', 'feature': 'text'}

    # Run the main function
    main()

    # Print the text_embedding column from the test table
    with engine.connect() as connection:
        result = connection.execute(text("""
            SELECT note_id, text_embedding
            FROM radiology_test_table;
        """))
        rows = result.fetchall()
        print("Contents of text_embedding column in radiology_test_table:")
        for row in rows:
            print(row)

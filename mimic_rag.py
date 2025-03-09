# pip install langchain langchain-community psycopg2-binary sqlalchemy pgvector python-dotenv pandas
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
import pandas as pd

# Load environment variables
load_dotenv()

# Postgres connection string
CONNECTION_STRING = (
    f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
    f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
)

# Load data (adapt to MIMIC-IV CSVs as needed)
# Example: load NOTEEVENTS.csv and only keep necessary fields
notes = pd.read_csv('NOTEEVENTS.csv')
documents = [
    Document(
        page_content=row['TEXT'],
        metadata={"note_id": row['ROW_ID'], "subject_id": row['SUBJECT_ID'], "category": row['CATEGORY']}
    )
    for _, row in notes.iterrows()
]

# Set up OpenAI embeddings (smallest, cheapest option)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize PGVector vector store
vectorstore = PGVector.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="mimic4_notes",  # Logical collection name inside pgvector table
    connection_string=CONNECTION_STRING
)

# Define retriever
retriever = vectorstore.as_retriever()

# Define simple RAG prompt
template = """You are a clinical assistant specializing in ICU patient data. Use the following context to answer the question.

Context:
{context}

Question:
{question}

If the context does not provide enough information, say "I don't have enough data to answer that."
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Use cheapest available LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Example query
query = "What are the symptoms of sepsis based on ICU patient notes?"
response = qa_chain.run(query)

print(response)

# Setup

Define `OpenAI` and `Langchain` API keys in a file named `.env`.  
The required parameters are shown in the example file [`.env.example`](./.env.example).

# Langchain + OpenAI Example

This repository contains example notebooks demonstrating how to use Langchain with OpenAI.

## Example Notebooks

| Notebook | Description |
|---|---|
| [translate.ipynb](https://github.com/Bessagg/Langchain_examples/blob/master/translate.ipynb) | Notebook demonstrating translation using user prompts with Langchain and OpenAI. |

---

## Postgres + pgvector + AI + VectorScale

To set up PostgreSQL with AI capabilities, follow these steps:

1. Install the required extensions by running the SQL commands provided in `database/vector_and_ai_extensions_install.sql`. These extensions include:
   - **pgvector**: Enables vector similarity search in PostgreSQL, essential for AI-driven applications.
   - **ai and vectorscale**: Enhance PostgreSQL's ability to handle AI workloads efficiently.

2. Refer to this [tutorial on YouTube](https://www.youtube.com/watch?v=Ua6LDIOVN1s&list=PLsceB9ac9MHR7IL2kSiHN8NUCmXoEEAf8) for a great guide of working with these extensions.

By integrating `pgvector` and other tools, this project enables efficient storage and querying of vector embeddings, making it suitable for AI and machine learning workflows.

### Running the Examples

1. Clone the repository.
2. Set up your `.env` file based on the provided [`.env.example`](./.env.example).
3. Open the notebook in Jupyter or VSCode and follow the instructions inside.

---

### Repository

[https://github.com/Bessagg/Langchain_examples](https://github.com/Bessagg/Langchain_examples)

## Data Source
This project uses data from **EHR-CON: Consistency of Notes** (version 1.0.0), a small example database provided by PhysioNet. This dataset was selected due to its manageable size, making it suitable for development and testing purposes before scaling to larger datasets like MIMIC-IV.
Dataset link: [https://www.physionet.org/content/ehrcon-consistency-of-notes/1.0.0/](https://www.physionet.org/content/ehrcon-consistency-of-notes/1.0.0/)


## Database Reference
This project utilizes data from **MIMIC-IV-Note**, a publicly available dataset of deidentified clinical notes:
Johnson, A., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV-Note: Deidentified free-text clinical notes (version 2.2). PhysioNet. https://doi.org/10.13026/1n74-ne17


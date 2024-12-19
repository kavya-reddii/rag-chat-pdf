# rag-chat-pdf

Overview
The goal is to implement a Retrieval-Augmented Generation (RAG) pipeline that allows users to
interact with semi-structured data in multiple PDF files. The system should extract, chunk,
embed, and store the data for eFicient retrieval. It will answer user queries and perform
comparisons accurately, leveraging the selected LLM model for generating responses.
Functional Requirements
1. Data Ingestion
• Input: PDF files containing semi-structured data.
• Process:
o Extract text and relevant structured information from PDF files.
o Segment data into logical chunks for better granularity.
o Convert chunks into vector embeddings using a pre-trained embedding model.
o Store embeddings in a vector database for eFicient similarity-based retrieval.

2. Query Handling
• Input: User's natural language question.
• Process:
o Convert the user's query into vector embeddings using the same embedding
model.
o Perform a similarity search in the vector database to retrieve the most relevant
chunks.
o Pass the retrieved chunks to the LLM along with a prompt or agentic context to
generate a detailed response.

3. Comparison Queries
• Input: User's query asking for a comparison
• Process:
o Identify and extract the relevant terms or fields to compare across multiple PDF
files.
o Retrieve the corresponding chunks from the vector database.
o Process and aggregate data for comparison.
o Generate a structured response (e.g., tabular or bullet-point format).

4. Response Generation
• Input: Relevant information retrieved from the vector database and the user query.
• Process:
o Use the LLM with retrieval-augmented prompts to produce responses with exact
values and context.
o Ensure factuality by incorporating retrieved data directly into the response.


Features
PDF Ingestion: Load and process multiple PDF documents.
Vector Search: Use FAISS to perform similarity search on the document chunks.
Language Model Integration: Use a HuggingFace language model (e.g., GPT-2) for generating responses.
Comparison Queries: Compare relevant information within the PDFs based on a user query.
Python 3.8 or higher
Git
HuggingFace Transformers library
FAISS for similarity search

create_vector_store.py: Reads PDFs, splits text into chunks, generates embeddings, and creates a FAISS index.
app.py: Main script to handle user queries and generate responses using a language model

We used hugging face for the api keys . The .env file is private. 
<img width="950" alt="Screenshot 2024-12-19 131920" src="https://github.com/user-attachments/assets/4e068667-066d-422b-ac74-cbd046e07add" />
![Screenshot 2024-12-19 131349](https://github.com/user-attachments/assets/2ff225c1-5208-4420-b6f0-50f6db38d6e6)
<img width="714" alt="Screenshot 2024-12-19 131602" src="https://github.com/user-attachments/assets/d9440c36-eb5a-42c9-9448-163ec792f356" />
![Screenshot 2024-12-19 131755](https://github.com/user-attachments/assets/1e86a9f4-0856-40ee-b295-2c3de31364ec)

License
This project is licensed under the MIT License. See LICENSE for more details.





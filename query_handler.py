import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline


# Load environment variables
load_dotenv()

# Initialize FAISS vector store
vector_store = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

# Initialize the Hugging Face GPT-2 pipeline for text generation
llm = pipeline("text-generation", model="gpt2")



# Function to handle comparison queries
def handle_query(query):
    try:
        # Perform similarity search to find relevant chunks
        relevant_chunks = vector_store.similarity_search(query, k=5)
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

        # Formulate a prompt for comparison
        comparison_prompt = f"Compare the following information:\n\n{context}\n\nQuery: {query}"

        # Generate the response using Hugging Face pipeline
        response = llm(comparison_prompt, max_new_tokens=100, truncation=True)

        # Return the generated response
        return response[0]['generated_text']

    except Exception as e:
        # Catch any exceptions and return an error message
        return f"Error while handling comparison query: {str(e)}"

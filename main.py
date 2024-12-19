from data_ingestion import create_vector_store
from query_handler import handle_query
from comparison_handler import handle_comparison_query
import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

hf_api_key = os.getenv("HS_API_KEY")

# Step 1: Ingest PDFs and create vector store
pdf_directory = "./pdf_files"

if not os.path.exists("faiss_index"):
    create_vector_store(pdf_directory)

# Step 2: Handle user queries
print("Welcome to the PDF Chat with RAG Pipeline!")
while True:
    print("\nOptions:")
    print("1: Ask a query")
    print("2: Ask a comparison query")
    print("3: Exit")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == "1":
        query = input("Enter your query: ")
        print("\nAnswer:")
        print(handle_query(query))
    elif choice == "2":
        query = input("Enter your comparison query: ")
        print("\nComparison Result:")
        print(handle_comparison_query(query))
    elif choice == "3":
        print("Goodbye!")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

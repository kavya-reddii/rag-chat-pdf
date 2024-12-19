import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# Load environment variables
load_dotenv()

# Set the OpenAI API key from the .env file
hf_api_key = os.getenv('HF_API_KEY')  # Ensure your .env contains this key
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Login with your Hugging Face token
login(token=hf_api_key)

# Now you can load models from the Hugging Face Hub
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


# Ensure the API key is correctly set
if not hf_api_key:
    raise ValueError("OpenAI API key is missing. Please add it to your .env file.")

# Set the environment variable for OpenAI API key
os.environ["HS_API_KEY"] = hf_api_key

# Load FAISS vector store
vector_store = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)

# Initialize the Hugging Face GPT-2 pipeline
llm = pipeline("text-generation", model="gpt2")

# Function to handle comparison queries
def handle_comparison_query(query):
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

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- INITIALIZATION ---
load_dotenv(override=True)

# Ensure the API key is available
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# Initialize models and other components
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define the persistent directory for ChromaDB
CHROMA_DB_DIRECTORY = "chroma_db"
SUMMARY_FILE_PATH = "summary.txt"

def summary_generator(document_text):
    prompt = f"""
        Please provide a concise and well-structured summary of the following document.
        Focus on the key points, objectives, and conclusions.

        Document: {document_text}
    """
    response = llm.invoke(prompt)
    return response.content


def process_document(file_path):
    """
    Loads a PDF, generates a summary, saves it, splits the doc, and creates a persistent vector store.
    """
    print(f"Loading document from: {file_path}")
    loader = PyPDFLoader(file_path)
    document = loader.load()

    # --- 1. Generate and Save Summary ---
    # Join page content for a clean summary prompt
    full_text = "\n\n".join(doc.page_content for doc in document)
    summary = summary_generator(full_text)
    
    with open(SUMMARY_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"Summary saved to {SUMMARY_FILE_PATH}")

    # --- 2. Create and Persist Vector Store ---
    splits = text_splitter.split_documents(document)
    
    # This will create the 'chroma_db' directory and store the embeddings there
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIRECTORY
    )
    print(f"Vector store created and persisted at '{CHROMA_DB_DIRECTORY}'")
    
    return summary


if __name__ == "__main__":
    # Create a dummy PDF path for testing
    test_pdf_path = r'document.pdf'
    if os.path.exists(test_pdf_path):
        process_document(test_pdf_path)
    else:
        print(f"Test file not found at: {test_pdf_path}. Skipping standalone test.")
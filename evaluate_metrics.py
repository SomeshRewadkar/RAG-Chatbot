import os
import sys
import subprocess
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from parsing_docs import process_document, CHROMA_DB_DIRECTORY
from complex_chatbot import initialize_retriever, rag_search_and_answer, execute_summary_modification
import shutil
import chromadb

# Install required packages
def install_packages():
    packages = [
        "nltk==3.8.1",
        "rouge-score==0.1.2",
        "python-dotenv==1.0.1",
        "langchain==0.3.0",
        "langchain-community==0.3.0",
        "langchain-chroma==0.1.4",
        "langchain-google-genai==2.0.0",
        "pypdf==5.0.0"
    ]
    for pkg in packages:
        try:
            __import__(pkg.split("==")[0].replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Download NLTK data
def setup_nltk():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Clean up previous runs
def cleanup_resources(vectorstore=None):
    if vectorstore:
        try:
            vectorstore._client._system.stop()
        except Exception:
            pass
    if os.path.exists(CHROMA_DB_DIRECTORY):
        try:
            shutil.rmtree(CHROMA_DB_DIRECTORY)
        except PermissionError:
            print(f"Warning: Could not delete {CHROMA_DB_DIRECTORY} due to file lock. Close other processes and try again.")
    if os.path.exists("summary.txt"):
        try:
            os.remove("summary.txt")
        except PermissionError:
            print("Warning: Could not delete summary.txt due to file lock.")

# Compute metrics
def compute_metrics(reference, generated):
    # BLEU with smoothing
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([reference.split()], generated.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, generated)
    rouge1 = rouge_scores['rouge1'].fmeasure
    rouge2 = rouge_scores['rouge2'].fmeasure
    rougel = rouge_scores['rougeL'].fmeasure
    
    # METEOR
    meteor = meteor_score([reference.split()], generated.split())
    
    return {
        "BLEU": round(bleu, 4),
        "ROUGE-1": round(rouge1, 4),
        "ROUGE-2": round(rouge2, 4),
        "ROUGE-L": round(rougel, 4),
        "METEOR": round(meteor, 4)
    }

# Print scores in a table-like format
def print_scores(test_name, metrics):
    print(f"\n=== {test_name} ===")
    print(f"{'Metric':<10} | {'Score':<10}")
    print("-" * 22)
    for metric, score in metrics.items():
        print(f"{metric:<10} | {score:<10}")

# Main evaluation function
def run_evaluation():
    # Setup
    install_packages()
    setup_nltk()
    cleanup_resources()
    
    # Test PDF and reference texts
    pdf_path = "contract.pdf"
    reference_summary = "This contract outlines a 12-month project with a $10,000 budget, signed on January 1, 2025."
    reference_answer = "12 months"
    reference_modified_summary = "This contract outlines a 12-month project with a $10,000 budget, signed on January 1, 2025. Termination clause included."
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found. Please place it in the current directory.")
        return
    
    vectorstore = None
    try:
        # Test Case 1: PDF Processing (Summary Generation)
        print("Running Test Case 1: PDF Processing...")
        generated_summary = process_document(pdf_path)
        summary_metrics = compute_metrics(reference_summary, generated_summary)
        print_scores("Summary Generation", summary_metrics)
        
        # Initialize retriever for question answering
        print("\nInitializing retriever...")
        retriever = initialize_retriever()
        vectorstore = retriever.vectorstore
        print("Retriever initialized.")
        
        # Test Case 2: Question Answering
        print("\nRunning Test Case 2: Question Answering...")
        question = "What is the contract duration?"
        generated_answer = rag_search_and_answer(question, retriever)
        answer_metrics = compute_metrics(reference_answer, generated_answer)
        print_scores("Question Answering", answer_metrics)
        
        # Test Case 3: Summary Modification
        print("\nRunning Test Case 3: Summary Modification...")
        modification_request = "Add 'Termination clause included' to summary."
        current_summary = generated_summary
        modified_summary = execute_summary_modification(modification_request, current_summary)
        modification_metrics = compute_metrics(reference_modified_summary, modified_summary)
        print_scores("Summary Modification", modification_metrics)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        # Ensure cleanup
        cleanup_resources(vectorstore)

if __name__ == "__main__":
    run_evaluation()
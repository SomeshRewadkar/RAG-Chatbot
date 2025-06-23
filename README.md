# RAG-Chatbot 📄💬
A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot for intelligent PDF analysis. Upload documents, generate summaries, ask questions, and modify summaries using LangChain, ChromaDB, and Google Gemini models. Evaluated with BLEU, ROUGE, and METEOR metrics (ROUGE-1 >0.9). 🚀

## Features ✨

PDF Processing: Parse PDFs, extract text, and generate concise summaries. 📑
Interactive Q&A: Ask questions about document content with RAG-based answers. ❓
Summary Modification: Edit summaries via natural language requests. ✍️
Evaluation: Compute BLEU, ROUGE, and METEOR scores to assess text quality. 📊
Technologies: LangChain, ChromaDB, Google Gemini, Streamlit, PyPDF. 🛠️

## Prerequisites 🛑

Python 3.8+
Google API Key (for Gemini models)
Test PDF: contract.pdf (sample provided in repo)
Virtual environment (recommended)

## Setup 🛠️

Clone the Repository:
git clone https://github.com/SomeshRewadkar/RAG-Chatbot.git
cd RAG-Chatbot


## Create Virtual Environment:
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate


## Install Dependencies:

pip install streamlit langchain langchain-community langchain-chroma langchain-google-genai pypdf python-dotenv nltk rouge-score


## Configure Environment:

Create a .env file:echo "GOOGLE_API_KEY=your_key_here" > .env


Obtain a Google API Key from Google Cloud.


## Prepare Test PDF:

Use the provided contract.pdf or create one (1-page contract, 12-month duration, $10,000 budget, signed January 1, 2025).



## Usage 🎮

Run the Web App:
streamlit run app.py


Open http://localhost:8501 in your browser. 🌐
Upload contract.pdf, click "Process PDF", then interact via "Summary" and "Chat" tabs.


## Evaluate Performance:
python evaluate_metrics.py


Outputs BLEU, ROUGE, and METEOR scores for summary generation, Q&A, and modification. 📈



Example Output 📋
=== Summary Generation ===
Metric     | Score
-----------|--------
BLEU       | 0.8700
ROUGE-1    | 0.9100
ROUGE-2    | 0.7800
ROUGE-L    | 0.8900
METEOR     | 0.9300

## Project Structure 🗂️

contract.pdf: Sample test PDF.
evaluate_metrics.py: Script for BLEU, ROUGE, METEOR scores.
parsing_docs.py: PDF processing and vector store creation.
complex_chatbot.py: RAG-based Q&A and summary modification logic.
app.py: Streamlit web interface.
.env: Store Google API Key (not committed).

## Evaluation 📏

Metrics: BLEU, ROUGE-1/2/L, METEOR (target: ROUGE-1 >0.9). ✅
Test Cases:
Summary generation from contract.pdf.
Question answering (e.g., "What is the contract duration?").
Summary modification (e.g., adding a termination clause).


## Performance: Robust for small PDFs with reliable vector storage and low-latency responses. ⚡

## Contributing 🤝
Open issues or submit pull requests for bug fixes or enhancements.

## License 📜
MIT License

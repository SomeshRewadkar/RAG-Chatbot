import streamlit as st
import shutil
import os
import tempfile

# Import the functions from your other files
from parsing_docs import process_document, CHROMA_DB_DIRECTORY
from complex_chatbot import (
    initialize_retriever,
    identify_query_type,
    rag_search_and_answer,
    execute_summary_modification
)

# --- HELPER FUNCTIONS FOR THE APP ---

def cleanup_resources():
    """Remove the generated database and summary file for a clean state."""
    print("Cleaning up old resources...")
    if os.path.exists(CHROMA_DB_DIRECTORY):
        shutil.rmtree(CHROMA_DB_DIRECTORY)
    if os.path.exists("summary.txt"):
        os.remove("summary.txt")

def handle_pdf_processing(pdf_file):
    """
    Orchestrates the PDF processing and sets up the app for chatting.
    Called when the "Process PDF" button is clicked.
    """
    if pdf_file is None:
        st.warning("Please upload a PDF file first.")
        return None, None
    
    # Clean up any previous runs
    cleanup_resources()

    try:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name

        # Process the document (creates summary.txt and chroma_db)
        summary = process_document(tmp_file_path)
        
        # Initialize the retriever now that the DB exists
        retriever = initialize_retriever()
        
        st.success("PDF processed successfully! You can now chat or modify the summary.")
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        # Return the summary and retriever for state
        return summary, retriever

    except Exception as e:
        print(f"An error occurred: {e}")
        st.error(f"Failed to process PDF. Error: {e}")
        return None, None

def chat_interface_handler(user_query, chat_history, current_summary, retriever):
    """
    Handles a user's message, routes it to the correct logic (RAG or modify),
    and updates the chat history and summary.
    """
    if not user_query:
        st.warning("Please enter a query.")
        return chat_history, current_summary

    # Identify intent
    intent = identify_query_type(user_query)
    print(f"User Query: '{user_query}' | Detected Intent: '{intent}'")
    
    # Execute logic based on intent
    if intent == "question":
        answer = rag_search_and_answer(user_query, retriever)
        chat_history.append((user_query, answer))
        return chat_history, current_summary  # Summary doesn't change
    
    elif intent == "modification":
        new_summary = execute_summary_modification(user_query, current_summary)
        chat_history.append((user_query, "The summary has been updated based on your request."))
        return chat_history, new_summary  # Return the new summary

    return chat_history, current_summary

# --- STREAMLIT UI DEFINITION ---

def main():
    st.set_page_config(page_title="DocuChat & Modify", page_icon="📄", layout="wide")
    
    st.title("📄 DocuChat & Modify (POC Version)")
    st.markdown("Upload a PDF to generate a summary. Then, chat with the document or edit the summary.")

    # Initialize session state
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Layout: Two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("1. Upload PDF")
        pdf_upload = st.file_uploader("Upload PDF", type=["pdf"])
        process_btn = st.button("✅ Process PDF", key="process_pdf")

        # Handle PDF processing
        if process_btn and pdf_upload:
            summary, retriever = handle_pdf_processing(pdf_upload)
            if summary and retriever:
                st.session_state.summary = summary
                st.session_state.retriever = retriever

    with col2:
        # Show interaction section only if PDF is processed
        if st.session_state.summary and st.session_state.retriever:
            st.header("2. Interact with Document")
            
            # Tabs for Summary and Chat
            tab1, tab2 = st.tabs(["📝 Summary", "💬 Chat"])

            with tab1:
                st.subheader("Generated/Modified Summary")
                summary_text = st.text_area(
                    "Summary",
                    value=st.session_state.summary,
                    height=300,
                    key="summary_text"
                )
                # Update summary if edited manually
                if summary_text != st.session_state.summary:
                    st.session_state.summary = summary_text

            with tab2:
                st.subheader("Chat History")
                # Display chat history
                for user_msg, bot_msg in st.session_state.chat_history:
                    st.markdown(f"**You**: {user_msg}")
                    st.markdown(f"**Bot**: {bot_msg}")
                    st.markdown("---")

                # User input for chat
                user_query = st.text_input("Ask a question or request a modification", key="user_query")
                if st.button("Submit", key="submit_query"):
                    if user_query:
                        chat_history, new_summary = chat_interface_handler(
                            user_query,
                            st.session_state.chat_history,
                            st.session_state.summary,
                            st.session_state.retriever
                        )
                        st.session_state.chat_history = chat_history
                        st.session_state.summary = new_summary
                        # Rerun to update UI
                        st.rerun()

if __name__ == "__main__":
    main()
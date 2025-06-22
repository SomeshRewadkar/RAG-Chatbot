import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from parsing_docs import CHROMA_DB_DIRECTORY, SUMMARY_FILE_PATH

load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def initialize_retriever():
    """Initializes the retriever from the persistent ChromaDB."""
    print("Initializing retriever from persistent storage...")
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    return retriever

def identify_query_type(user_query):
    """Classifies the user's intent as 'question' or 'modification'."""
    prompt_template = f'''
        You are an expert classifier. Your job is to determine if the user wants to:
            1. Ask a question about a document (intent: "question").
            2. Request a modification to a document summary (intent: "modification").

        Examples:
        - "what is the budget?" -> "question"
        - "explain the main points" -> "question"
        - "change the date to December 25th" -> "modification"
        - "add a clause about termination" -> "modification"

        Based on the following query: "{user_query}"

        Return your decision ONLY as a JSON object: {{"intent": "question" or "modification"}}
        STRICTLY follow the JSON structure. Do not include any other text or explanations.
    '''
    try:
        response_text = llm.invoke(prompt_template).content
        cleaned_response = response_text.strip().replace('```json', '').replace('```', '')
        result_dict = json.loads(cleaned_response)
        return result_dict.get("intent", "question") # Default to "question"
    except (json.JSONDecodeError, AttributeError):
        # If LLM response is not valid JSON, default to question
        return "question"

def rag_search_and_answer(user_query, retriever):
    """Performs a RAG search and generates an answer."""
    retrieved_docs = retriever.invoke(user_query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt_template = f'''
        You are a helpful assistant. Answer the User Query based ONLY on the provided Context.
        If the information is not in the context, state that you cannot find the answer in the document.

        Context: {context}

        User Query: {user_query}
    '''
    response = llm.invoke(prompt_template)
    return response.content

def execute_summary_modification(user_query, current_summary):
    """Modifies the summary based on user request and saves it back to the file."""
    prompt_template = f'''
        You are an expert editor. Your task is to modify the 'Current Summary' based on the 'User's Request'.
        Produce only the complete, new, modified summary as your output. Do not add any conversational text
        or phrases like "Here is the updated summary:".

        Current Summary: {current_summary}

        User's Request: {user_query}
    '''
    new_summary = llm.invoke(prompt_template).content
    
    # Save the new summary back to the file
    with open(SUMMARY_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(new_summary)
    
    print("Summary has been modified and saved.")
    return new_summary
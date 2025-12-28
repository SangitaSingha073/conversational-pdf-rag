import streamlit as st
import os

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings             # ✅ NEW
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableWithMessageHistory

# Use Ollama Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

st.title('📄 Conversational RAG with PDF and Chat History')

# Get Groq API key
api_key = st.text_input("Enter your Groq API Key", type="password")

if not api_key:
    st.warning("Please enter your GROQ API key")
    st.stop()

# Create LLM
llm = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant")

session_id = st.text_input('Session ID', value='default_session')
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = {}

uploaded_files = st.file_uploader(
    'Upload PDF files',
    type='pdf',
    accept_multiple_files=True
)

documents = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        temp_pdf = f'temp_{uploaded_file.name}'
        with open(temp_pdf, 'wb') as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever()

    # QUESTION REWRITING PROMPT
    contextualize_q_system_prompt = (
        "Given previous chat history and the current user question, "
        "rephrase the question so it makes sense by itself."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    # ANSWERING PROMPT

    answer_q_system_prompt = (
        "You are an assistant for answering questions based on the provided context.\n"
        "Use ONLY the following context to answer:\n\n"
        "{context}\n\n"
        "If the answer is not in the context, say: I don't know."
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ('system', answer_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])


    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    def get_session_history(session_id: str):
        if session_id not in st.session_state['chat_history']:
            st.session_state['chat_history'][session_id] = InMemoryChatMessageHistory()
        return st.session_state['chat_history'][session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

    user_input = st.text_input('Ask a question from the PDFs')

    if user_input:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"session_id": session_id}
        )

        st.subheader("✅ Answer")
        st.write(response['answer'])

        st.subheader("📜 Chat History")
        st.write(get_session_history(session_id))

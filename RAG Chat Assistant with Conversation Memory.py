

import streamlit as st
from langchain_community.document_loaders import  YoutubeLoader, WebBaseLoader , PyPDFLoader
from PyPDF2 import PdfReader
import tempfile

import bs4 # filtering html tags
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

# Imports related to chat history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage  , HumanMessage , AIMessage

from langchain_community.document_loaders import PyPDFLoader
# Set up environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def format_message(messages):
    """Convert message objects to string format for the prompt"""
    formatted = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            formatted += f"Assistant: {msg.content}\n"
    return formatted

def chat(user_input ):
        # Append user message to chat history
        st.session_state.conversation_history.append(HumanMessage(content=user_input))


        # Prepare messages for the LLM
        messages = [system_message] + st.session_state.conversation_history


        # Get response from LLM
        response = llm.invoke(messages)

        # Append AI message to chat history
        st.session_state.conversation_history.append(AIMessage(content=response.content))

        return response.content

def updated_chat(user_input ):
    

    # Append user message to chat history
    st.session_state.conversation_history.append(HumanMessage(content=user_input))

    # Get response from LLM using rag_chain

    st.write(st.session_state.conversation_history)
    rag_chain = (
        {
            "context": retriever | RunnableLambda(lambda docs: "\n".join([doc.page_content for doc in docs])),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda x: format_message(st.session_state.conversation_history))
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(user_input)
    st.write(st.session_state.conversation_history)
   
    # Append AI message to chat history
    st.session_state.conversation_history.append(AIMessage(content=response))


# functions to use


    
st.title("Naive RAG Assignment Q2 with chat history")

# Upload PDF files
uploaded_files = PyPDFLoader('HistoryOfEncryption.pdf')
if uploaded_files:
    #1 - Load  
    documents = uploaded_files.load()

    # 2- Split
    # text_splitter -> Doer
    #split_docs ->  acted Upon
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    split_docs =text_splitter.split_documents(documents)

    # 3 - Embedding , 4- Store
    vectorstore = Chroma.from_documents(
        documents = split_docs,
        embedding = OpenAIEmbeddings(model = "text-embedding-3-large")
    )

    # 5 - Retrieve 
    retriever = vectorstore.as_retriever()

    # 6- Generation
    #llm = ChatOpenAI(model = "gpt-5-nano", temperature = 0)
    llm = ChatGroq(model= "llama-3.3-70b-versatile")

    # 7 - Prompt
    template = """Answer the question based only on the following context: {context}

        Chat History:
        {chat_history}

        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Define a system message (moved before rag_chain)
    system_message = SystemMessage(
        content="You are a helpful assistant. Keep responses short and concise."
    )
    
    
    
    
    
    
    for msg in st.session_state.conversation_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", key="user_input")
        submit_button = st.form_submit_button("Send")


    if user_input and submit_button:

        updated_chat(user_input)
   
else:
    st.info("Please upload at least one PDF document to proceed.")
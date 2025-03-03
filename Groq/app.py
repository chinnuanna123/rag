import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
#from langchain.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

## Load Groq API
groq_api_key=os.environ['GROQ_API_KEY']
if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/evaluation")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=220)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("ChatDemo")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="mixtral-8x7b-32768"
             )   

prompt=ChatPromptTemplate.from_template(
    """
Answer the question based on the context provided
<context>
{context}
questions:{input}
"""
)

documents_chain=create_stuff_documents_chain(llm,prompt)
retriever=st.session_state.vectors.as_retriever()
retrieval_chain=create_retrieval_chain(retriever, documents_chain)
prompt=st.text_input("enter your prompt here")

if prompt:
    response=retrieval_chain.invoke({"input":prompt})
    st.write(response['answer'])
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit App Title
st.title("🤖 RAG-based Q&A Chatbot")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Load Documents
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    
    # Text Splitting
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    
    # Create FAISS Vector Store
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context provided.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Retrieval Chain
retriever = st.session_state.vectors.as_retriever()
documents_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, documents_chain)

# User Input
user_query = st.text_input("💬 Ask a question:")

if user_query:
    response = retrieval_chain.invoke({"input": user_query})
    st.write("### 🤖 AI Answer:")
    st.write(response["answer"])

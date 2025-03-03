import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq AI Model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

st.title("🤖 GenAI Text Generator")

# User input
user_prompt = st.text_area("Enter your prompt:", "Write a poem about space")

if st.button("Generate"):
    response = llm.invoke(user_prompt)
    st.write(response.content)

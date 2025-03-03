#LangServe is a tool that makes it easy to deploy and serve LangChain models as APIs using FastAPI. It simplifies the process of exposing LLMs, RAG pipelines, and Agents as web services.

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_groq import ChatGroq
from langserve import add_routes

# Load API Key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize FastAPI App
app = FastAPI(title="LangServe API", version="1.0")

# Load LLM (Groq's Mixtral)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Add API Route for LLM
add_routes(app, llm, path="/chat")

# Run using:
# uvicorn langserve_api:app --host 0.0.0.0 --port 8000

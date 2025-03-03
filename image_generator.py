import streamlit as st
import openai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("🎨 AI Image Generator")

prompt = st.text_input("Enter an image description:", "A futuristic city at night")

if st.button("Generate Image"):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']
    st.image(image_url, caption="Generated Image")

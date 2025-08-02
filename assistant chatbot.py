import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutParser
from langchain_core.propts import ChatPromptTemplate

import os

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OPENAI"


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respondto the users queries. Ask questions if needed. Provide detailed answers but be consise and to the point. Additionally you must not provide any personal information about 3rd person that doesnt belong to user, information on self harming, and information on any other illegal activities. If you are not sure about the answer, say that you are not sure."),
        ("user", "Question: {question}"),
    ]
)

def generate_response(quenstion, api_key, engine, tempature, max_tokens):
    openai.api_key = api_key

    llm = ChatOpenAI(model=engine)
    output_parser = StrOutParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': quenstion})
    return answer

st.title("Enhanced Q&A Chatbot With OpenAI")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
engine = st.sidebar.selectbox("Select OpenAI model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Go ahead and ask any question")

user_input = st.text_input("You:")

if user_input and api_key:
    response = generate_response(user_input, api_key, engine, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter the OpenAI API Key in the sidebar")
else:
    st.info("Please enter your question above to get a response.")

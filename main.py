import streamlit as st
from streamlit_chat import message

import time
import os
import sys
import csv
import ast

from langchain.document_loaders import CSVLoader
from langchain.llms import OpenAI
from langchain.chains import  RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

import pinecone


st.set_page_config(page_title="NDIS Chatbot", 
                   page_icon=":pixel-art-neutral:",
                   layout="centered", 
                   initial_sidebar_state="auto")
col1, col2 = st.columns([1,1])


col1.markdown(" # NDIS Chatbot ")
col1.markdown(" ##### Ask me anything about NDIS ")
col2.markdown(" Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat ")

#configurations
csv.field_size_limit(sys.maxsize)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

def llm_chain(query):

    #loading the data
    loader = CSVLoader(file_path="content.csv", csv_args={"delimiter": "\n"})
    data = loader.load()

    #split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    #select which embeddings we want to use
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")

    # initialize pinecone
    pinecone.init(api_key=PINECONE_API_KEY,  environment=PINECONE_ENV)
    index_name = "ndis-chatbot"
    db = Pinecone.from_existing_index(index_name, embeddings)

    #expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template = """You are a conversational NDIS expert with access to NDIS context. 
    You are informative and provides details from the context. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    #create a chain to answer questions
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(ChatOpenAI(model_name="gpt-3.5-turbo"), 
                                    chain_type="stuff", retriever=retriever, 
                                    chain_type_kwargs=chain_type_kwargs)

    result = qa({"query": query})
    
    return ast.literal_eval(result["answer"])

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

placeholder = st.empty() 
def get_text():
    input_text = st.text_input("You: ", value="", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = llm_chain(user_input)
    with st.spinner('Wait for it...'):
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
        time.sleep(0.03)


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

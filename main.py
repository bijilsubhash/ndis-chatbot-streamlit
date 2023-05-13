import streamlit as st
from streamlit_chat import message

import time
import os
import sys
import csv

from langchain.document_loaders import CSVLoader
from langchain.llms import OpenAI
from langchain.chains import  ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
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
st.write("OPENAI_API_KEY:", st.secrets["OPENAI_API_KEY"])
st.write("PINECONE_API_KEY:", st.secrets["PINECONE_API_KEY"])
st.write("PINECONE_ENV:", st.secrets["PINECONE_ENV"])

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

    #create a chain to answer questions using conversational retrieval chain i.e. with memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever)

    chat_history = []
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    
    return result

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
        st.session_state.generated.append(output["answer"])
        time.sleep(2)


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

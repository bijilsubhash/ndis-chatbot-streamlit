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

import openai
import pinecone

#configurations
csv.field_size_limit(sys.maxsize)
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

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


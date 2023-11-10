import os

import splitter as splitter
import streamlit as lit
import pickle
import time
from langchain import OpenAI, text_splitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader


open_api_key = ''

# UI Components
lit.title("Parser Tool")
lit.sidebar.title("My Urls")
store_url = []
place_folder = lit.empty()
for i in range(2):
    store_url.append(lit.sidebar.text_input(f"URL {i + 1}"))

process_url = lit.sidebar.button("process")


if process_url:
    loader = UnstructuredURLLoader(urls= store_url)
    data = loader.load()

    RecursiveCharacterTextSplitter(

        separators=['\n\n', '\n', '.', ','],
        chunk_size=2000

    )
    place_folder.text("Text Splitter in prog")
    document = text_splitter.split_documents(data)
    embed = OpenAIEmbeddings()
    vect = FAISS.from_documents(document, embed)

    with open('', 'WB') as func:
        pickle.dump(vect, func)

place_folder.text_input("Question: ")
if query:
    if os.path.exists(''):
        with open('', 'rb') as func:
            vect = pickle.load(func)
            chain = RetrievalQAWithSourcesChain.from_llm(llm='llm', retriever=vect.as_retriever())
            res = chain({"question" : query}, return_only_outputs=True)
            lit.header('answer')
            lit.subheader(res['answer'])


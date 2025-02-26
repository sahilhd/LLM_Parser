# implemting benchmark to compare various llm performance on user requests
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
import os
import streamlit as st
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader

# Set your OpenAI API key here or use environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
VECTORSTORE_PATH = 'faiss_store.pkl'

st.set_page_config(page_title="Web Source Chatbot", layout="wide")
st.title("Web Source Chatbot")
st.write("""
This tool lets you add URLs, process their content, and chat with the information using OpenAI, LangChain, and FAISS.
""")

# Sidebar for URL input and processing
def url_input_section():
    st.sidebar.header("Add URLs to Ingest")
    url_list = st.sidebar.text_area(
        "Enter URLs (one per line):",
        height=150,
        placeholder="https://example.com\nhttps://another.com"
    )
    urls = [u.strip() for u in url_list.splitlines() if u.strip()]
    process_btn = st.sidebar.button("Process URLs and Build Knowledge Base")
    return urls, process_btn

# Load or create FAISS vectorstore
def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def save_vectorstore(vect):
    with open(VECTORSTORE_PATH, 'wb') as f:
        pickle.dump(vect, f)

# Main logic
urls, process_btn = url_input_section()
status_placeholder = st.sidebar.empty()

if process_btn:
    if not OPENAI_API_KEY:
        status_placeholder.error("OpenAI API key not set. Set the OPENAI_API_KEY environment variable.")
    elif not urls:
        status_placeholder.warning("Please enter at least one URL.")
    else:
        try:
            status_placeholder.info("Loading and processing URLs...")
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            status_placeholder.info("Splitting documents...")
            splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=2000
            )
            documents = splitter.split_documents(data)
            status_placeholder.info("Generating embeddings and building vectorstore...")
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vect = FAISS.from_documents(documents, embeddings)
            save_vectorstore(vect)
            status_placeholder.success(f"Knowledge base built from {len(urls)} URLs!")
        except Exception as e:
            status_placeholder.error(f"Error processing URLs: {e}")

# Chat interface
st.header("Ask Questions about Your URLs")
query = st.text_input("Enter your question:")
submit = st.button("Get Answer")

if submit and query:
    vect = load_vectorstore()
    if not vect:
        st.warning("No knowledge base found. Please add and process URLs first.")
    elif not OPENAI_API_KEY:
        st.error("OpenAI API key not set. Set the OPENAI_API_KEY environment variable.")
    else:
        try:
            llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=vect.as_retriever(),
                return_source_documents=True
            )
            with st.spinner("Getting answer..."):
                res = chain({"question": query}, return_only_outputs=True)
            st.subheader("Answer:")
            st.write(res.get('answer', 'No answer found.'))
            sources = res.get('sources', '')
            if sources:
                st.markdown(f"**Sources:** {sources}")
        except Exception as e:
            st.error(f"Error during retrieval: {e}")

st.markdown("---")
st.caption("Powered by OpenAI, LangChain, and FAISS. Built with Streamlit.")


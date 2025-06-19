import os
import streamlit as st
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader

# Set your OpenAI API key here or use environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
VECTORSTORE_PATH = 'faiss_store.pkl'

st.set_page_config(page_title="Web Source Chatbot", layout="wide")
st.title("Web Source Chatbot: Advanced RAG Demo")
st.write("""
This tool lets you add URLs and upload documents, process their content, and chat with the information using OpenAI, LangChain, and FAISS. Demonstrates advanced LLM and retrieval-augmented generation (RAG) techniques.
""")

# --- Sidebar: Model selection, session management, and ingestion ---
st.sidebar.header("Knowledge Base Setup")
model_name = st.sidebar.selectbox(
    "Choose OpenAI Model:",
    ["gpt-3.5-turbo", "gpt-4"],
    index=1
)

url_list = st.sidebar.text_area(
    "Enter URLs (one per line):",
    height=100,
    placeholder="https://example.com\nhttps://another.com"
)
urls = [u.strip() for u in url_list.splitlines() if u.strip()]

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF, TXT, DOCX):", accept_multiple_files=True, type=["pdf", "txt", "docx"]
)

col1, col2 = st.sidebar.columns(2)
process_btn = col1.button("Process Data")
clear_kb_btn = col2.button("Clear KB")

# --- Session state for chat and knowledge base ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore_exists' not in st.session_state:
    st.session_state.vectorstore_exists = os.path.exists(VECTORSTORE_PATH)

# --- Helper functions ---
def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def save_vectorstore(vect):
    with open(VECTORSTORE_PATH, 'wb') as f:
        pickle.dump(vect, f)
    st.session_state.vectorstore_exists = True

def clear_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        os.remove(VECTORSTORE_PATH)
    st.session_state.vectorstore_exists = False

def process_files(files):
    docs = []
    for file in files:
        loader = UnstructuredFileLoader(file)
        docs.extend(loader.load())
    return docs

# --- Knowledge base ingestion ---
status_placeholder = st.sidebar.empty()
if process_btn:
    if not OPENAI_API_KEY:
        status_placeholder.error("OpenAI API key not set. Set the OPENAI_API_KEY environment variable.")
    elif not (urls or uploaded_files):
        status_placeholder.warning("Please enter URLs or upload files.")
    else:
        try:
            status_placeholder.info("Loading and processing data...")
            data = []
            if urls:
                url_loader = UnstructuredURLLoader(urls=urls)
                data.extend(url_loader.load())
            if uploaded_files:
                data.extend(process_files(uploaded_files))
            status_placeholder.info(f"Splitting {len(data)} documents...")
            splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=2000
            )
            documents = splitter.split_documents(data)
            status_placeholder.info("Generating embeddings and building vectorstore...")
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vect = FAISS.from_documents(documents, embeddings)
            save_vectorstore(vect)
            status_placeholder.success(f"Knowledge base built from {len(urls)} URLs and {len(uploaded_files) if uploaded_files else 0} files!")
        except Exception as e:
            status_placeholder.error(f"Error processing data: {e}")

if clear_kb_btn:
    clear_vectorstore()
    status_placeholder.info("Knowledge base cleared.")

# --- Chat interface ---
st.header("Chat with Your Knowledge Base")
if st.button("Clear Chat History"):
    st.session_state.chat_history = []

if not st.session_state.vectorstore_exists:
    st.info("No knowledge base found. Please add and process URLs or files first.")
else:
    query = st.text_input("Your question:", key="user_query")
    submit = st.button("Send", key="send_btn")

    if submit and query:
        vect = load_vectorstore()
        if not vect:
            st.warning("No knowledge base found. Please add and process URLs or files first.")
        elif not OPENAI_API_KEY:
            st.error("OpenAI API key not set. Set the OPENAI_API_KEY environment variable.")
        else:
            try:
                llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name=model_name)
                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm,
                    retriever=vect.as_retriever(),
                    return_source_documents=True
                )
                with st.spinner("Getting answer..."):
                    res = chain({"question": query}, return_only_outputs=True)
                answer = res.get('answer', 'No answer found.')
                sources = res.get('sources', '')
                st.session_state.chat_history.append({
                    'question': query,
                    'answer': answer,
                    'sources': sources
                })
            except Exception as e:
                st.error(f"Error during retrieval: {e}")

    # --- Display chat history ---
    for i, turn in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q: {turn['question']}", expanded=(i==0)):
            st.markdown(f"**A:** {turn['answer']}")
            if turn['sources']:
                st.markdown(f"**Sources:** {turn['sources']}")

st.markdown("---")
st.caption("Powered by OpenAI, LangChain, and FAISS. Built with Streamlit. | Demo by Sahil Handa")


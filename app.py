import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

import streamlit as st
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# UI
st.set_page_config(page_title="Lab Manual AI", page_icon="🤖")
st.title("🤖 Lab Manual AI Assistant")

uploaded_file = st.file_uploader("Upload your Lab Manual (PDF)", type="pdf")

# If file uploaded
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File uploaded!")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Vector store
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    st.success("📚 PDF processed! Ask your question below 👇")

    # Question input
    query = st.text_input("Ask a question:")

    if query:
        # Search relevant docs
        docs = vectorstore.similarity_search(query)

        context = "\n".join([doc.page_content for doc in docs])

        # LLM (UPDATED MODEL ✅)
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="llama-3.1-8b-instant"
        )

        # Prompt
        prompt = f"""
        Answer the question based on the context below.

        Context:
        {context}

        Question:
        {query}
        """

        # Get answer
        response = llm.invoke(prompt)

        st.subheader("🤖 Answer:")
        st.write(response.content)

else:
    st.warning("Please upload a PDF to continue.")
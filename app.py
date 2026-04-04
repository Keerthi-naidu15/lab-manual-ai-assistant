import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

import streamlit as st
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# UI setup
st.set_page_config(page_title="Lab Manual AI", layout="centered")
st.title("📚 Lab Manual AI Assistant")
st.markdown("Upload a lab manual and ask questions or generate a summary.")

uploaded_file = st.file_uploader("Upload Lab Manual PDF", type="pdf")

# Stop if no file
if uploaded_file is None:
    st.warning("⚠️ Please upload a PDF to continue")
    st.stop()

# Save file
with open("temp.pdf", "wb") as f:
    f.write(uploaded_file.read())

st.success("✅ PDF Uploaded")

# -------------------------
# 🔄 PROCESS WITH PROGRESS
# -------------------------
progress = st.progress(0)
status = st.empty()

# Step 1: Load PDF
status.text("📄 Loading PDF...")
loader = PyPDFLoader("temp.pdf")
documents = loader.load()
progress.progress(20)

# Step 2: Split text
status.text("✂️ Splitting text...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = splitter.split_documents(documents[:10])  # limit for speed
progress.progress(40)

# Step 3: Embeddings
status.text("🧠 Creating embeddings...")
embeddings = HuggingFaceEmbeddings()
progress.progress(70)

# Step 4: Vector DB
status.text("📦 Building vector database...")
db = FAISS.from_documents(texts, embeddings)
progress.progress(100)

status.text("✅ Done!")
st.success("📚 PDF processed successfully!")

# -------------------------
# 🔧 TOOL SELECTION
# -------------------------
option = st.selectbox(
    "Choose an action",
    ["Ask Questions", "Summarize Document"]
)

# LLM setup
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=groq_api_key
)

# -------------------------
# 💬 Q&A FEATURE
# -------------------------
if option == "Ask Questions":
    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Thinking... 🤖"):
            docs = db.similarity_search(query)
            context = "\n".join([d.page_content for d in docs])

            prompt = f"""
            Answer based only on the context below.
            If not found, say "Not found in document".

            Context:
            {context}

            Question: {query}
            """

            response = llm.invoke(prompt)

            st.subheader("🤖 Answer")
            st.write(response.content)

# -------------------------
# 📄 SUMMARY FEATURE
# -------------------------
elif option == "Summarize Document":
    if st.button("Generate Summary"):
        with st.spinner("Summarizing... 🧠"):
            full_text = " ".join([doc.page_content for doc in documents[:10]])

            prompt = f"""
            Summarize this document in simple bullet points:

            {full_text}
            """

            response = llm.invoke(prompt)

            st.subheader("📄 Summary")
            st.write(response.content)
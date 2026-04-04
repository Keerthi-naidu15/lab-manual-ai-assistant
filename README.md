# 🤖 Lab Manual AI Assistant

An AI-powered chatbot that answers questions from lab manuals using **RAG (Retrieval-Augmented Generation)**.  
The system processes uploaded PDFs and provides accurate, context-based answers along with document summaries.

---

## 🚀 Features
- 📄 Upload PDF lab manuals  
- 💬 Ask questions from the document  
- 🤖 Get AI-generated, context-based answers  
- 📄 Generate document summaries  
- 📊 Progress bar for document processing  
- ⚡ Fast responses using Groq LLaMA models  

---

## 🛠 Tech Stack
- Python  
- Streamlit  
- LangChain  
- FAISS (Vector Database)  
- HuggingFace Embeddings  
- Groq API  

---

## 🧠 How It Works
1. Upload a lab manual PDF  
2. The system splits the document into chunks  
3. Converts text into embeddings using HuggingFace  
4. Stores them in FAISS for similarity search  
5. Retrieves relevant content and generates answers using Groq LLM  

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
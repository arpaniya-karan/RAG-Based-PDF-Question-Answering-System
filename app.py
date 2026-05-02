# ============================================================
# RAG-Based PDF Question Answering System
# Week 3 Assignment | Mini Project
# Tech Stack: Python, Streamlit, Gemini API, FAISS, LangChain
# ============================================================

import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS


# ============================================================
# Load Gemini API key from .env file
# ============================================================
load_dotenv()

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# Extract text from PDF
# ============================================================
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    return full_text


# ============================================================
# Split extracted text into chunks
# ============================================================
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# ============================================================
# Create embeddings and store them in FAISS vector database
# ============================================================
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


# ============================================================
# Retrieve relevant chunks from FAISS
# ============================================================
def retrieve_relevant_chunks(vector_store, question):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)
    return relevant_docs


# ============================================================
# Generate answer using Gemini based only on retrieved context
# ============================================================
def generate_answer(relevant_docs, question):
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the question using ONLY the context given below.
Do not use outside/general knowledge.

If the answer is not present in the context, respond exactly:
"I could not find this in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    response = llm.invoke(prompt)
    return response.content


# ============================================================
# Streamlit App
# ============================================================
def main():
    st.set_page_config(
        page_title="RAG PDF Q&A System",
        page_icon="📄",
        layout="centered"
    )

    st.title("📄 RAG-Based PDF Question Answering System")
    st.markdown("**Week 3 RAG Mini Project**")
    st.write("Upload a PDF and ask questions. The answer will be generated only from the uploaded document.")
    st.divider()

    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found. Please create a .env file and add your Gemini API key.")
        st.stop()

    with st.expander("How this RAG system works"):
        st.markdown("""
        1. User uploads a PDF.
        2. Text is extracted from the PDF using `pypdf`.
        3. Text is split into smaller chunks.
        4. Gemini Embeddings convert chunks into vectors.
        5. FAISS stores the vectors as a vector database.
        6. When the user asks a question, FAISS retrieves relevant chunks.
        7. Gemini generates the answer using only the retrieved chunks.
        """)

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            raw_text = extract_text_from_pdf(uploaded_file)

        if not raw_text.strip():
            st.error("No text could be extracted. This PDF may be scanned or image-based.")
            st.stop()

        st.success(f"Text extracted successfully! Characters found: {len(raw_text)}")

        with st.spinner("Splitting text into chunks..."):
            chunks = split_text_into_chunks(raw_text)

        st.info(f"Created {len(chunks)} text chunks.")

        with st.spinner("Creating embeddings and FAISS vector database..."):
            vector_store = create_vector_store(chunks)

        st.success("FAISS vector database is ready!")
        st.divider()

        question = st.text_input(
            "Ask a question from the PDF:",
            placeholder="Example: What is the main topic of this document?"
        )

        if st.button("Get Answer") and question:
            with st.spinner("Retrieving relevant chunks from FAISS..."):
                relevant_docs = retrieve_relevant_chunks(vector_store, question)

            with st.expander("View retrieved chunks"):
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"**Chunk {i + 1}:**")
                    st.write(doc.page_content[:500] + "...")

            with st.spinner("Generating answer using Gemini..."):
                answer = generate_answer(relevant_docs, question)

            st.subheader("Answer")
            st.write(answer)

            st.caption("This answer is generated only from the uploaded PDF content.")

    else:
        st.info("Please upload a PDF to start.")


if __name__ == "__main__":
    main()

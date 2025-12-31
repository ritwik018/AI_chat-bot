import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(page_title="AI Knowledge Chatbot")
st.title("📚 AI Knowledge Base Chatbot (Local Embeddings)")

uploaded_file = st.file_uploader("Upload company document (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # ---------------- Local embeddings ----------------
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Embed chunks
        embeddings = [model.encode(doc.page_content) for doc in chunks]

        # Create FAISS index
        import faiss
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))

        # Save chunks and index in session state
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.success("✅ Document indexed successfully!")

# ---------------- Chat ----------------
if "index" in st.session_state:
    question = st.text_input("Ask a question from the document")
    if question:
        # Embed question
        model = SentenceTransformer('all-MiniLM-L6-v2')
        q_vec = model.encode(question)

        # Search FAISS
        D, I = st.session_state.index.search(np.array([q_vec]), k=4)
        docs = [st.session_state.chunks[i] for i in I[0]]

        # Build context
        context = "\n\n".join([doc.page_content for doc in docs])

        # LLM (OpenAI still used for generation)
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        prompt = f"""
Answer the question ONLY using the context below.
If not found, say "Not found in document".

Context:
{context}

Question:
{question}
"""

        answer = llm.invoke(prompt)
        st.write("### ✅ Answer")
        st.write(answer.content)

        st.write("### 📌 Sources")
        for doc in docs:
            st.write(doc.metadata)

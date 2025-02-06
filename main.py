import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

 

# Load Documents (Supports PDF, TXT, DOCX)
def load_documents(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    else:
        raise ValueError("Unsupported file format. Use PDF, TXT, or DOCX.")
    return documents

# Split Documents into Chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Create FAISS Vector Store with DeepSeek Embeddings
def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return vector_store

# Initialize DeepSeek Model
def initialize_llm():
    return OllamaLLM(model="deepseek-r1:1.5b")

# Retrieve Context and Generate Answer
def ask_question(vector_store, llm, query):
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = qa_chain.invoke({"query": query})
    return response["result"]

#Run RAG Pipeline
if __name__ == "__main__":
    file_path = r"C:\Deepseek_R1\pdfs\Akash Thorat_resume.pdf"  
    query = "What is this document about?"

    documents = load_documents(file_path)
    chunks = split_documents(documents)
    
    vector_store = create_vector_store(chunks)

    llm = initialize_llm()
    answer = ask_question(vector_store, llm, query)
    
    print("Answer:", answer)

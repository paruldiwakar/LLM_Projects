import os
import numpy as np
import pickle as pkl
import streamlit as st
import time
import subprocess

from keys import token

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.llms import HuggingFaceEndpoint

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain

from dotenv import load_dotenv

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
subprocess.run(["huggingface-cli", "login", "--token", token])

# Streamlit app setup
st.title("SkyBot🚀: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_button = st.sidebar.button("Process URLs")

file_path = "vectorIndex_faiss.pkl"

main_placeholder = st.empty()

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
token = "YOUR_HUGGINGFACE_TOKEN"  # Make sure to replace this with your actual token

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.2, token=token
)

# Initialize session state for URLs processed
if 'urls_processed' not in st.session_state:
    st.session_state['urls_processed'] = False

if process_url_button:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save it to FAISS index
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorIndex = FAISS.from_documents(docs, embedding_model)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pkl.dump(vectorIndex, f)

    main_placeholder.text("Data Processing Completed...✅✅✅")
    st.session_state['urls_processed'] = True

if st.session_state['urls_processed']:
    # Show query input field after processing URLs
    query = st.text_input("Question: ")
    process_query_button = st.button("Process Query")

    if process_query_button:
        main_placeholder.text("Processing Query...Started...✅✅✅")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorIndex = pkl.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        main_placeholder.text("Generating Answer...Started...✅✅✅")
        
        # Result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)


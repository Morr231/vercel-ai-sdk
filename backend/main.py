from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from fastapi import FastAPI, Request
import dotenv
dotenv.load_dotenv()

import os
import sys
import time
app = FastAPI()


def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The function {func.__name__} took {execution_time:.2f} seconds to complete.")
        return result

    return wrapper


openai_api_key = os.environ.get('OPENAI_API_KEY')

def extract_chunks_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def extract_chunks_from_txt(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_chunks_to_doc(folder_name):
    documents = []
    
    #                      /docs
    for file in os.listdir(folder_name):
        file_path = f"{folder_name}/{file}"

        if file.endswith('.pdf'):
            documents.extend(extract_chunks_from_pdf(file_path))
        elif file.endswith('.txt'):
            documents.extend(extract_chunks_from_txt(file_path))

    return documents


@track_time
def split_chunks_into_documents():
    documents = load_chunks_to_doc(f"docs")
    
    chunk_size = 1000
    chunk_overlap = 200

    # Recursive -> "\n\n" -> "\n" -> " "
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents


def generate_index_from_documents():
    persist_directory = f'db'

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        docs = split_chunks_into_documents()
        db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        db.persist()
        return db


@track_time
def generate_context_from_documents(query):
    db = generate_index_from_documents()
    docs = db.similarity_search(query)

    context = [i.page_content for i in docs]
    return context

@app.get("/context")
async def get_context(search: str):
    context = generate_context_from_documents(search)
    return {"context": context}
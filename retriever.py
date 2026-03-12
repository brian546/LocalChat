import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_community.embeddings import Model2vecEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from custom_bm25 import CustomBM25Retriever as BM25Retriever
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_sparse():
    collection_split = pd.read_json(
        "./data/collection.jsonl", lines=True
    ).drop_duplicates(subset=["text"], keep="first")
    collection_loader = DataFrameLoader(collection_split, page_content_column="text")
    collection_docs = collection_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(collection_docs)

    bm25_retriever = BM25Retriever.from_documents(documents, k=5)
    return bm25_retriever


def load_vector_store(
    embeddings: str,
) -> Chroma | BM25Retriever:
    """
    embeddings: supports following embeddings to query from: static, dense, minilm
    """
    embedding_options = {
        "static": lambda: Model2vecEmbeddings(model="minishlab/potion-base-8M"),
        "dense": lambda: HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5"),
        "sparse": load_sparse(),
        "qwen": lambda: HuggingFaceEmbeddings(model="Qwen/Qwen3-Embedding-0.6B"),
        # "colbert": lambda: HuggingFaceEmbeddings(model="colbert-ir/colbertv2.0"),
    }

    if embeddings in embedding_options:
        embedding_function = embedding_options[embeddings]()
        if embeddings == "sparse":
            return embedding_function
        # Load and chunk documents for BM25
        if os.path.exists("./data/chromadb"):
            return Chroma(
                persist_directory="./data/chromadb",
                embedding_function=embedding_function,
                collection_name=f"{embeddings}_collection",
            )
        else:
            raise FileNotFoundError(
                "Directory does not exist. Run dataloader.py to generate chromadb embeddings with persistence."
            )
    else:
        raise Exception("Provide embedding options that are acceptable")

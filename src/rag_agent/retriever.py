import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_community.embeddings import Model2vecEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import load_agent_config, resolve_project_path


CONFIG = load_agent_config()
DATA_CONFIG = CONFIG["data"]
RAG_CONFIG = CONFIG["rag"]
RETRIEVER_CONFIG = RAG_CONFIG["retriever"]
VECTOR_STORE_CONFIG = RAG_CONFIG["vector_store"]
COLLECTION_PATH = resolve_project_path(DATA_CONFIG["collection_path"])
PERSIST_DIRECTORY = resolve_project_path(VECTOR_STORE_CONFIG["persistence_dir"])
EMBEDDING_MODELS = RETRIEVER_CONFIG["embedding"]
SPARSE_TOP_K = RETRIEVER_CONFIG.get("sparse_top_k", RETRIEVER_CONFIG["top_k"])


def load_sparse():
    try:
        from .custom_bm25 import CustomBM25Retriever as BM25Retriever
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Sparse retrieval requires the 'rank-bm25' package. Install it to use embed='sparse'."
        ) from exc

    collection_split = pd.read_json(COLLECTION_PATH, lines=True).drop_duplicates(
        subset=["text"], keep="first"
    )
    collection_loader = DataFrameLoader(collection_split, page_content_column="text")
    collection_docs = collection_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(collection_docs)

    bm25_retriever = BM25Retriever.from_documents(documents, k=SPARSE_TOP_K)
    return bm25_retriever


def load_vector_store(
    embeddings: str,
) -> Chroma | object:
    """
    embeddings: supports following embeddings to query from: static, dense, minilm
    """
    embedding_options = {
        "static": lambda: Model2vecEmbeddings(model=EMBEDDING_MODELS["static"]),
        "dense": lambda: HuggingFaceEmbeddings(model=EMBEDDING_MODELS["dense"]),
        "sparse": load_sparse,
        "qwen": lambda: HuggingFaceEmbeddings(model=EMBEDDING_MODELS["qwen"]),
        # "colbert": lambda: HuggingFaceEmbeddings(model="colbert-ir/colbertv2.0"),
    }

    if embeddings in embedding_options:
        embedding_function = embedding_options[embeddings]()
        if embeddings == "sparse":
            return embedding_function
        # Load and chunk documents for BM25
        if os.path.exists(PERSIST_DIRECTORY):
            return Chroma(
                persist_directory=str(PERSIST_DIRECTORY),
                embedding_function=embedding_function,
                collection_name=f"{embeddings}_collection",
            )
        else:
            raise FileNotFoundError(
                "Directory does not exist. Run dataloader.py to generate chromadb embeddings with persistence."
            )
    else:
        raise Exception("Provide embedding options that are acceptable")

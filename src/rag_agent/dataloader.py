import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import Model2vecEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import time

embedding_options = {
    "static": lambda: Model2vecEmbeddings(model="minishlab/potion-base-8M"),
    "dense": lambda: HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5"),
    "qwen": lambda: HuggingFaceEmbeddings(model="Qwen/Qwen3-Embedding-0.6B"),
    # "colbert": lambda: HuggingFaceEmbeddings(model="colbert-ir/colbertv2.0"),
}

print("Loading dataset")
collection_split = pd.read_json("./data/collection.jsonl", lines=True)

unique_collection_split = collection_split.copy().drop_duplicates(
    subset=["text"], keep="first"
)

collection_loader = DataFrameLoader(unique_collection_split, page_content_column="text")
collection_docs = collection_loader.load()

# splitting documents into chucks of size of 1000 tokens and overlap of 100 tokens
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
print("Splitting loaded documents!")
collection_text_splitter = text_splitter.split_documents(collection_docs)

print("Generating vector embeddings!")
for each in embedding_options.keys():
    print(f"Generating vector embeddings for {each}_collections")
    print(time.ctime())
    embedding_function = embedding_options[each]()
    Chroma.from_documents(
        documents=collection_text_splitter,
        embedding=embedding_function,
        persist_directory="./data/chromadb",
        collection_name=f"{each}_collection",
    )

print(time.ctime())

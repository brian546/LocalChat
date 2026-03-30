# File: ragtype.py
# Description: This module defines TypedDicts for messages and RAG state management.

from typing import TypedDict


class Message(TypedDict):
    role: str  
    content: str

class RAGState(TypedDict):
    question: str
    history: list[Message]
    retrieved_docs: list[str]
    retrieved_id_scores: list[list[str, float]]
    count_retrieval: int = 0
    answer: str = ""
    thought: str = ""
    rewritten_query: str
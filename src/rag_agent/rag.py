# File: rag.py
# Description: This script provides modules to load LangGraph RAG Workflow with ChromaDB vector store and Ollama Qwen2.5 model

from .ragtype import Message, RAGState
from .config import load_agent_config

# chat model
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

from langgraph.checkpoint.memory import MemorySaver

from .retriever import load_vector_store

# %%
#===============================
#           Config
#===============================

CONFIG = load_agent_config()
LLM_CONFIG = CONFIG["llm"]
RAG_CONFIG = CONFIG["rag"]
MODEL_NAME = LLM_CONFIG["name"]
MODEL_PROVIDER = LLM_CONFIG.get("provider", "ollama")
MODEL_TEMPERATURE = LLM_CONFIG.get("temperature", 0)
DOC_NUM = RAG_CONFIG["retriever"]["top_k"]
MAX_MEMORY_SIZE = LLM_CONFIG["history_length"]


# %%
def load_graph(embed: str = "static", chain_of_thought: bool = True) -> StateGraph:
    """Load the RAG agent.
    
    Parameters
    ==========
    embed: str
        Embedding type to use: static, dense, sparse, qwen
        
    chain_of_thought: bool
        Whether to use chain of thought reasoning

    Returns
    =======
    StateGraph  
    
    """
    vector_store = load_vector_store(embed)

    response_model = init_chat_model(
        model=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        temperature=MODEL_TEMPERATURE,
    )

    def rewrite_query(state: RAGState) -> str:
        """
            Rewrite a user query to improve retrieval quality.

            Returns:
            rewritten_query: The rewritten query optimized for retrieval
        """
        question = state["question"]
        history = state.get("history", [])
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history]) if history else "None"

        rewrite_prompt = f"""Rewrite the following query to improve retrieval. If there are previsou conversations, please reference them.
        Original query: {question}
        Previous conversations: {history}
        Directly return the rewritten query, no other text."""

        response = response_model.invoke(
            [{"role": "user", "content": rewrite_prompt}])

        return {"rewritten_query": response.content}

    def search_documents(state: RAGState):
        """Search documents."""
        query = state["rewritten_query"]
        if embed == "sparse":
            retrieved_docs = vector_store.invoke(query) 
            context = [doc.page_content for doc in retrieved_docs]
            doc_id_scores = [[doc.metadata['id'],doc.metadata['score']] for doc in retrieved_docs]
        else:
            retrieved_docs = vector_store.similarity_search_with_relevance_scores(query, k=DOC_NUM)
            context = [doc.page_content for doc, _ in retrieved_docs]
            doc_id_scores = [[doc.metadata['id'],score] for doc, score in retrieved_docs]

        all_retrieved_docs = state.get("retrieved_docs", []) + context
        all_retrieved_docs = list(set(all_retrieved_docs))
        return {"retrieved_docs": all_retrieved_docs, "retrieved_id_scores": doc_id_scores}


    def chain_of_thought(state: RAGState):
        """Think step by step using chain of thought prompting."""


        GENERATE_PROMPT = (
            "You are an assistant for question-answering tasks. "
            "Not all context and previsou concversation are related to the question. You need to only use the relevant parts to think step by step to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            """ Think step by step using the following structure:
            1. **Understand**: Restate the question in your own words and identify key elements.
            2. **Gather**: List all relevant facts, assumptions, or data from the context.
            3. **Break Down**: Divide the problem into smaller, manageable sub-problems or steps.
            4. **Solve/Analyze**: Address each sub-step logically. Show calculations, reasoning justifications, or trade-offs.
            5. **Verify**: Check for errors, inconsistencies, or alternative perspectives.
            6. **Conclude**: Summarize the final answer or recommendation clearly.\n\n"""
            "Question: {question} \n"
            "Previous Conversations: {history}\n"
            "Context: {context}"
        )

        question = state["question"]
        context = "\n\n".join(state["retrieved_docs"])
        history = state.get("history",[])
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history]) if history else "None"
        prompt = GENERATE_PROMPT.format(question=question, context=context, history=history)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"thought": response.content}


    def generate_answer(state: RAGState):
        """Generate the final answer"""

        GENERATE_PROMPT = (
            "You are an assistant for question-answering tasks. "
            "Based on the conclusion of chain of thoughts to answer the question. "
            "Beginning with 'Answer:' to answer the question"
            "Keep the answer concise and simple.\n"

            "Question: {question} \n"
            "Chain of Thought: {thought}\n"
        )


        question = state["question"]
        thought = state["thought"]

        prompt = GENERATE_PROMPT.format(question=question,thought=thought)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        response = response.content.split("Answer:")[-1].strip()


        # update history
        history = state.get("history", [])
        history.append(Message(role="user", content=question))
        history.append(Message(role="assistant", content=response))

        # remove old history
        if len(history) > MAX_MEMORY_SIZE:
            history = history[MAX_MEMORY_SIZE:]

        # reply the question, 
        # save the conversation history 
        # and clean the retrived docs for question in next round
        return {"answer": response, "history": history, "retrieved_docs": [], "retrieved_id_scores": [] }
    
    def generate_answer_from_context(state: RAGState):
        """Generate the final answer"""

        GENERATE_PROMPT = (
            "You are an assistant for question-answering tasks. "
            "Based on the context and previous conversations if available to answer the question. "
            "Beginning with 'Answer:' to answer the question"
            "Keep the answer concise and simple.\n"

            "Question: {question} \n"
            "Previous Conversations: {history}\n"
            "Context: {context}"
            
            )


        question = state["question"]
        context = "\n\n".join(state["retrieved_docs"])
        history = state.get("history",[])
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history]) if history else "None"

        prompt = GENERATE_PROMPT.format(question=question,context=context, history=history)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        response = response.content.split("Answer:")[-1].strip()


        # update history
        history = state.get("history", [])
        history.append(Message(role="user", content=question))
        history.append(Message(role="assistant", content=response))

        # remove old history
        if len(history) > MAX_MEMORY_SIZE:
            history = history[MAX_MEMORY_SIZE:]

        # reply the question, 
        # save the conversation history 
        # and clean the retrived docs for question in next round
        return {"answer": response, "history": history, "retrieved_docs": [], "retrieved_id_scores": [] }


    workflow = StateGraph(RAGState)
    workflow.add_node(rewrite_query)
    workflow.add_node(search_documents)
    if chain_of_thought:
        workflow.add_node(chain_of_thought)
        workflow.add_node(generate_answer)
    else:
        workflow.add_node(generate_answer_from_context)

    workflow.add_edge(START, "rewrite_query")
    workflow.add_edge("rewrite_query", "search_documents")
    if chain_of_thought:
        workflow.add_edge("search_documents", "chain_of_thought")
        workflow.add_edge("chain_of_thought", "generate_answer")
        workflow.add_edge("generate_answer", END)
    else:
        workflow.add_edge("search_documents", "generate_answer_from_context")
        workflow.add_edge("generate_answer_from_context", END)

    # memory checkpointer for mult-turn conversation
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph
    # import IPython
    # IPython.display.Image(graph.get_graph().draw_mermaid_png())

# %%

# # fix thread id for mult-turn conversation
# config = {"configurable": {"thread_id": str(uuid.uuid4())}}
# questions = [
#     "What island is included in the geographic area home to a variety of publications describing Hong Kong Cantonese as \"Hong Kong speech\"?",
#     "where is the island",
#     "what country does it belong to?",
# ]
# for q in questions:
#     for chunk in graph.stream({"question": q}, config):
#         for node, update in chunk.items():
#             print(node)
#             print(update)
#             print("-" * 10 + f" {node} " + "-" * 10)
#             if node == "search_documents":
#                 print(f"Retrieved {len(update['retrieved_docs'])} docs")
#             else:
#                 print(update)
#             print("\n")

# %%




# file: batch_generate.py
# Description: This script runs batch generation of answers for questions in a data file using a RAG workflow.

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.config import load_agent_config, resolve_project_path
from rag_agent.rag import load_graph
import uuid
import json
from tqdm import tqdm
import os
import argparse
# ===================
#      Config
# ===================
CONFIG = load_agent_config()
BATCH_CONFIG = CONFIG["batch"]
DATA_FILE = BATCH_CONFIG["data_file"]
RESULT_DIR = BATCH_CONFIG["result_dir"]
EMBEDDING_TYPE = BATCH_CONFIG["embedding_type"]

def run_in_batch(data_file: str = DATA_FILE ,embed_type: str = EMBEDDING_TYPE, chain_of_thought: bool = True):
    """
    Run batch generation for questions in DATA_FILE and save results to OUTPUT_FILE.

    Parameters
    ==========
    data_file: str
        Path to the input data file containing questions in JSONL format.
    embed_type: str
        Embedding type to use: static, dense, sparse, qwen
    chain_of_thought: bool
        Whether to use chain of thought reasoning.
    """

    data_file = resolve_project_path(data_file)
    result_dir = resolve_project_path(RESULT_DIR)
    output_file = result_dir / f"{data_file.stem}_{embed_type}.jsonl"

    # create result directory if not exists
    os.makedirs(result_dir, exist_ok=True)

    #
    with open(data_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]


    graph = load_graph(embed=embed_type, chain_of_thought=chain_of_thought)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Start responding to questions...")

    # remove existing output file
    if os.path.exists(output_file):
        os.remove(output_file)

    for d in tqdm(data):
        q = d["text"]
        for chunk in graph.stream({"question": q}, config):
            for node, update in chunk.items():
                if node == "search_documents":
                    id_scrores = update["retrieved_id_scores"]
                if node == "generate_answer_from_context" or node == "generate_answer":
                    answer = update["answer"]
        
        mode = "a" if os.path.exists(output_file) else "w"

        data = {
            "id": d["id"],
            "text": q,
            "answer": answer,
            "retrieved_docs": id_scrores,
        }

        with open(output_file, mode, encoding="utf-8") as f:
            json.dump(data, f)
            f.write("\n")

    print('Answers are saved to {}'.format(output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch generate answers for questions.')
    parser.add_argument('-f','--file', type=str, default=DATA_FILE, help='Path to the input data file.')
    parser.add_argument('-e','--embed', type=str, default=EMBEDDING_TYPE, help='Embedding type: static, dense, sparse, qwen, colbert.')
    parser.add_argument('-s','--skip_chain', action='store_true', help='Skip Chain of thought')

    args = parser.parse_args()
    data_file = args.file
    embed = args.embed
    chain_of_thought = not args.skip_chain
    run_in_batch(data_file=data_file, embed_type=embed, chain_of_thought=chain_of_thought)



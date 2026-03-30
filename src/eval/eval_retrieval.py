import argparse
import json
import time
from pathlib import Path

import pytrec_eval
import pandas as pd
import yaml


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "eval.yaml"
PROJECT_ROOT = CONFIG_PATH.parent.parent


def load_eval_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_jsonl(file_path):
    data = list()
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(f'[{time.asctime()}] Read {len(data)} from {file_path}')
    return data


def compute_metrics(qrels: dict[str, dict[str, int]], results: dict[str, dict[str, float]], k_values: list[int]):
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    # qrels = {str(qid): {str(docid): s for docid, s in v.items()} for qid, v in qrels.items()}
    # results = {str(qid): {str(docid): s for s, docid in v} for qid, v in result_heaps.items()}
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores_by_query = evaluator.evaluate(results)
    scores = pd.DataFrame.from_dict(scores_by_query.values()).mean()
    metrics = dict()  # TODO
    for prefix in ('map_cut', 'ndcg_cut', 'recall', 'P'):
        name = 'precision' if prefix == 'P' else prefix.split('_')[0]
        for k in k_values:
            metrics[f'{name}_at_{k}'] = scores[f'{prefix}_{k}']
    return metrics


def main():
    cfg = load_eval_config()
    defaults_cfg = cfg.get('defaults', {})
    retrieval_cfg = cfg.get('retrieval', {})
    default_gold = defaults_cfg.get('gold')
    default_pred = defaults_cfg.get('pred')
    default_k_values = retrieval_cfg.get('k_values', [2, 5, 10])

    parser = argparse.ArgumentParser(description='Evaluate retrieval results.')
    parser.add_argument('--gold', '-g', type=str, default=default_gold, help='Path to the gold file.')
    parser.add_argument('--pred', '-p', type=str, default=default_pred, help='Path to the predicted file.')
    parser.add_argument('--k_values', '-k', type=int, nargs='+', default=default_k_values, help='Cutoff values for MAP/NDCG/Recall/Precision.')
    args = parser.parse_args()

    if not args.gold or not args.pred:
        parser.error('Missing required input files. Provide --gold and --pred, or set defaults in configs/eval.yaml.')

    gold_data = read_jsonl(str(resolve_project_path(args.gold)))
    pred_data = read_jsonl(str(resolve_project_path(args.pred)))
    assert len(gold_data) == len(pred_data), "Gold and pred files must have the same number of entries."

    qrels = {i['id']: {d: 1 for d in i['supporting_ids']} for i in gold_data}
    results = {i['id']: {d: s for d, s in i['retrieved_docs']} for i in pred_data}
    metrics = compute_metrics(qrels, results, args.k_values)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    main()
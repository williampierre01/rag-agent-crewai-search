"""
extraction_eval.py — Mede a acurácia CAMPO A CAMPO da extração estruturada,
comparando contra um gabarito (ground truth).

Diferença em relação ao rag_eval.py:
- rag_eval.py mede se a RECUPERAÇÃO achou o trecho certo do documento.
- extraction_eval.py mede se a EXTRAÇÃO final (JSON) tem os valores certos.
Juntos cobrem as duas partes do pipeline: "achei o contexto certo?" e
"extraí os dados certos a partir dele?".

Como a extração roda o LLM (Qwen2.5-7B, precisa de GPU), este eval chama o
Space JÁ PUBLICADO via API, em vez de rodar local — mesma lógica do
test_deploy.py.

Uso:
    python extraction_eval.py --space seu-usuario/nome-do-space \\
        --pdf knowledge/laudo_BM-102_2026-07-12.pdf \\
        --ground-truth laudo_ground_truth.json
"""
import argparse
import json
import os
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List

from gradio_client import Client, handle_file

# Campos categóricos/curtos: comparados com match "quase exato" (normalizado,
# tolerante a espaço/maiúscula, mas não a parafraseamento).
EXACT_FIELDS = [
    "numero_laudo", "tag_equipamento", "criticidade",
    "data_inspecao", "prazo_correcao", "proxima_inspecao",
]

# Campos de texto livre: comparados por similaridade (o modelo pode
# reformular sem estar "errado").
FUZZY_FIELDS = [
    "equipamento", "local_instalacao", "inspetor_responsavel",
    "tipo_inspecao", "status_equipamento", "recomendacoes",
]
FUZZY_THRESHOLD = 0.6  # similaridade mínima para contar como "correto"

# Campos de lista: cada item do gabarito é casado contra o item mais
# parecido na predição (não exige ordem nem literalidade).
LIST_FIELDS = ["normas_referenciadas", "nao_conformidades"]
LIST_ITEM_THRESHOLD = 0.5


def normalize(s: Any) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def eval_exact_field(gold: Any, pred: Any) -> Dict:
    match = normalize(gold) == normalize(pred)
    return {"correct": match, "gold": gold, "pred": pred}


def eval_fuzzy_field(gold: Any, pred: Any) -> Dict:
    sim = similarity(gold, pred)
    return {"correct": sim >= FUZZY_THRESHOLD, "similarity": round(sim, 3), "gold": gold, "pred": pred}


def eval_list_field(gold: List[str], pred: Any) -> Dict:
    if not isinstance(pred, list):
        pred = []
    gold = gold or []

    matched_gold = 0
    for g_item in gold:
        best = max((similarity(g_item, p_item) for p_item in pred), default=0.0)
        if best >= LIST_ITEM_THRESHOLD:
            matched_gold += 1

    matched_pred = 0
    for p_item in pred:
        best = max((similarity(p_item, g_item) for g_item in gold), default=0.0)
        if best >= LIST_ITEM_THRESHOLD:
            matched_pred += 1

    recall = matched_gold / len(gold) if gold else None
    precision = matched_pred / len(pred) if pred else None

    return {
        "recall": round(recall, 3) if recall is not None else None,
        "precision": round(precision, 3) if precision is not None else None,
        "gold": gold,
        "pred": pred,
    }


def evaluate_extraction(gold: Dict, pred: Dict) -> Dict:
    field_results = {}

    for field in EXACT_FIELDS:
        field_results[field] = {"type": "exact", **eval_exact_field(gold.get(field), pred.get(field))}

    for field in FUZZY_FIELDS:
        field_results[field] = {"type": "fuzzy", **eval_fuzzy_field(gold.get(field), pred.get(field))}

    for field in LIST_FIELDS:
        field_results[field] = {"type": "list", **eval_list_field(gold.get(field), pred.get(field))}

    # Agregados
    exact_scores = [field_results[f]["correct"] for f in EXACT_FIELDS]
    fuzzy_scores = [field_results[f]["correct"] for f in FUZZY_FIELDS]
    list_recalls = [field_results[f]["recall"] for f in LIST_FIELDS if field_results[f]["recall"] is not None]

    exact_accuracy = sum(exact_scores) / len(exact_scores) if exact_scores else 0.0
    fuzzy_accuracy = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0.0
    avg_list_recall = sum(list_recalls) / len(list_recalls) if list_recalls else 0.0

    overall = (
        (sum(exact_scores) + sum(fuzzy_scores) + sum(list_recalls))
        / (len(exact_scores) + len(fuzzy_scores) + len(list_recalls))
    )

    summary = {
        "exact_field_accuracy": round(exact_accuracy, 4),
        "fuzzy_field_accuracy": round(fuzzy_accuracy, 4),
        "avg_list_recall": round(avg_list_recall, 4),
        "overall_score": round(overall, 4),
    }

    return {"summary": summary, "fields": field_results}


def print_report(report: Dict):
    s = report["summary"]
    print(f"\n{'=' * 55}")
    print("  RESUMO — EXTRAÇÃO ESTRUTURADA")
    print(f"{'=' * 55}")
    print(f"exact_field_accuracy:  {s['exact_field_accuracy']}   (campos curtos/categóricos)")
    print(f"fuzzy_field_accuracy:  {s['fuzzy_field_accuracy']}   (campos de texto livre, similaridade >= {FUZZY_THRESHOLD})")
    print(f"avg_list_recall:       {s['avg_list_recall']}   (normas/não conformidades encontradas)")
    print(f"overall_score:         {s['overall_score']}")

    print(f"\n{'-' * 55}")
    print("  DETALHE POR CAMPO")
    print(f"{'-' * 55}")
    for field, r in report["fields"].items():
        if r["type"] in ("exact", "fuzzy"):
            status = "OK " if r["correct"] else "ERRO"
            extra = f" sim={r['similarity']}" if "similarity" in r else ""
            print(f"[{status}] {field}{extra}")
            if not r["correct"]:
                print(f"       gold: {r['gold']!r}")
                print(f"       pred: {r['pred']!r}")
        else:  # list
            print(f"[list] {field}  recall={r['recall']}  precision={r['precision']}")
            print(f"       gold: {r['gold']}")
            print(f"       pred: {r['pred']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True)
    parser.add_argument("--pdf", default="knowledge/laudo_BM-102_2026-07-12.pdf")
    parser.add_argument("--ground-truth", default="laudo_ground_truth.json")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--out", default="extraction_eval_results.json")
    args = parser.parse_args()

    if not args.hf_token:
        print("Aviso: sem HF_TOKEN — cota anônima do ZeroGPU pode interromper o teste.")

    with open(args.ground_truth, "r", encoding="utf-8") as f:
        gold = json.load(f)

    print(f"Conectando ao Space {args.space}...")
    client = Client(args.space, token=args.hf_token)

    print(f"Enviando {args.pdf}...")
    client.predict(handle_file(args.pdf), api_name="/upload_pdf")
    time.sleep(2)  # dá um respiro para a indexação (CPU) terminar

    print("Rodando extração estruturada...")
    t0 = time.time()
    pred = client.predict(api_name="/extract")
    latency = time.time() - t0
    print(f"Extração concluída em {latency:.1f}s")

    if isinstance(pred, dict) and "error" in pred and "raw_output" in pred:
        print("\nA extração falhou em gerar JSON válido:")
        print(pred["raw_output"][:500])
        return

    report = evaluate_extraction(gold, pred)
    report["latency_s"] = round(latency, 2)
    print_report(report)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nDetalhes salvos em: {args.out}")


if __name__ == "__main__":
    main()
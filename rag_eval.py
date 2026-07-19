rag eval · PY
"""
rag_eval.py — Avalia a qualidade da RECUPERAÇÃO de contexto do RAG,
isolada da geração de texto do LLM.
 
Por quê separar retrieval de generation:
- Testar sem precisar de GPU/cota do ZeroGPU nem do LLM de 7B — embeddings
  e BM25 rodam em CPU, então o eval é rápido e barato de rodar sempre que
  você mexer em chunking, pesos do ensemble, ou k.
- Mede exatamente o que está sendo otimizado: "achei os trechos certos do
  documento?" — separado de "o LLM formulou uma boa resposta com eles?".
  Um LLM bom pode mascarar recuperação ruim (ele "adivinha" a partir de
  conhecimento próprio); medir retrieval isoladamente evita esse viés.
 
Uso básico:
    python rag_eval.py --pdf caminho/documento.pdf --dataset eval_dataset.json
 
Compara vetorial-puro vs híbrido (BM25+vetorial) lado a lado:
    python rag_eval.py --pdf caminho/documento.pdf --dataset eval_dataset.json --compare
 
Formato do dataset (eval_dataset.json):
[
  {
    "question": "Qual o prazo de garantia do produto?",
    "expected_keywords": ["12 meses", "garantia"]
  },
  ...
]
"expected_keywords": termos que DEVEM aparecer em algum chunk recuperado
para considerar a pergunta "respondível" com o contexto retornado. Não
precisa ser a resposta inteira — só o suficiente para confirmar que o
trecho certo do documento foi encontrado.
"""
import argparse
import json
import time
from typing import List, Dict
 
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
 
try:
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever
 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
 
 
def load_and_split_pdf(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 100):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
 
 
def build_vector_only_retriever(splits, k: int):
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embedding_model, collection_name="eval_store", persist_directory=None
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
 
 
def build_hybrid_retriever(splits, k: int, bm25_weight: float, vector_weight: float):
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = k
 
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embedding_model, collection_name="eval_store_hybrid", persist_directory=None
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
 
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )
 
 
def evaluate_retriever(retriever, dataset: List[Dict], label: str) -> Dict:
    """
    Roda cada pergunta do dataset contra o retriever e mede:
    - hit: pelo menos 1 keyword esperada apareceu em algum chunk recuperado.
    - keyword_recall: fração das keywords esperadas que apareceram (em qualquer chunk).
    - first_hit_rank: posição (1-indexed) do primeiro chunk que contém alguma keyword,
      ou None se nenhum chunk recuperado contém.
    """
    results = []
    for item in dataset:
        question = item["question"]
        expected = [kw.lower() for kw in item.get("expected_keywords", [])]
 
        t0 = time.time()
        docs = retriever.invoke(question)
        latency = time.time() - t0
 
        chunks_text = [d.page_content.lower() for d in docs]
 
        found_keywords = set()
        first_hit_rank = None
        for rank, chunk in enumerate(chunks_text, start=1):
            for kw in expected:
                if kw in chunk and kw not in found_keywords:
                    found_keywords.add(kw)
                    if first_hit_rank is None:
                        first_hit_rank = rank
 
        keyword_recall = len(found_keywords) / len(expected) if expected else None
        hit = len(found_keywords) > 0 if expected else None
 
        results.append({
            "question": question,
            "hit": hit,
            "keyword_recall": keyword_recall,
            "first_hit_rank": first_hit_rank,
            "n_retrieved": len(docs),
            "latency_s": round(latency, 3),
        })
 
    valid = [r for r in results if r["hit"] is not None]
    hit_rate = sum(r["hit"] for r in valid) / len(valid) if valid else 0.0
    avg_recall = sum(r["keyword_recall"] for r in valid) / len(valid) if valid else 0.0
    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0.0
    ranks = [r["first_hit_rank"] for r in valid if r["first_hit_rank"] is not None]
    mrr = sum(1 / r for r in ranks) / len(valid) if valid else 0.0
 
    summary = {
        "label": label,
        "total_questions": len(results),
        "hit_rate": round(hit_rate, 4),
        "avg_keyword_recall": round(avg_recall, 4),
        "mrr": round(mrr, 4),
        "avg_latency_s": round(avg_latency, 3),
    }
    return {"summary": summary, "details": results}
 
 
def print_report(report: Dict):
    s = report["summary"]
    print(f"\n{'=' * 50}")
    print(f"  {s['label']}")
    print(f"{'=' * 50}")
    print(f"total_questions:     {s['total_questions']}")
    print(f"hit_rate:            {s['hit_rate']}   (achou ao menos 1 keyword esperada)")
    print(f"avg_keyword_recall:  {s['avg_keyword_recall']}   (fração de keywords esperadas encontradas)")
    print(f"mrr:                 {s['mrr']}   (Mean Reciprocal Rank do 1º acerto)")
    print(f"avg_latency_s:       {s['avg_latency_s']}")
 
    misses = [d for d in report["details"] if d["hit"] is False]
    if misses:
        print(f"\nPerguntas SEM nenhuma keyword encontrada ({len(misses)}):")
        for m in misses:
            print(f"  - {m['question']}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia a qualidade de recuperação do RAG (retrieval only).")
    parser.add_argument("--pdf", required=True, help="Caminho do PDF de teste")
    parser.add_argument("--dataset", required=True, help="JSON com perguntas + keywords esperadas")
    parser.add_argument("--k", type=int, default=3, help="Número de chunks a recuperar por pergunta")
    parser.add_argument("--bm25-weight", type=float, default=0.4, help="Peso do BM25 no ensemble")
    parser.add_argument("--vector-weight", type=float, default=0.6, help="Peso da busca vetorial no ensemble")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument(
        "--compare", action="store_true",
        help="Roda vetorial-puro E híbrido lado a lado, para comparar o efeito da otimização",
    )
    args = parser.parse_args()
 
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
 
    print(f"Carregando e indexando {args.pdf}...")
    splits = load_and_split_pdf(args.pdf, args.chunk_size, args.chunk_overlap)
    print(f"{len(splits)} chunks gerados. {len(dataset)} perguntas no dataset.\n")
 
    if args.compare:
        vector_retriever = build_vector_only_retriever(splits, args.k)
        vector_report = evaluate_retriever(vector_retriever, dataset, "VETORIAL PURO (baseline)")
        print_report(vector_report)
 
        hybrid_retriever = build_hybrid_retriever(splits, args.k, args.bm25_weight, args.vector_weight)
        hybrid_report = evaluate_retriever(hybrid_retriever, dataset, "HÍBRIDO (BM25 + vetorial)")
        print_report(hybrid_report)
 
        print(f"\n{'=' * 50}")
        print("  DELTA (híbrido - vetorial puro)")
        print(f"{'=' * 50}")
        vs, hs = vector_report["summary"], hybrid_report["summary"]
        for key in ["hit_rate", "avg_keyword_recall", "mrr"]:
            delta = hs[key] - vs[key]
            sign = "+" if delta >= 0 else ""
            print(f"{key}: {sign}{round(delta, 4)}")
    else:
        retriever = build_hybrid_retriever(splits, args.k, args.bm25_weight, args.vector_weight)
        report = evaluate_retriever(retriever, dataset, "HÍBRIDO (BM25 + vetorial)")
        print_report(report)
 
        with open("rag_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("\nDetalhes salvos em: rag_eval_results.json")
"""
test_deploy.py — Testa o Space publicado com duas chamadas distintas:

1. CHAMADA NORMAL: uma saudação simples. Segundo o system_prompt do Gerente
   ("If the user greets you, answer directly"), essa chamada NÃO deveria
   acionar o Pesquisador nem o retriever — só o LLM respondendo direto.
   Serve para confirmar que o modelo básico está carregando e respondendo,
   isolado de qualquer complexidade de RAG/agentes.

2. CHAMADA COM IA (multi-agente): uma pergunta real sobre o conteúdo do
   laudo. Essa obriga o Gerente a delegar pro Pesquisador via
   'ask_researcher', que por sua vez usa 'search_pdf' contra o retriever
   híbrido (BM25 + vetorial). Se essa falhar mas a #1 funcionar, o problema
   está na camada de agentes/RAG, não no modelo em si.

Uso:
    python test_deploy.py --space seu-usuario/nome-do-space
    python test_deploy.py --space seu-usuario/nome-do-space --pdf knowledge/laudo_BM-102_2026-07-12.pdf
"""
import argparse
import os
import time

from gradio_client import Client, handle_file


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="ID do Space, ex: usuario/nome-do-space")
    parser.add_argument("--pdf", default="knowledge/laudo_BM-102_2026-07-12.pdf")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if not args.hf_token:
        print("Aviso: sem HF_TOKEN — cota anônima do ZeroGPU é bem menor e pode interromper o teste.")

    print(f"Conectando ao Space {args.space}...")
    client = Client(args.space, token=args.hf_token)

    print_section("SCHEMA REAL DA API (confira os nomes de parâmetro antes de interpretar erros abaixo)")
    client.view_api()

    # -------------------------------------------------------------------
    # PASSO 0 — Upload do PDF (necessário antes de qualquer uma das 2 chamadas)
    # -------------------------------------------------------------------
    print_section("UPLOAD DO PDF")
    try:
        upload_result = client.predict(handle_file(args.pdf), api_name="/upload_pdf")
        print(f"Resultado do upload: {upload_result}")
    except Exception as e:
        print(f"ERRO no upload: {e}")
        print("Verifique o nome/ordem dos parâmetros contra o schema impresso acima.")
        return

    # Dá um respiro para a indexação (CPU) terminar antes do primeiro chat
    time.sleep(2)

    # -------------------------------------------------------------------
    # CHAMADA 1 — NORMAL (saudação, sem precisar de RAG/agentes)
    # -------------------------------------------------------------------
    print_section("CHAMADA 1 — NORMAL (saudação, deve responder direto sem usar ferramentas)")
    t0 = time.time()
    try:
        job = client.submit("Olá! Tudo bem?", [], api_name="/chat")
        for update in job:
            pass  # chat_function é generator (yield); pega só o resultado final
        latency = time.time() - t0
        print(f"Resposta final: {update}")
        print(f"Latência: {latency:.1f}s")
    except Exception as e:
        print(f"ERRO na chamada normal: {e}")

    # -------------------------------------------------------------------
    # CHAMADA 2 — COM IA (pergunta real, exige Gerente -> Pesquisador -> search_pdf)
    # -------------------------------------------------------------------
    print_section("CHAMADA 2 — COM IA (pergunta sobre o laudo, exige pipeline de agentes + RAG)")
    t0 = time.time()
    try:
        job = client.submit(
            "Qual foi a criticidade atribuída ao equipamento e por quê?",
            [],
            api_name="/chat",
        )
        for update in job:
            pass
        latency = time.time() - t0
        print(f"Resposta final: {update}")
        print(f"Latência: {latency:.1f}s")
        print("\nEsperado no gabarito (laudo_ground_truth.json): criticidade = 'Alta',")
        print("por causa da vibração acima do limite no mancal traseiro.")
    except Exception as e:
        print(f"ERRO na chamada com IA: {e}")


if __name__ == "__main__":
    main()
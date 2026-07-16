---
title: RAG Industrial — Extração de Laudos Técnicos
emoji: 🔧
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.34.0
app_file: app.py
pinned: false
license: mit
---

# 🔧 RAG Industrial — Extração de Laudos Técnicos

Pipeline de RAG (Retrieval-Augmented Generation) com **recuperação híbrida**
(BM25 + busca vetorial) e **extração estruturada** de campos a partir de
laudos técnicos de manutenção/inspeção industrial em texto não estruturado,
rodando **100% local** com Qwen2.5-7B-Instruct (4-bit).

## O problema

Laudos técnicos de inspeção/manutenção industrial chegam em texto livre, sem
padronização entre inspetores e fornecedores. Extrair manualmente os dados
críticos (não conformidades, criticidade, prazos) é lento e sujeito a erro —
justamente onde erro tem consequência operacional real.

## O que o sistema faz

1. **Indexação híbrida**: cada PDF é quebrado em chunks e indexado tanto por
   BM25 (bom para termos exatos — códigos de equipamento, normas técnicas
   citadas, valores numéricos) quanto por embeddings semânticos (bom para
   paráfrases e perguntas em linguagem natural). Os dois rankings são
   combinados via Reciprocal Rank Fusion (`EnsembleRetriever`).
2. **Chat multi-agente**: um agente "Gerente" delega perguntas sobre o
   documento a um agente "Pesquisador", que busca no índice híbrido.
3. **Extração estruturada**: além do chat, o sistema extrai um conjunto fixo
   de campos (ver `laudo_schema.json`) em formato JSON, adequado para
   alimentar outro sistema (planilha, banco de dados, dashboard).

## Arquitetura

```
Upload PDF
    ↓
Indexação (CPU): chunking → BM25 + embeddings → EnsembleRetriever
    ↓
┌─────────────────────┐        ┌──────────────────────────┐
│  Chat (multi-agente) │        │  Extração Estruturada    │
│  Gerente → Pesquisador│       │  Prompt schema-driven    │
│  → search_pdf (híbrido)│      │  → JSON validado          │
└─────────────────────┘        └──────────────────────────┘
    ↓                                    ↓
Qwen2.5-7B-Instruct (4-bit, local, ZeroGPU)
```

## Nota sobre dados e conformidade (LGPD)

O schema de extração (`laudo_schema.json`) documenta explicitamente quais
campos são dado técnico/operacional (não pessoal) e quais são dado pessoal
sujeito à LGPD. Nesse domínio, apenas `inspetor_responsavel` (nome + registro
profissional) é dado pessoal — o restante do laudo é informação sobre o
equipamento.

O pipeline roda inteiramente na sua própria infraestrutura (Qwen2.5
quantizado, local): nenhum conteúdo de documento é enviado a APIs de
terceiros. Isso reduz a superfície de exposição do campo pessoal e evita
questões de transferência internacional de dados (LGPD art. 33) que
surgiriam ao usar um provedor de LLM hospedado fora do Brasil.

## Avaliação

- `rag_eval.py` — mede a qualidade da **recuperação** (retrieval) isolada do
  LLM: hit-rate, recall de keywords e MRR, comparando vetorial puro vs.
  híbrido. Roda em CPU, sem depender de GPU nem do modelo carregado.
- `extraction_eval.py` *(próximo passo)* — mede a acurácia campo-a-campo da
  **extração estruturada** contra um gabarito (`laudo_ground_truth.json`).

## Stack

- Qwen2.5-7B-Instruct (4-bit, `bitsandbytes`)
- `smolagents` (CodeAgent multi-agente)
- LangChain (`PyPDFLoader`, `RecursiveCharacterTextSplitter`, `Chroma`,
  `BM25Retriever`, `EnsembleRetriever`)
- `sentence-transformers/all-MiniLM-L6-v2` (embeddings)
- Gradio + Hugging Face ZeroGPU

## Documento de teste

`laudo_BM-102_2026-07-12.pdf` é um laudo sintético (dados fictícios) usado
para desenvolvimento e avaliação, com gabarito em `laudo_ground_truth.json`.
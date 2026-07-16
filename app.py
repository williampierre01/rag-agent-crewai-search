import gradio as gr
import os
import json
import spaces
import torch
import gc
import traceback
from smolagents import CodeAgent, Tool, Model, ChatMessage

# Imports de RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

try:
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever

# Imports Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# ==============================================================================
#  ESTADO GLOBAL
# ==============================================================================
# Apenas a engine (LLM) é global: é cara de carregar (7B em 4-bit) e não
# depende de dados do usuário, então é seguro compartilhá-la entre sessões.
#
# O retriever e o texto do documento NUNCA são globais — cada usuário sobe um
# PDF diferente, e um estado global aqui misturaria documentos entre sessões
# simultâneas. Ambos vivem em gr.State, por usuário/aba do navegador.
GLOBAL_ENGINE = None

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "laudo_schema.json")
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    LAUDO_SCHEMA = json.load(f)

# ==============================================================================
#  1. ENGINE CUSTOMIZADA (CORREÇÃO DE TIPOS LIST vs STR)
# ==============================================================================
class LocalQwenEngine(Model):
    def __init__(self, pipeline_obj):
        super().__init__()
        self.pipeline = pipeline_obj
        self.tokenizer = pipeline_obj.tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, messages, stop_sequences=None, grammar=None, **kwargs):
        """
        Gera resposta tratando erros de tipo (List vs String) que o smolagents pode enviar.
        """
        clean_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ['pipeline', 'grammar', 'stop_sequences', 'adapter_id']
        }

        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
            else:
                role = getattr(msg, 'role', 'user')
                content = getattr(msg, 'content', str(msg))

            if isinstance(content, list):
                content_str = "\n".join([str(item) for item in content])
            else:
                content_str = str(content)

            formatted_messages.append({"role": role, "content": content_str})

        try:
            prompt = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            prompt = str(formatted_messages)

        # stop_sequences vem do smolagents e é essencial: é como o CodeAgent
        # sabe onde termina o bloco de código gerado.
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
                stop_strings=stop_sequences if stop_sequences else None,
                tokenizer=self.tokenizer,
                **clean_kwargs
            )

            response_text = outputs[0]["generated_text"]
            return ChatMessage(role="assistant", content=response_text)

        except Exception as e:
            print(f"Erro na geração: {e}")
            return ChatMessage(role="assistant", content=f"Error: {str(e)}")

    def raw_generate(self, prompt_text: str, max_new_tokens: int = 1024) -> str:
        """
        Geração direta (sem o formato de mensagens do smolagents), usada pela
        extração estruturada — não precisamos do roteamento de agente/ferramentas
        para isso, só de um prompt único pedindo JSON.
        """
        outputs = self.pipeline(
            prompt_text,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # extração quer determinismo, não criatividade
            temperature=0.0,
        )
        return outputs[0]["generated_text"]

# ==============================================================================
#  2. CARREGAMENTO (SETUP)
# ==============================================================================
def get_or_create_engine():
    global GLOBAL_ENGINE
    if GLOBAL_ENGINE is not None:
        return GLOBAL_ENGINE

    print("--- CARREGANDO QWEN 2.5 (4-BIT) ---")

    model_id = "Qwen/Qwen2.5-7B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False
    )

    GLOBAL_ENGINE = LocalQwenEngine(pipeline_obj=text_pipeline)
    print("--- ENGINE PRONTA ---")
    return GLOBAL_ENGINE

# ==============================================================================
#  3. FERRAMENTAS E AGENTES (CHAT)
# ==============================================================================

class PDFSearchTool(Tool):
    name = "search_pdf"
    description = "Search for specific information within the uploaded PDF document."
    inputs = {"query": {"type": "string", "description": "Keywords to search."}}
    output_type = "string"

    def __init__(self, retriever):
        self.retriever = retriever
        super().__init__()

    def forward(self, query: str) -> str:
        if self.retriever is None:
            return "Error: No PDF uploaded."
        try:
            docs = self.retriever.invoke(query)
            if not docs:
                return "No info found."
            return "\n\n".join([f"--- Content ---\n{doc.page_content}" for doc in docs])
        except Exception as e:
            return f"Error: {str(e)}"

class AgentAsTool(Tool):
    def __init__(self, agent, name, description):
        self.agent = agent
        self.name = name
        self.description = description
        self.inputs = {
            "task": {"type": "string", "description": "The task for the agent."}
        }
        self.output_type = "string"
        super().__init__()

    def forward(self, task: str) -> str:
        try:
            return str(self.agent.run(task))
        except Exception as e:
            return f"Error: {str(e)}"

def get_manager_agent(retriever):
    engine = get_or_create_engine()

    researcher = CodeAgent(
        tools=[PDFSearchTool(retriever)],
        model=engine,
        add_base_tools=False,
        name="pdf_researcher",
        description="Reads PDFs."
    )

    researcher_tool = AgentAsTool(
        agent=researcher,
        name="ask_researcher",
        description="Ask the Researcher to find info in the PDF."
    )

    manager = CodeAgent(
        tools=[researcher_tool],
        model=engine,
        add_base_tools=False
    )
    return manager

# ==============================================================================
#  4. RAG — INDEXAÇÃO E RECUPERAÇÃO HÍBRIDA
# ==============================================================================
#  Funções puras (sem decorators, sem estado global) — reusáveis pelo
#  rag_eval.py sem precisar subir o Gradio nem o LLM.
# ==============================================================================
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
DEFAULT_K = 3
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_and_split_pdf(pdf_path, chunk_size=800, chunk_overlap=100):
    """Carrega o PDF e quebra em chunks. Isolado para ser reusável e testável."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def build_hybrid_retriever(splits, k=DEFAULT_K):
    """
    Constrói um retriever híbrido combinando BM25 (lexical/keyword — forte em
    termos exatos, normas técnicas, códigos de equipamento) com busca vetorial
    (semântica — forte em paráfrases). O EnsembleRetriever combina os
    rankings via Reciprocal Rank Fusion.
    """
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = k

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name="pdf_store",
        persist_directory=None,
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[BM25_WEIGHT, VECTOR_WEIGHT],
    )


def build_vector_store(pdf_path):
    """
    Ponto de entrada usado pelo Gradio. Indexação roda inteiramente em CPU
    (embeddings leves + BM25) — não precisa de @spaces.GPU aqui, já que a
    LLM de 7B não é carregada nessa etapa. Retorna (retriever, texto_completo):
    o texto completo é usado pela extração estruturada (ver seção 5), que
    precisa ver o documento inteiro de uma vez, não só os top-k chunks de
    uma busca — extração de campos não é a mesma tarefa que responder 1
    pergunta pontual.
    """
    print(f"[RAG] Indexando: {pdf_path}")
    try:
        splits = load_and_split_pdf(pdf_path)
        retriever = build_hybrid_retriever(splits)
        full_text = "\n\n".join(d.page_content for d in splits)
        return retriever, full_text
    except Exception as e:
        print(f"Erro: {e}")
        return None, None

# ==============================================================================
#  5. EXTRAÇÃO ESTRUTURADA (SCHEMA-DRIVEN)
# ==============================================================================
def build_extraction_prompt(document_text: str, schema: dict) -> str:
    fields_desc = "\n".join(
        f'- "{name}" ({info.get("type")}): {info.get("description", "")}'
        for name, info in schema.get("properties", {}).items()
    )
    required = ", ".join(schema.get("required", []))

    return f"""Você é um extrator de dados técnicos preciso. Sua única tarefa é ler o
documento abaixo e devolver APENAS um objeto JSON válido com os campos a seguir,
sem nenhum texto antes ou depois, sem markdown, sem explicações.

CAMPOS A EXTRAIR:
{fields_desc}

Campos obrigatórios (nunca deixe em branco se a informação existir no texto): {required}

Regras:
- Se um campo não for encontrado no documento, use null.
- Para campos do tipo "array", sempre devolva uma lista, mesmo que com 1 item.
- Não invente valores. Extraia apenas o que está explicitamente no texto.
- Normalize números (ex: "7,8 mm/s" e "7.8 mm/s" são o mesmo valor).

DOCUMENTO:
\"\"\"
{document_text}
\"\"\"

Responda APENAS com o JSON:"""


def parse_extraction_output(raw_text: str) -> dict:
    """
    Tenta parsear o JSON da saída do modelo de forma tolerante: o modelo pode
    (mesmo instruído a não fazer) envolver a resposta em texto ou markdown.
    """
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {"error": "Não foi possível parsear JSON da resposta do modelo.", "raw_output": raw_text}


@spaces.GPU(duration=90)
def extract_fields_function(document_text):
    if not document_text:
        return {"error": "Nenhum documento indexado. Faça upload de um PDF primeiro."}

    try:
        engine = get_or_create_engine()
        prompt = build_extraction_prompt(document_text, LAUDO_SCHEMA)
        raw_output = engine.raw_generate(prompt)
        return parse_extraction_output(raw_output)
    except Exception as e:
        print(traceback.format_exc())
        return {"error": f"Erro na extração: {str(e)}"}

# ==============================================================================
#  6. CHAT
# ==============================================================================
@spaces.GPU(duration=120)
def chat_function(message, history, retriever):
    if not message:
        return history, ""
    if history is None:
        history = []

    history.append([message, "🤖 Gerente processando..."])
    yield history, ""

    try:
        manager = get_manager_agent(retriever)

        system_prompt = """
        You are a Manager.
        If the user asks about the PDF, use 'ask_researcher'.
        If the user greets you, answer directly.
        """

        response = manager.run(f"{system_prompt}\nUser: {message}")
        history[-1] = [message, str(response)]

    except Exception as e:
        history[-1] = [message, f"Erro: {str(e)}"]
        print(traceback.format_exc())

    yield history, ""

# ==============================================================================
#  INTERFACE
# ==============================================================================
def process_pdf(file_obj):
    if not file_obj:
        return "Sem arquivo.", None, None
    pdf_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
    retriever, full_text = build_vector_store(pdf_path)
    if retriever is None:
        return "Erro ao indexar.", None, None
    return "PDF Pronto! Agentes Ativos.", retriever, full_text

with gr.Blocks(title="RAG Industrial · Extração de Laudos Técnicos (Qwen2.5-7B Local)") as demo:
    gr.Markdown("# 🔧 Extração de Laudos Técnicos Industriais")
    gr.Markdown(
        "RAG híbrido (BM25 + vetorial) + extração estruturada, rodando 100% local "
        "(Qwen2.5-7B em 4-bit) — nenhum dado do documento é enviado a APIs de terceiros."
    )

    # Estado por sessão/aba — cada usuário tem seu próprio retriever/texto,
    # sem interferir no de outros usuários conectados ao mesmo Space.
    retriever_state = gr.State(None)
    doc_text_state = gr.State(None)

    with gr.Row():
        upl = gr.File(label="Upload do Laudo (PDF)")
        st = gr.Markdown("...")

    with gr.Tab("💬 Chat"):
        chat = gr.Chatbot(height=500)
        msg = gr.Textbox(label="Pergunta")

    with gr.Tab("📋 Extração Estruturada"):
        gr.Markdown(
            "Extrai os campos definidos em `laudo_schema.json` diretamente do documento. "
            "O campo `inspetor_responsavel` contém dado pessoal (LGPD) — trate a saída "
            "com o mesmo cuidado que trataria o documento original."
        )
        extract_btn = gr.Button("Extrair Dados Estruturados", variant="primary")
        extract_output = gr.JSON(label="Dados Extraídos")

    upl.change(process_pdf, upl, [st, retriever_state, doc_text_state])
    msg.submit(chat_function, [msg, chat, retriever_state], [chat, msg])
    extract_btn.click(extract_fields_function, doc_text_state, extract_output)

if __name__ == "__main__":
    demo.launch()
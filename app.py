import gradio as gr
import os
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

# Imports Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# ==============================================================================
#  ESTADO GLOBAL
# ==============================================================================
# Apenas a engine (LLM) é global: é cara de carregar (7B em 4-bit) e não
# depende de dados do usuário, então é seguro compartilhá-la entre sessões.
#
# O retriever (índice do PDF) NUNCA deve ser global — cada usuário sobe um
# PDF diferente, e um estado global aqui misturaria documentos entre sessões
# simultâneas. Ele agora vive em gr.State, por usuário/aba do navegador.
GLOBAL_ENGINE = None

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
        # 1. Limpeza de Argumentos
        clean_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ['pipeline', 'grammar', 'stop_sequences', 'adapter_id']
        }

        # 2. Conversão e Sanitização de Mensagens
        formatted_messages = []
        for msg in messages:
            # Identifica Role e Content
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
            else:
                role = getattr(msg, 'role', 'user')
                content = getattr(msg, 'content', str(msg))

            # --- CORREÇÃO DO ERRO DE CONCATENAÇÃO ---
            # Se o conteúdo for uma lista (ex: uso de ferramenta), converte para string
            if isinstance(content, list):
                content_str = "\n".join([str(item) for item in content])
            else:
                content_str = str(content)

            formatted_messages.append({"role": role, "content": content_str})

        # 3. Aplica o template de chat
        try:
            prompt = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback de emergência se o template falhar
            prompt = str(formatted_messages)

        # 4. Geração
        # stop_sequences vem do smolagents e é essencial: é como o CodeAgent
        # sabe onde termina o bloco de código gerado. Sem isso, o modelo pode
        # continuar gerando texto além do esperado e alucinar a próxima
        # "observação" da ferramenta como se fosse dele mesmo.
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
#  3. FERRAMENTAS E AGENTES
# ==============================================================================

class PDFSearchTool(Tool):
    name = "search_pdf"
    description = "Search for specific information within the uploaded PDF document."
    inputs = {"query": {"type": "string", "description": "Keywords to search."}}
    output_type = "string"

    def __init__(self, retriever):
        # Recebe o retriever explicitamente (por sessão) em vez de ler de
        # uma variável global — é isso que impede o vazamento de dados
        # entre usuários que sobem PDFs diferentes ao mesmo tempo.
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

    # 1. Pesquisador — recebe o retriever da sessão atual
    researcher = CodeAgent(
        tools=[PDFSearchTool(retriever)],
        model=engine,
        add_base_tools=False,
        name="pdf_researcher",
        description="Reads PDFs."
    )

    # 2. Ferramenta Pesquisador
    researcher_tool = AgentAsTool(
        agent=researcher,
        name="ask_researcher",
        description="Ask the Researcher to find info in the PDF."
    )

    # 3. Gerente
    manager = CodeAgent(
        tools=[researcher_tool],
        model=engine,
        add_base_tools=False
    )
    return manager

# ==============================================================================
#  4. RAG E CHAT
# ==============================================================================
@spaces.GPU
def build_vector_store(pdf_path):
    """
    Indexa o PDF e RETORNA o retriever (não seta estado global).
    Só usa o modelo de embeddings (leve) — a LLM de 7B não é carregada
    aqui, pois indexação não precisa dela. Isso evita estourar o teto de
    tempo do ZeroGPU só carregando um modelo que ainda não é necessário.
    """
    print(f"[RAG] Indexando: {pdf_path}")
    try:
        torch.cuda.empty_cache()
        gc.collect()

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            collection_name="pdf_store",
            persist_directory=None
        )
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"Erro: {e}")
        return None

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
        return "Sem arquivo.", None
    retriever = build_vector_store(file_obj.name if hasattr(file_obj, 'name') else file_obj)
    if retriever is None:
        return "Erro ao indexar.", None
    return "PDF Pronto! Agentes Ativos.", retriever

with gr.Blocks(title="Multi-Agent RAG · Qwen2.5-7B") as demo:
    gr.Markdown("# 🤖 Multi-Agent RAG (Qwen 7B Local)")

    # Estado por sessão/aba — cada usuário tem seu próprio retriever,
    # sem interferir no de outros usuários conectados ao mesmo Space.
    retriever_state = gr.State(None)

    with gr.Row():
        upl = gr.File(label="Upload PDF")
        st = gr.Markdown("...")
    chat = gr.Chatbot(height=600)
    msg = gr.Textbox(label="Pergunta")

    upl.change(process_pdf, upl, [st, retriever_state])
    msg.submit(chat_function, [msg, chat, retriever_state], [chat, msg])

if __name__ == "__main__":
    demo.launch()
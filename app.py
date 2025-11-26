import gradio as gr
import os
import spaces
import torch
import gc
import traceback
from smolagents import CodeAgent, Tool, TransformersModel

# Imports de RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Imports Transformers (Para criar o Pipeline Manual)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# ==============================================================================
#  ESTADO GLOBAL
# ==============================================================================
GLOBAL_RETRIEVER = None
GLOBAL_AGENT_MANAGER = None # Armazena o Agente Principal

# ==============================================================================
#  1. CARREGAMENTO DO MODELO VIA PIPELINE (CORREÇÃO DEFINITIVA)
# ==============================================================================
def load_model_engine():
    """
    Cria um 'TransformersModel' usando um pipeline pré-carregado.
    Isso evita erros de kwargs e erros de classe abstrata.
    """
    print("--- CARREGANDO QWEN 2.5 (4-BIT) ---")
    
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    # Configuração de Memória
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Modelo
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Pipeline (A Ponte Segura)
    # Configuramos os parâmetros de geração AQUI para o smolagents usar
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.1,
        top_p=0.95
    )

    # Criamos o Wrapper Oficial do Smolagents passando o pipeline
    engine = TransformersModel(pipeline=text_pipeline)
    
    print("--- ENGINE PRONTA ---")
    return engine

# ==============================================================================
#  2. FERRAMENTA DE PDF
# ==============================================================================
class PDFSearchTool(Tool):
    name = "search_pdf"
    description = "Search for specific information within the uploaded PDF document."
    inputs = {"query": {"type": "string", "description": "Keywords to search."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        global GLOBAL_RETRIEVER
        if GLOBAL_RETRIEVER is None: return "Error: No PDF uploaded."
        try:
            docs = GLOBAL_RETRIEVER.invoke(query)
            return "\n\n".join([f"--- Content ---\n{doc.page_content}" for doc in docs])
        except Exception as e: return f"Error: {str(e)}"

# ==============================================================================
#  3. ADAPTADOR DE AGENTE (AGENT AS TOOL)
# ==============================================================================
class AgentAsTool(Tool):
    """
    Permite que um Agente seja usado como ferramenta por outro Agente.
    """
    def __init__(self, agent, name, description):
        self.agent = agent
        self.name = name
        self.description = description
        self.inputs = {
            "task": {
                "type": "string",
                "description": "The question to ask this agent."
            }
        }
        self.output_type = "string"
        super().__init__()

    def forward(self, task: str) -> str:
        try:
            return str(self.agent.run(task))
        except Exception as e:
            return f"Error from sub-agent: {str(e)}"

# ==============================================================================
#  4. SISTEMA MULTI-AGENTE (SETUP)
# ==============================================================================
def get_or_create_manager():
    global GLOBAL_AGENT_MANAGER
    
    # Se já existe, retorna (Singleton)
    if GLOBAL_AGENT_MANAGER is not None:
        return GLOBAL_AGENT_MANAGER

    # Carrega a Engine (Modelo)
    model_engine = load_model_engine()

    # --- AGENTE 1: PESQUISADOR (SUB-AGENTE) ---
    researcher = CodeAgent(
        tools=[PDFSearchTool()],
        model=model_engine,
        add_base_tools=False,
        name="pdf_researcher",
        description="An expert analyst that reads PDFs."
    )

    # Transformamos em Ferramenta
    researcher_tool = AgentAsTool(
        agent=researcher,
        name="ask_researcher",
        description="Use this tool to ask the Researcher to find facts in the PDF."
    )

    # --- AGENTE 2: GERENTE (PRINCIPAL) ---
    manager = CodeAgent(
        tools=[researcher_tool],
        model=model_engine,
        add_base_tools=False,
        name="manager",
        description="A manager that delegates tasks."
    )
    
    GLOBAL_AGENT_MANAGER = manager
    return manager

# ==============================================================================
#  5. INDEXAÇÃO
# ==============================================================================
@spaces.GPU
def build_vector_store(pdf_path):
    global GLOBAL_RETRIEVER
    # Forçamos a criação do agente aqui para carregar o modelo na GPU logo no início
    get_or_create_manager()
    
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
        GLOBAL_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 3})
        return True
    except Exception as e:
        print(f"Erro: {e}")
        return False

# ==============================================================================
#  6. CHAT
# ==============================================================================
@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    history.append([message, "🤖 Gerente coordenando..."])
    yield history

    try:
        # Pega o gerente (já carregado)
        manager = get_or_create_manager()
        
        # Instrução clara para o modelo usar a ferramenta
        system_prompt = """
        You are a Manager Agent.
        If the user asks about the document, use the tool 'ask_researcher'.
        If the user just says 'hi' or asks general questions, answer directly.
        
        User Query:
        """
        
        # Executa
        response = manager.run(f"{system_prompt} {message}")
        history[-1] = [message, str(response)]
        
    except Exception as e:
        history[-1] = [message, f"Erro: {str(e)}"]
        print(traceback.format_exc())
    
    yield history

# ==============================================================================
#  INTERFACE
# ==============================================================================
def process_pdf(file_obj):
    if not file_obj: return "Sem arquivo."
    if build_vector_store(file_obj.name if hasattr(file_obj, 'name') else file_obj):
        return "PDF Pronto! Agentes Ativos."
    return "Erro ao indexar."

with gr.Blocks(title="Multi-Agent Pipeline Fix") as demo:
    gr.Markdown("# 🤖 Multi-Agent RAG (Pipeline Fix)")
    gr.Markdown("Usando **Qwen 2.5 7B** via Pipeline para máxima compatibilidade.")
    with gr.Row():
        upl = gr.File(label="Upload PDF")
        st = gr.Markdown("...")
    chat = gr.Chatbot(height=600)
    msg = gr.Textbox(label="Pergunta")
    upl.change(process_pdf, upl, st)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()

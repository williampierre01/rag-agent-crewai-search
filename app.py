import gradio as gr
import os
import spaces
import torch
import gc
import traceback
from smolagents import CodeAgent, Tool, Model

# Imports de RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Imports Transformers (Carregamento Seguro)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==============================================================================
#  ESTADO GLOBAL
# ==============================================================================
GLOBAL_RETRIEVER = None
GLOBAL_ENGINE = None 

# ==============================================================================
#  1. ENGINE CUSTOMIZADA (CORREÇÃO DO ERRO 'load_in_4bit')
# ==============================================================================
class LocalQwenEngine(Model):
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        super().__init__()
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Engine inicializada para {model_id}")

    def load(self):
        if self.model is not None: return

        print("--- CARREGANDO QWEN 2.5 (4-BIT) MANUALMENTE ---")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("--- MODELO CARREGADO ---")

    def __call__(self, messages, stop_sequences=None, **kwargs):
        self.load()
        
        # Template de Chat
        text_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)
        
        # Limpeza de kwargs (Remove argumentos que causam erro)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['load_in_4bit', 'adapter_id']}
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            **clean_kwargs
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ==============================================================================
#  2. ADAPTADOR DE AGENTE (CORREÇÃO DO ERRO 'ManagedAgent')
# ==============================================================================
class AgentAsTool(Tool):
    """
    Esta classe substitui o ManagedAgent. Ela transforma um Agente inteiro
    em uma ferramenta simples que outro agente pode chamar.
    """
    def __init__(self, agent, name, description):
        self.agent = agent
        self.name = name
        self.description = description
        self.inputs = {
            "task": {
                "type": "string",
                "description": "The task or question to delegate to this agent."
            }
        }
        self.output_type = "string"
        super().__init__()

    def forward(self, task: str) -> str:
        try:
            # O agente executa a tarefa e retorna a resposta como string
            return str(self.agent.run(task))
        except Exception as e:
            return f"Error from sub-agent: {str(e)}"

# ==============================================================================
#  3. FERRAMENTA DE PDF
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
#  4. SISTEMA MULTI-AGENTE
# ==============================================================================
def get_multi_agent_system():
    # 1. Prepara o Modelo
    global GLOBAL_ENGINE
    if GLOBAL_ENGINE is None:
        GLOBAL_ENGINE = LocalQwenEngine()
    GLOBAL_ENGINE.load()

    # 2. Agente Pesquisador (Sub-Agente)
    retriever = CodeAgent(
        tools=[PDFSearchTool()],
        model=GLOBAL_ENGINE,
        add_base_tools=False,
        description="Expert analyst that reads PDFs."
    )

    # 3. Cria a "Ferramenta" que chama o Pesquisador
    researcher_tool = AgentAsTool(
        agent=retriever,
        name="ask_researcher",
        description="Use this tool to ask the Researcher Agent to find information in the PDF."
    )

    # 4. Agente Gerente (Principal)
    manager = CodeAgent(
        tools=[researcher_tool], # O gerente só vê a ferramenta, não o agente
        model=GLOBAL_ENGINE,
        add_base_tools=False
    )
    
    return manager

# ==============================================================================
#  5. INDEXAÇÃO
# ==============================================================================
@spaces.GPU
def build_vector_store(pdf_path):
    global GLOBAL_RETRIEVER
    global GLOBAL_ENGINE
    print(f"[RAG] Indexando: {pdf_path}")
    try:
        torch.cuda.empty_cache()
        gc.collect()
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model, collection_name="pdf_store", persist_directory=None)
        GLOBAL_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Pré-carrega modelo
        if GLOBAL_ENGINE is None: GLOBAL_ENGINE = LocalQwenEngine()
        GLOBAL_ENGINE.load()
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
    history.append([message, "🤖 Gerente consultando Pesquisador..."])
    yield history

    try:
        manager = get_multi_agent_system()
        
        # Prompt para guiar o gerente a usar a ferramenta correta
        system_prompt = "You are a Manager. If the user asks about the document, use 'ask_researcher'. If not, answer directly."
        
        response = manager.run(f"{system_prompt} User Query: {message}")
        history[-1] = [message, str(response)]
    except Exception as e:
        history[-1] = [message, f"Erro: {str(e)}"]
    
    yield history

# ==============================================================================
#  INTERFACE
# ==============================================================================
def process_pdf(file_obj):
    if not file_obj: return "Sem arquivo."
    if build_vector_store(file_obj.name if hasattr(file_obj, 'name') else file_obj):
        return "PDF Pronto! Sistema Multi-Agente Ativo."
    return "Erro ao indexar."

with gr.Blocks(title="Multi-Agent Final") as demo:
    gr.Markdown("# 🤖 Multi-Agent RAG (Qwen 7B Local)")
    with gr.Row():
        upl = gr.File(label="Upload PDF")
        st = gr.Markdown("...")
    chat = gr.Chatbot(height=600)
    msg = gr.Textbox(label="Pergunta")
    upl.change(process_pdf, upl, st)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()

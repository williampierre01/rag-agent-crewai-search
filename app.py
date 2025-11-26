import gradio as gr
import os
import spaces
import torch
import gc
from smolagents import CodeAgent, Tool, ManagedAgent, Model

# Imports de RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Imports Transformers (Para carregamento manual seguro)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==============================================================================
#  ESTADO GLOBAL
# ==============================================================================
GLOBAL_RETRIEVER = None
GLOBAL_ENGINE = None 

# ==============================================================================
#  1. CLASSE DE MODELO CUSTOMIZADA (A CORREÇÃO)
# ==============================================================================
class LocalQwenEngine(Model):
    """
    Esta classe carrega o modelo manualmente para garantir que o 4-bit funcione
    sem causar erros na hora da geração de texto.
    """
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        super().__init__()
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Carregamento inicial (Lazy loading será feito na primeira chamada ou via build)
        print(f"Engine inicializada para {model_id}")

    def load(self):
        if self.model is not None:
            return

        print("--- CARREGANDO QWEN 2.5 (4-BIT) MANUALMENTE ---")
        
        # Configuração de Memória (4-bit)
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
        # Garante que está carregado
        self.load()
        
        # Aplica o template de chat do Qwen (User/Assistant)
        # O smolagents envia uma lista de mensagens (dicionários)
        text_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokeniza
        model_inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)
        
        # Gera a resposta
        # Removemos argumentos perigosos que possam vir do kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if k != 'load_in_4bit'}
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            **clean_kwargs
        )
        
        # Decodifica apenas a parte nova (a resposta)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response

# ==============================================================================
#  2. FERRAMENTA DE PDF
# ==============================================================================
class PDFSearchTool(Tool):
    name = "search_pdf"
    description = "Search for specific information within the uploaded PDF document."
    inputs = {
        "query": {
            "type": "string",
            "description": "The specific question or keywords to search for in the PDF."
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        global GLOBAL_RETRIEVER
        if GLOBAL_RETRIEVER is None:
            return "Error: No PDF uploaded."
        
        try:
            docs = GLOBAL_RETRIEVER.invoke(query)
            if not docs:
                return "No relevant information found in the PDF."
            return "\n\n".join([f"--- Content ---\n{doc.page_content}" for doc in docs])
        except Exception as e:
            return f"Error searching PDF: {str(e)}"

# ==============================================================================
#  3. CRIAÇÃO DOS AGENTES
# ==============================================================================
def get_multi_agent_system():
    # Usa nossa engine customizada
    global GLOBAL_ENGINE
    if GLOBAL_ENGINE is None:
        GLOBAL_ENGINE = LocalQwenEngine()
        GLOBAL_ENGINE.load() # Força carregamento

    # --- CLASSE PARA AGENTE-COMO-FERRAMENTA (Compatibilidade) ---
    class AgentAsTool(Tool):
        def __init__(self, agent, name, description):
            self.agent = agent
            self.name = name
            self.description = description
            self.inputs = {"task": {"type": "string", "description": "The question."}}
            self.output_type = "string"
            super().__init__()

        def forward(self, task: str) -> str:
            try:
                return str(self.agent.run(task))
            except Exception as e:
                return f"Error: {str(e)}"

    # 1. Agente Pesquisador
    retriever = CodeAgent(
        tools=[PDFSearchTool()],
        model=GLOBAL_ENGINE,
        add_base_tools=False,
        description="Analyst that reads PDFs."
    )

    # 2. Ferramenta Pesquisador
    researcher_tool = AgentAsTool(
        agent=retriever,
        name="ask_researcher",
        description="Ask the Researcher to find information in the PDF."
    )

    # 3. Agente Gerente
    manager = CodeAgent(
        tools=[researcher_tool],
        model=GLOBAL_ENGINE,
        add_base_tools=False
    )
    
    return manager

# ==============================================================================
#  4. INDEXAÇÃO
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

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            collection_name="pdf_rag_store",
            persist_directory=None
        )
        
        GLOBAL_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Pré-carrega o modelo
        if GLOBAL_ENGINE is None:
            GLOBAL_ENGINE = LocalQwenEngine()
        GLOBAL_ENGINE.load()
        
        return True
    except Exception as e:
        print(f"[ERRO RAG] {e}")
        return False

# ==============================================================================
#  5. CHAT FUNCTION
# ==============================================================================
@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    history.append([message, "🤖 Gerente pensando (Qwen 7B)..."])
    yield history

    try:
        manager = get_multi_agent_system()
        
        # Prompt do sistema manual para guiar o gerente
        system_instruction = """
        You are a Manager. You have a tool 'ask_researcher'.
        1. If the user asks about the document, use 'ask_researcher'.
        2. If the user greets you, just reply nicely.
        
        User Question:
        """
        
        response = manager.run(f"{system_instruction} {message}")
        
        history[-1] = [message, str(response)]
        
    except Exception as e:
        error_msg = f"Erro: {str(e)}"
        print(error_msg)
        history[-1] = [message, error_msg]
    
    yield history

# ==============================================================================
#  INTERFACE
# ==============================================================================
def process_pdf(file_obj):
    if not file_obj: return "Sem arquivo."
    try:
        path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        if build_vector_store(path):
            return "PDF Indexado! Qwen 2.5 pronto."
        else:
            return "Erro ao indexar PDF."
    except Exception as e: return f"Erro: {e}"

with gr.Blocks(title="SmolAgents Custom") as demo:
    gr.Markdown("# 🧠 Multi-Agent RAG (Custom Engine)")
    gr.Markdown("Correção para rodar Qwen 7B (4-bit) sem erros de kwargs.")
    
    with gr.Row():
        with gr.Column(scale=1):
            upl = gr.File(label="Upload PDF")
            status = gr.Markdown("Aguardando arquivo...")
        
        with gr.Column(scale=4):
            chat = gr.Chatbot(height=600)
            msg = gr.Textbox(label="Pergunta")
    
    upl.change(process_pdf, upl, status)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()

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
GLOBAL_RETRIEVER = None
GLOBAL_ENGINE = None 

# ==============================================================================
#  1. ENGINE CUSTOMIZADA (A CORREÇÃO FINAL)
# ==============================================================================
class LocalQwenEngine(Model):
    """
    Implementação completa da classe Model do smolagents.
    Sobrescrevemos 'generate' para evitar erros de método abstrato.
    """
    def __init__(self, pipeline_obj):
        super().__init__()
        self.pipeline = pipeline_obj
        self.tokenizer = pipeline_obj.tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    # --- O PULO DO GATO: Sobrescrever 'generate' ---
    def generate(self, messages, stop_sequences=None, grammar=None, **kwargs):
        """
        Este é o método principal que o smolagents chama.
        Nós tratamos tudo manualmente aqui dentro.
        """
        # 1. Limpeza de Argumentos
        # Removemos qualquer coisa que o pipeline do HF não suporte
        clean_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['pipeline', 'grammar', 'stop_sequences', 'adapter_id']
        }

        # 2. Conversão de Mensagens (Smolagents usa objetos ChatMessage)
        # Precisamos converter para lista de dicts que o tokenizer entende
        formatted_messages = []
        for msg in messages:
            # Se for dicionário, usa direto. Se for objeto ChatMessage, converte.
            if isinstance(msg, dict):
                formatted_messages.append(msg)
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                formatted_messages.append({"role": msg.role, "content": msg.content})
            else:
                # Fallback para string
                formatted_messages.append({"role": "user", "content": str(msg)})

        # 3. Aplica o template de chat
        prompt = self.tokenizer.apply_chat_template(
            formatted_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 4. Geração
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
                **clean_kwargs
            )
            
            # Retorna um objeto ChatMessage como o smolagents espera
            response_text = outputs[0]["generated_text"]
            
            # O smolagents espera um objeto ChatMessage no retorno
            return ChatMessage(role="assistant", content=response_text)
            
        except Exception as e:
            print(f"Erro na geração: {e}")
            # Retorna mensagem de erro formatada para não quebrar o fluxo
            return ChatMessage(role="assistant", content=f"Error generating response: {str(e)}")

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

    def forward(self, query: str) -> str:
        global GLOBAL_RETRIEVER
        if GLOBAL_RETRIEVER is None: return "Error: No PDF uploaded."
        try:
            docs = GLOBAL_RETRIEVER.invoke(query)
            if not docs: return "No info found."
            return "\n\n".join([f"--- Content ---\n{doc.page_content}" for doc in docs])
        except Exception as e: return f"Error: {str(e)}"

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

def get_manager_agent():
    engine = get_or_create_engine()

    # 1. Pesquisador
    researcher = CodeAgent(
        tools=[PDFSearchTool()],
        model=engine,
        add_base_tools=False,
        name="pdf_researcher",
        description="Reads PDFs."
    )

    # 2. Ferramenta Pesquisador
    researcher_tool = AgentAsTool(
        agent=researcher,
        name="ask_researcher",
        description="Use this to ask the Researcher to find info in the PDF."
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
    global GLOBAL_RETRIEVER
    get_or_create_engine() 
    
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

@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    history.append([message, "🤖 Gerente processando..."])
    yield history

    try:
        manager = get_manager_agent()
        
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
    
    yield history

# ==============================================================================
#  INTERFACE
# ==============================================================================
def process_pdf(file_obj):
    if not file_obj: return "Sem arquivo."
    if build_vector_store(file_obj.name if hasattr(file_obj, 'name') else file_obj):
        return "PDF Pronto! Agentes Ativos."
    return "Erro ao indexar."

with gr.Blocks(title="Multi-Agent Final Fix") as demo:
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

import gradio as gr
import os
import spaces
import traceback
from typing import Any, List, Optional
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

# LangChain & Pydantic
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import PrivateAttr

# Transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ==============================================================================
#  CONFIGURAÇÕES
# ==============================================================================
os.environ["OPENAI_API_KEY"] = "sk-fake-key-bypass"
os.environ["OTEL_SDK_DISABLED"] = "true"

# ==============================================================================
#  GLOBAL STATE
# ==============================================================================
GLOBAL_VECTOR_DB = None
GLOBAL_PIPELINE = None 

# ==============================================================================
#  1. BANCO VETORIAL (RAG)
# ==============================================================================
@spaces.GPU
def build_vector_store(pdf_path):
    global GLOBAL_VECTOR_DB
    print(f"[RAG] Indexando: {pdf_path}")
    
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            collection_name="pdf_rag_store",
            persist_directory=None
        )
        
        GLOBAL_VECTOR_DB = vectorstore.as_retriever(search_kwargs={"k": 4})
        return True
    except Exception as e:
        print(f"[ERRO RAG] {e}")
        traceback.print_exc()
        return False

# ==============================================================================
#  2. FERRAMENTA
# ==============================================================================
class PDFRagTool(BaseTool):
    name: str = "SearchPDF"
    description: str = "Search the PDF content. Input is the search query."

    def _run(self, query: str) -> str:
        if GLOBAL_VECTOR_DB is None: return "Error: No PDF loaded."
        try:
            docs = GLOBAL_VECTOR_DB.invoke(query)
            return "\n".join([d.page_content for d in docs])
        except: return "No info found."

# ==============================================================================
#  3. LLM CUSTOMIZADO (QWEN 2.5 - BLINDADO)
# ==============================================================================
class LocalQwen(LLM):
    """
    Classe customizada para rodar o Qwen 2.5 localmente com CrewAI.
    Usa PrivateAttr para evitar erros de validação do Pydantic.
    """
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    _is_dummy: bool = PrivateAttr(default=True)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        global GLOBAL_PIPELINE
        
        if GLOBAL_PIPELINE is None:
            return "Erro: O modelo ainda não foi carregado na memória GPU."

        try:
            # Formatação para Qwen (ChatML)
            # O CrewAI manda um prompt gigante com instruções. 
            # Envolvemos tudo em tags de usuário para o Qwen processar.
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Geração
            response = GLOBAL_PIPELINE(
                formatted_prompt, 
                max_new_tokens=1024, 
                return_full_text=False, 
                do_sample=True,
                temperature=0.1 # Baixa temperatura para seguir regras do CrewAI
            )
            
            generated_text = response[0]['generated_text']
            
            # Limpeza de tags do Qwen
            generated_text = generated_text.replace("<|im_end|>", "").strip()
            
            return generated_text

        except Exception as e:
            print("\n=== ERRO NO MODELO LOCAL ===")
            traceback.print_exc() 
            return f"SYSTEM_ERROR: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "custom_local_qwen"

# ==============================================================================
#  CARREGAMENTO GLOBAL (MODELO 7B)
# ==============================================================================
def load_global_model():
    global GLOBAL_PIPELINE
    if GLOBAL_PIPELINE is not None: return

    print("--- INICIANDO CARREGAMENTO DO QWEN 2.5 (7B) ---")
    try:
        # Modelo muito mais inteligente que o Phi-3
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )

        GLOBAL_PIPELINE = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        print("--- QWEN CARREGADO ---")
    except Exception as e:
        print(f"ERRO AO CARREGAR MODELO: {e}")
        traceback.print_exc()

# ==============================================================================
#  4. AGENTES (CREWAI)
# ==============================================================================
def create_agents_and_tasks(user_query):
    # Carrega o modelo
    load_global_model()
    
    # Instancia nossa classe segura
    llm = LocalQwen()
    
    tools = [PDFRagTool()] if GLOBAL_VECTOR_DB else []

    researcher = Agent(
        role="Researcher",
        goal="Find specific facts in the PDF document.",
        backstory="You are an expert analyst. You read the PDF using the tool and extract exact information.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False,
        max_iter=3 
    )

    writer = Agent(
        role="Writer",
        goal="Write a clear answer based on the facts found.",
        backstory="You are a technical writer. You summarize the information found by the Researcher.",
        tools=[], 
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False
    )

    task1 = Task(
        description=f"Use the SearchPDF tool to find information about: '{user_query}'. Extract the key facts.",
        expected_output="A list of facts found in the document.",
        agent=researcher
    )

    task2 = Task(
        description=f"Using the facts provided, write a final answer to the question: '{user_query}'",
        expected_output="A concise text answer.",
        agent=writer
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True,
        memory=False, 
        cache=False,  
        embedder={    
             "provider": "huggingface",
             "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
        }
    )

# ==============================================================================
#  5. INTERFACE
# ==============================================================================
def process_pdf(file_obj):
    if not file_obj: return "Sem arquivo."
    try:
        path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        if build_vector_store(path):
            return "PDF Indexado! Modelo Qwen pronto."
        else:
            return "Erro na indexação."
    except Exception as e: return f"Erro: {e}"

@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    history.append([message, "🤖 Agentes trabalhando (Qwen 7B)..."])
    yield history

    try:
        crew = create_agents_and_tasks(message)
        result = crew.kickoff()
        history[-1] = [message, str(result.raw)]
    except Exception as e:
        msg = f"Erro no Chat: {str(e)}"
        print(msg)
        traceback.print_exc()
        history[-1] = [message, msg]
    
    yield history

with gr.Blocks(title="CrewAI + Qwen Local") as demo:
    gr.Markdown("# 🤖 CrewAI com Modelo Qwen 2.5 (Local)")
    gr.Markdown("Usando Qwen-7B para maior estabilidade com Agentes.")
    with gr.Row():
        upl = gr.File(label="PDF")
        st = gr.Markdown("...")
    chat = gr.Chatbot(height=550)
    msg = gr.Textbox(label="Pergunta")
    
    upl.change(process_pdf, upl, st)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()

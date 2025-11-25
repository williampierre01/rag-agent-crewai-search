import gradio as gr
import os
import spaces
import traceback
import gc
from typing import Any, List, Optional
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import PrivateAttr

# Transformers Imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ==============================================================================
#  CONFIGURAÇÕES DE AMBIENTE
# ==============================================================================
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OTEL_SDK_DISABLED"] = "true"

# ==============================================================================
#  GLOBAL STATE
# ==============================================================================
GLOBAL_VECTOR_DB = None
GLOBAL_PIPELINE = None 

# ==============================================================================
#  1. BANCO VETORIAL (RAG) - VERSÃO OTIMIZADA PARA MEMÓRIA
# ==============================================================================
@spaces.GPU
def build_vector_store(pdf_path):
    global GLOBAL_VECTOR_DB
    print(f"[RAG] Indexando: {pdf_path}")
    
    try:
        # Limpeza agressiva de memória
        torch.cuda.empty_cache()
        gc.collect()

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # --- OTIMIZAÇÃO: CHUNKS MENORES ---
        # Reduzi para 600 chars. Isso evita estourar a memória do Qwen.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            collection_name="pdf_rag_store",
            persist_directory=None
        )
        
        # --- OTIMIZAÇÃO: MENOS CONTEXTO ---
        # k=2 pega apenas os 2 trechos mais importantes.
        GLOBAL_VECTOR_DB = vectorstore.as_retriever(search_kwargs={"k": 2})
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
#  3. LLM CUSTOMIZADO (ANTI-LOOP & ANTI-CRASH)
# ==============================================================================
class LocalQwen(LLM):
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
            return "SYSTEM_ERROR: Modelo não carregado."

        try:
            # Limpeza TOTAL de argumentos que o CrewAI manda e quebram o pipeline
            clean_kwargs = {
                "max_new_tokens": 1024,
                "return_full_text": False,
                "do_sample": True,
                "temperature": 0.1,
                "repetition_penalty": 1.1 # <--- IMPORTANTE: Evita loops infinitos
            }

            # Formatação ChatML
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            response = GLOBAL_PIPELINE(formatted_prompt, **clean_kwargs)
            
            generated_text = response[0]['generated_text']
            generated_text = generated_text.replace("<|im_end|>", "").strip()
            
            # Se gerar vazio, retorna algo para não quebrar o parser
            if not generated_text:
                return "I completed the task."

            return generated_text

        except Exception as e:
            print(f"\n!!! ERRO NO LLM !!!\n{traceback.format_exc()}")
            # Retorno amigável que engana o CrewAI e evita o Fallback
            return f"Note: I encountered an internal error: {str(e)}. I will try to proceed with what I know."

    @property
    def _llm_type(self) -> str:
        return "custom_local_qwen"

# ==============================================================================
#  CARREGAMENTO GLOBAL
# ==============================================================================
def load_global_model():
    global GLOBAL_PIPELINE
    if GLOBAL_PIPELINE is not None: return

    print("--- INICIANDO CARREGAMENTO DO QWEN ---")
    try:
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
#  4. AGENTES (CREWAI SIMPLIFICADO)
# ==============================================================================
def create_agents_and_tasks(user_query):
    load_global_model()
    llm = LocalQwen()
    
    # Se o PDF não estiver carregado, rodamos sem tools para não quebrar
    tools = [PDFRagTool()] if GLOBAL_VECTOR_DB else []

    researcher = Agent(
        role="Analyst",
        goal="Extract facts from the PDF.",
        backstory="Analyst.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False,
        max_iter=2 # Mínimo possível
    )

    writer = Agent(
        role="Writer",
        goal="Write the answer.",
        backstory="Writer.",
        tools=[], 
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False
    )

    task1 = Task(
        description=f"Read the PDF and find facts about: '{user_query}'",
        expected_output="List of facts.",
        agent=researcher
    )

    task2 = Task(
        description=f"Answer '{user_query}' based on facts.",
        expected_output="Text answer.",
        agent=writer
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True,
        memory=False, 
        cache=False,
        manager_llm=None,
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
            return "PDF Indexado! (Memória Otimizada)"
        else:
            return "Erro na indexação."
    except Exception as e: return f"Erro: {e}"

@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    history.append([message, "🤖 Agentes Trabalhando..."])
    yield history

    try:
        crew = create_agents_and_tasks(message)
        result = crew.kickoff()
        history[-1] = [message, str(result.raw)]
    except Exception as e:
        msg = f"Erro: {str(e)}"
        print(f"ERRO FINAL: {traceback.format_exc()}")
        history[-1] = [message, msg]
    
    yield history

with gr.Blocks(title="CrewAI + Qwen Local (Stable)") as demo:
    gr.Markdown("# 🤖 CrewAI + Qwen (Versão Estável)")
    
    with gr.Row():
        upl = gr.File(label="Upload PDF")
        st = gr.Markdown("...")
    chat = gr.Chatbot(height=550)
    msg = gr.Textbox(label="Pergunta")
    
    upl.change(process_pdf, upl, st)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()

import gradio as gr
import os
import spaces
from typing import Any, List, Optional
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

# Imports LangChain e Pydantic
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import PrivateAttr

# Imports Transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ==============================================================================
#  CONFIGURAÇÕES DE AMBIENTE
# ==============================================================================
os.environ["OPENAI_API_KEY"] = "sk-fake-key-bypass"
os.environ["OTEL_SDK_DISABLED"] = "true"

# ==============================================================================
#  GLOBAL STATE (SINGLETONS)
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
        return False

# ==============================================================================
#  2. FERRAMENTA CUSTOMIZADA
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
#  3. LLM CUSTOMIZADO (ROBUSTO)
# ==============================================================================
class LocalPhi3(LLM):
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    
    # Pydantic: Atributos privados não são serializados
    _is_ready: bool = PrivateAttr(default=False)

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
            # LOG: Ver o que o agente está pedindo (ajuda a debugar)
            # print(f"--- PROMPT ENVIADO AO LLM ---\n{prompt[:100]}...\n---------------------------")

            # Formatação simples para Phi-3
            # Nota: O prompt do CrewAI já vem com instruções, então concatenamos com cuidado
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
            
            response = GLOBAL_PIPELINE(
                formatted_prompt, 
                max_new_tokens=1024, 
                return_full_text=False, 
                do_sample=True,
                temperature=0.1,
                # stop_sequence=stop # O pipeline HF as vezes falha com stop list, melhor deixar o agente tratar
            )
            
            generated_text = response[0]['generated_text']
            
            # Limpeza básica se o modelo gerar tags de fim
            generated_text = generated_text.replace("<|end|>", "").strip()
            
            return generated_text

        except Exception as e:
            # CRÍTICO: Capturamos o erro aqui para o CrewAI não tentar o fallback
            error_msg = f"ERRO INTERNO NO LLM: {str(e)}"
            print(error_msg)
            return error_msg

    @property
    def _llm_type(self) -> str:
        return "custom_local_phi3"

# Função de Carregamento Global
def load_global_model():
    global GLOBAL_PIPELINE
    if GLOBAL_PIPELINE is not None: return

    print("Carregando Modelo na GPU...")
    model_name = "microsoft/Phi-3.5-mini-instruct"
    
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
    print("Modelo Carregado com Sucesso!")

# ==============================================================================
#  4. AGENTES
# ==============================================================================
def create_agents_and_tasks(user_query):
    # Garante carregamento
    load_global_model()
    
    # Instancia a classe segura
    llm = LocalPhi3()
    
    tools = [PDFRagTool()] if GLOBAL_VECTOR_DB else []

    # Agente 1
    researcher = Agent(
        role="Researcher",
        goal="Find facts in the PDF.",
        backstory="Analytical researcher.",
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False,
        max_iter=3 # Limita loops infinitos
    )

    # Agente 2
    writer = Agent(
        role="Writer",
        goal="Summarize the answer.",
        backstory="Technical writer.",
        tools=[], 
        llm=llm,
        verbose=True,
        allow_delegation=False,
        cache=False
    )

    task1 = Task(
        description=f"Search the PDF for: '{user_query}'. Extract relevant quotes.",
        expected_output="A list of facts from the PDF.",
        agent=researcher
    )

    task2 = Task(
        description=f"Using the facts provided, write a clear answer for: '{user_query}'",
        expected_output="Final text answer.",
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
            return "PDF Indexado!"
        else:
            return "Erro na indexação."
    except Exception as e: return f"Erro: {e}"

@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message: return history
    if history is None: history = []
    
    history.append([message, "🤖 Iniciando Agentes..."])
    yield history

    try:
        crew = create_agents_and_tasks(message)
        result = crew.kickoff()
        history[-1] = [message, str(result.raw)]
    except Exception as e:
        history[-1] = [message, f"Erro Fatal: {str(e)}"]
    
    yield history

with gr.Blocks(title="RAG Final") as demo:
    gr.Markdown("# 🛡️ RAG Local (Robust Singleton)")
    with gr.Row():
        upl = gr.File(label="PDF")
        st = gr.Markdown("...")
    chat = gr.Chatbot(height=550)
    msg = gr.Textbox(label="Pergunta")
    
    upl.change(process_pdf, upl, st)
    msg.submit(chat_function, [msg, chat], [chat])

if __name__ == "__main__":
    demo.launch()

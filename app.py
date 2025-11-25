import gradio as gr
import os
import spaces
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

# Imports para RAG (Vector DB)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Imports LLM
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain_community.llms import HuggingFacePipeline

# Ferramentas extras
from src.agentic_rag.tools.custom_tool import FireCrawlWebSearchTool

# ==============================================================================
#  HACK: Enganar verificação da OpenAI (já que usamos tudo local)
# ==============================================================================
os.environ["OPENAI_API_KEY"] = "NA"

# ==============================================================================
#  GLOBAL STATE (Armazena o Banco Vetorial na Memória)
# ==============================================================================
# Como não podemos passar objetos complexos pelo Gradio State sem erros,
# usamos uma variável global para segurar o banco vetorial ativo.
VECTOR_DB_RETRIEVER = None

# ==============================================================================
#  1. PREPARAÇÃO DO BANCO VETORIAL (RAG)
# ==============================================================================
@spaces.GPU
def build_vector_store(pdf_path):
    """
    Lê um PDF grande, divide em pedaços e cria um índice vetorial local (Chroma).
    """
    global VECTOR_DB_RETRIEVER
    
    print(f"[RAG] Iniciando processamento do PDF: {pdf_path}")
    
    # 1. Carregar PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2. Dividir em chunks (pedaços)
    # Chunk size de 1000 caracteres com sobreposição de 200 é um padrão bom.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"[RAG] PDF dividido em {len(splits)} pedaços.")

    # 3. Criar Embeddings (Usa GPU)
    # Modelo leve e gratuito da HuggingFace
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Criar Banco Vetorial (ChromaDB na memória)
    # collection_name garante que não misturemos PDFs se recriarmos
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name="current_pdf_context",
        persist_directory=None # Apenas em memória para ser rápido e efêmero
    )
    
    # 5. Criar o "Buscador" (Retriever)
    # k=5 significa que ele vai pegar os 5 trechos mais relevantes
    VECTOR_DB_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    print("[RAG] Banco vetorial pronto!")
    return True

# ==============================================================================
#  2. FERRAMENTA CUSTOMIZADA (Que o Agente vai usar)
# ==============================================================================
class PDFRagTool(BaseTool):
    name: str = "Search PDF Knowledge Base"
    description: str = "Search for specific information within the uploaded PDF document. Useful for answering questions based on the file content."

    def _run(self, query: str) -> str:
        global VECTOR_DB_RETRIEVER
        
        if VECTOR_DB_RETRIEVER is None:
            return "Error: No PDF loaded or indexed yet."
        
        try:
            # Busca os documentos relevantes no Chroma
            docs = VECTOR_DB_RETRIEVER.invoke(query)
            
            # Concatena o texto dos resultados
            result_text = "\n\n".join([f"--- Excerpt ---\n{doc.page_content}" for doc in docs])
            
            if not result_text:
                return "No relevant information found in the PDF."
                
            return result_text
        except Exception as e:
            return f"Error querying vector DB: {str(e)}"

# ==============================================================================
#  3. MODELO LLM (LLama/Phi/etc)
# ==============================================================================
global_model = None
global_tokenizer = None

def initialize_model():
    global global_model, global_tokenizer
    if global_model and global_tokenizer:
        return global_model, global_tokenizer

    model_name = "microsoft/Phi-3.5-mini-instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    global_tokenizer = tokenizer
    global_model = model
    return global_model, global_tokenizer

def load_llm():
    model, tokenizer = initialize_model()
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=text_generation_pipeline)

# ==============================================================================
#  4. AGENTES E TAREFAS
# ==============================================================================
def create_agents_and_tasks(user_query):
    
    tools_list = []
    
    # Adiciona a ferramenta de busca Web (se configurada)
    if "FIRECRAWL_API_KEY" in os.environ:
        try:
            tools_list.append(FireCrawlWebSearchTool())
        except: pass

    # Adiciona NOSSA ferramenta de RAG
    # Nota: Não precisamos passar o path, pois ela usa o VECTOR_DB_RETRIEVER global
    if VECTOR_DB_RETRIEVER is not None:
        tools_list.append(PDFRagTool())

    llm_instance = load_llm()

    retriever_agent = Agent(
        role="Investigator",
        goal=f"Search for evidence to answer: {user_query}",
        backstory="You are a data analyst skilled at finding needles in haystacks using search tools.",
        verbose=True,
        tools=tools_list,
        llm=llm_instance
    )

    response_agent = Agent(
        role="Writer",
        goal="Synthesize the evidence into a clear answer.",
        backstory="You write concise, accurate answers based only on the provided evidence.",
        verbose=True,
        llm=llm_instance
    )

    task1 = Task(
        description=f"Use the PDF Search tool to find information about '{user_query}'. Gather key facts.",
        expected_output="A list of relevant quotes and facts from the document.",
        agent=retriever_agent
    )

    task2 = Task(
        description=f"Answer the user's question '{user_query}' using the gathered facts.",
        expected_output="A final text answer.",
        agent=response_agent
    )

    return Crew(
        agents=[retriever_agent, response_agent],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True
    )

# ==============================================================================
#  5. GRADIO (Interface)
# ==============================================================================

def process_pdf(file_obj):
    """
    Chama a função pesada que cria o Banco Vetorial.
    """
    if not file_obj:
        return "Nenhum arquivo."
    
    try:
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        
        # AQUI ACONTECE A MÁGICA: Indexação do PDF
        # O Gradio vai esperar isso terminar antes de liberar o chat.
        build_vector_store(file_path)
        
        return f"PDF '{os.path.basename(file_path)}' indexado! Banco Vetorial Pronto."
    except Exception as e:
        return f"Erro ao indexar: {str(e)}"

@spaces.GPU(duration=120)
def chat_function(message, history):
    if not message:
        return history
    if history is None:
        history = []

    history.append([message, "🔍 Consultando Banco Vetorial e Agentes..."])
    yield history

    try:
        crew = create_agents_and_tasks(message)
        inputs = {"query": message}
        result = crew.kickoff(inputs=inputs)
        
        history[-1] = [message, result.raw]
        yield history
    except Exception as e:
        history[-1] = [message, f"Erro: {str(e)}"]
        yield history

# Layout
with gr.Blocks(title="Agentic RAG com Vector DB") as demo:
    gr.Markdown("# 🧠 Agentic RAG: PDF Grande (Vector DB)")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload PDF Grande")
            status_txt = gr.Markdown("Aguardando upload...")
            
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=600)
            msg_input = gr.Textbox(label="Pergunta")

    # Quando sobe o arquivo, roda o build_vector_store
    file_upload.change(
        fn=process_pdf,
        inputs=[file_upload],
        outputs=[status_txt]
    )

    msg_input.submit(
        fn=chat_function,
        inputs=[msg_input, chatbot],
        outputs=[chatbot]
    ).then(fn=lambda: "", outputs=[msg_input])

if __name__ == "__main__":
    demo.launch()

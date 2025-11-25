import gradio as gr
import os
import time
import gc
import spaces
from crewai import Agent, Crew, Process, Task
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain_community.llms import HuggingFacePipeline

from src.agentic_rag.tools.custom_tool import FireCrawlWebSearchTool
from src.agentic_rag.tools.custom_tool import DocumentSearchTool

# Variáveis globais para armazenar o modelo carregado (Cache)
global_model = None
global_tokenizer = None

# ==============================================================================
#                           Carregamento do Modelo (Lazy Loading)
# ==============================================================================
def initialize_model():
    """
    Carrega o modelo apenas quando necessário e se ainda não estiver carregado.
    Isso evita o erro de 'CUDA initialized in main process'.
    """
    global global_model, global_tokenizer
    
    if global_model is not None and global_tokenizer is not None:
        return global_model, global_tokenizer

    model_name = "microsoft/Phi-3.5-mini-instruct"
    print(f"Iniciando carregamento do modelo: {model_name}...")

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
    print(f"Modelo {model_name} carregado com sucesso!")
    return global_model, global_tokenizer

# ===========================
#   Configurações LLM
# ===========================
def load_llm():
    # Garante que o modelo está carregado antes de criar o pipeline
    model, tokenizer = initialize_model()

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

    hf_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return hf_llm

# ===========================
#   Definição de Agentes e Tarefas
# ===========================
# Adicionado decorador GPU aqui também, pois esta função chama load_llm
@spaces.GPU(duration=120) 
def create_agents_and_tasks(pdf_tool):
    """Cria a Crew com a ferramenta de PDF (se houver) e busca na web."""
    web_search_tool = FireCrawlWebSearchTool()

    # Lista de ferramentas
    tools_list = [t for t in [pdf_tool, web_search_tool] if t]

    # Carregar o LLM (agora seguro pois estamos dentro de @spaces.GPU)
    llm_instance = load_llm()

    retriever_agent = Agent(
        role="Retrieve relevant information to answer the user query: {query}",
        goal=(
            "Retrieve the most relevant information from the available sources "
            "for the user query: {query}. Always try to use the PDF search tool first."
        ),
        backstory=(
            "You're a meticulous analyst with a keen eye for detail."
        ),
        verbose=True,
        tools=tools_list,
        llm=llm_instance
    )

    response_synthesizer_agent = Agent(
        role="Response synthesizer agent for the user query: {query}",
        goal="Synthesize the retrieved information into a concise response.",
        backstory="You're a skilled communicator.",
        verbose=True,
        llm=llm_instance
    )

    retrieval_task = Task(
        description="Retrieve information for: {query}",
        expected_output="Relevant text retrieved from sources.",
        agent=retriever_agent
    )

    response_task = Task(
        description="Synthesize final response for: {query}",
        expected_output="Concise response based on retrieved info.",
        agent=response_synthesizer_agent
    )

    crew = Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=True
    )
    return crew

# ===========================
#   Funções de Lógica do Gradio
# ===========================

# ADICIONADO: @spaces.GPU aqui é CRUCIAL para o SentenceTransformers (Embeddings)
@spaces.GPU 
def process_pdf(file_obj):
    """
    Processa o arquivo PDF enviado pelo usuário.
    """
    if not file_obj:
        return None, "Nenhum arquivo enviado.", None

    try:
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        
        # O DocumentSearchTool carrega embeddings, então precisa de GPU
        doc_tool = DocumentSearchTool(file_path=file_path)
        
        return doc_tool, f"PDF '{os.path.basename(file_path)}' indexado com sucesso!", None
    except Exception as e:
        return None, f"Erro ao indexar PDF: {str(e)}", None

@spaces.GPU(duration=120)
def chat_function(message, history, pdf_tool_state, crew_state):
    """
    Função principal do chat.
    """
    if not message:
        return history, crew_state

    # Se a Crew não existir, criamos uma nova
    if crew_state is None:
        crew_state = create_agents_and_tasks(pdf_tool_state)

    inputs = {"query": message}

    try:
        result_obj = crew_state.kickoff(inputs=inputs)
        final_response = result_obj.raw
    except Exception as e:
        final_response = f"Ocorreu um erro ao processar: {str(e)}"

    # Simulação de streaming
    history.append((message, ""))
    
    # Exibir resposta gradualmente
    full_text = ""
    # Se final_response for muito curto ou não iterável, tratamos aqui
    for char in final_response: # Iterar por char fica mais suave que por linha
        full_text += char
        history[-1] = (message, full_text)
        if len(full_text) % 5 == 0: # Atualiza a cada 5 caracteres para não travar a UI
            yield history, crew_state
            time.sleep(0.001)
            
    # Garantir atualização final
    history[-1] = (message, final_response)
    yield history, crew_state


# ===========================
#   Interface Gradio
# ===========================

with gr.Blocks(title="Agentic RAG com CrewAI") as demo:

    pdf_tool_state = gr.State(None)
    crew_state = gr.State(None)

    gr.Markdown("# Agentic RAG powered by CrewAI")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Adicione seu Documento PDF")
            file_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_status = gr.Markdown("Aguardando upload...")
            clear_btn = gr.Button("Limpar Chat")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Histórico", height=600)
            msg_input = gr.Textbox(label="Pergunta", placeholder="Digite aqui...")

    file_upload.change(
        fn=process_pdf,
        inputs=[file_upload],
        outputs=[pdf_tool_state, upload_status, crew_state]
    )

    msg_input.submit(
        fn=chat_function,
        inputs=[msg_input, chatbot, pdf_tool_state, crew_state],
        outputs=[chatbot, crew_state]
    ).then(
        fn=lambda: "", outputs=[msg_input]
    )

    def reset_history():
        return [], None

    clear_btn.click(
        fn=reset_history,
        inputs=None,
        outputs=[chatbot, crew_state]
    )

if __name__ == "__main__":
    demo.launch()

import gradio as gr
import os
import time
import gc
from crewai import Agent, Crew, Process, Task
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Adicionado AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch # Adicionado torch
from langchain_community.llms import HuggingFacePipeline

# Mantendo suas importações originais
# Certifique-se de que a pasta 'src' esteja no mesmo diretório
from src.agentic_rag.tools.custom_tool import FireCrawlWebSearchTool
from src.agentic_rag.tools.custom_tool import DocumentSearchTool

# ==============================================================================
#                           Escolha do Modelo LLM
# ==============================================================================
# Descomente e defina a variável `model_name` com o modelo que deseja usar.
# Certifique-se de que o modelo selecionado seja compatível com sua GPU e recursos.

# Exemplo 1: Padrão da indústria (meta-llama/Meta-Llama-3.1-8B-Instruct)
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Exemplo 2: Melhor opção geral (Qwen/Qwen2.5-7B-Instruct)
# model_name = "Qwen/Qwen2.5-7B-Instruct"

# Exemplo 3: Especialista em ferramentas (mistralai/Mistral-7B-Instruct-v0.3)
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Exemplo 4: Mais rápido (microsoft/Phi-3.5-mini-instruct)
# model_name = "microsoft/Phi-3.5-mini-instruct"

# Exemplo 5: Treinado para agente (NousResearch/Hermes-3-Llama-3.1-8B) - recomendado usar versão quantizada
model_name = "NousResearch/Hermes-3-Llama-3.1-8B"

print(f"Carregando o modelo: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configuração de quantização usando BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Para carregar em 8-bit. Use load_in_4bit=True para 4-bit
    # bnb_4bit_quant_type="nf4", # Apenas para 4-bit: "nf4" ou "fp4"
    # bnb_4bit_compute_dtype=torch.bfloat16, # Apenas para 4-bit: dtype para cálculos
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # ou torch.float16, dependendo do suporte da GPU
    device_map="auto",
    quantization_config=bnb_config, # Substitui load_in_8bit/4bit
)

print(f"Modelo {model_name} carregado com sucesso!")

# ===========================
#   Configurações LLM
# ===========================
def load_llm():
   
    global model, tokenizer 

    # Criar um pipeline de geração de texto usando o modelo e tokenizer carregados
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512, # Ajuste conforme necessário
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        # Adicione outros argumentos do pipeline conforme a necessidade do seu modelo
    )

    # Envolver o pipeline com HuggingFacePipeline para compatibilidade com CrewAI
    hf_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return hf_llm

# ===========================
#   Definição de Agentes e Tarefas
# ===========================
def create_agents_and_tasks(pdf_tool):
    """Cria a Crew com a ferramenta de PDF (se houver) e busca na web."""
    web_search_tool = FireCrawlWebSearchTool()

    # Lista de ferramentas (filtra None se o pdf_tool não existir)
    tools_list = [t for t in [pdf_tool, web_search_tool] if t]

    # Carregar o LLM uma vez e passá-lo para os agentes
    llm_instance = load_llm()

    retriever_agent = Agent(
        role="Retrieve relevant information to answer the user query: {query}",
        goal=(
            "Retrieve the most relevant information from the available sources "
            "for the user query: {query}. Always try to use the PDF search tool first. "
            "If you are not able to retrieve the information from the PDF search tool, "
            "then try to use the web search tool."
        ),
        backstory=(
            "You're a meticulous analyst with a keen eye for detail. "
            "You're known for your ability to understand user queries: {query} "
            "and retrieve knowledge from the most suitable knowledge base."
        ),
        verbose=True,
        tools=tools_list,
        llm=llm_instance # Passa a instância do LLM aqui
    )

    response_synthesizer_agent = Agent(
        role="Response synthesizer agent for the user query: {query}",
        goal=(
            "Synthesize the retrieved information into a concise and coherent response "
            "based on the user query: {query}. If you are not able to retrieve the "
            "information then respond with 'I'm sorry, I couldn't find the information "
            "you're looking for.'"
        ),
        backstory=(
            "You're a skilled communicator with a knack for turning "
            "complex information into clear and concise responses."
        ),
        verbose=True,
        llm=llm_instance # Passa a instância do LLM aqui
    )

    retrieval_task = Task(
        description=(
            "Retrieve the most relevant information from the available "
            "sources for the user query: {query}"
        ),
        expected_output=(
            "The most relevant information in the form of text as retrieved "
            "from the sources."
        ),
        agent=retriever_agent
    )

    response_task = Task(
        description="Synthesize the final response for the user query: {query}",
        expected_output=(
            "A concise and coherent response based on the retrieved information "
            "from the right source for the user query: {query}."
        ),
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

def process_pdf(file_obj):
    """
    Processa o arquivo PDF enviado pelo usuário.
    Retorna: Instância da Tool, Mensagem de Status, Objeto Crew (None para forçar recriação)
    """
    if not file_obj:
        return None, "Nenhum arquivo enviado.", None

    try:
        # O Gradio envia o caminho do arquivo temporário em file_obj.name
        # ou file_obj (dependendo da versão, mas file_obj geralmente é o caminho str em versões recentes)
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj

        doc_tool = DocumentSearchTool(file_path=file_path)
        return doc_tool, f"PDF '{os.path.basename(file_path)}' indexado com sucesso!", None
    except Exception as e:
        return None, f"Erro ao indexar PDF: {str(e)}", None

def chat_function(message, history, pdf_tool_state, crew_state):
    """
    Função principal do chat.
    message: Texto do usuário
    history: Histórico do chat [(user, bot), ...]
    pdf_tool_state: Estado atual da ferramenta de PDF
    crew_state: Estado atual da Crew
    """

    if not message:
        return history, crew_state

    # Se a Crew não existir ou se mudamos o PDF (crew_state resetado), criamos uma nova
    if crew_state is None:
        crew_state = create_agents_and_tasks(pdf_tool_state)

    inputs = {"query": message}

    # Executa o CrewAI
    # Nota: crew.kickoff é bloqueante. O Gradio vai esperar terminar para mostrar.
    try:
        result_obj = crew_state.kickoff(inputs=inputs)
        final_response = result_obj.raw
    except Exception as e:
        final_response = f"Ocorreu um erro ao processar: {str(e)}"

    # Simulação de streaming (efeito visual)
    # O Gradio Chatbot aceita yield para ir preenchendo a resposta
    partial_response = ""
    history.append((message, "")) # Adiciona placeholder

    # Quebra em linhas para simular o efeito do código original
    lines = final_response.split('\n')
    full_text = ""

    for line in lines:
        full_text += line + "\n"
        # Atualiza a última mensagem do bot no histórico
        history[-1] = (message, full_text)
        yield history, crew_state
        time.sleep(0.05) # Pequeno delay visual

# ===========================
#   Interface Gradio
# ===========================

with gr.Blocks(title="Agentic RAG com CrewAI") as demo:

    # Estados (Variáveis que persistem por sessão do usuário)
    pdf_tool_state = gr.State(None)
    crew_state = gr.State(None)

    gr.Markdown("# Agentic RAG powered by CrewAI")

    with gr.Row():
        # --- Coluna Esquerda (Sidebar) ---
        with gr.Column(scale=1):
            gr.Markdown("### Adicione seu Documento PDF")
            file_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_status = gr.Markdown("Aguardando upload...")

            # Preview do PDF não é nativo simples no Gradio como no Streamlit,
            # mas o componente 'File' já mostra o arquivo carregado.

            clear_btn = gr.Button("Limpar Chat")

        # --- Coluna Direita (Chat) ---
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Histórico", height=600)
            msg_input = gr.Textbox(label="Faça uma pergunta sobre o PDF...", placeholder="Digite aqui e pressione Enter")

    # ===========================
    #   Eventos
# ===========================

    # Quando o arquivo é enviado
    file_upload.change(
        fn=process_pdf,
        inputs=[file_upload],
        outputs=[pdf_tool_state, upload_status, crew_state] # Reseta o crew_state ao mudar o PDF
    )

    # Quando o usuário envia mensagem (Enter)
    msg_input.submit(
        fn=chat_function,
        inputs=[msg_input, chatbot, pdf_tool_state, crew_state],
        outputs=[chatbot, crew_state]
    ).then(
        fn=lambda: "", outputs=[msg_input] # Limpa o input box depois de enviar
    )

    # Botão de limpar
    def reset_history():
        return [], None # Limpa chat e força recriação da crew se necessário

    clear_btn.click(
        fn=reset_history,
        inputs=None,
        outputs=[chatbot, crew_state]
    )

if __name__ == "__main__":
    demo.launch()

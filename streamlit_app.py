import os, base64
from pathlib import Path
import hmac
import tempfile
import pandas as pd
import uuid

import streamlit as st


from langchain_community.vectorstores import AstraDB
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import AstraDBChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema import StrOutputParser

from langchain.callbacks.base import BaseCallbackHandler

import openai

st.set_page_config(page_title=" ", page_icon='./customizations/logo/anim-logo-1fps-verde.gif', layout="wide")

# ▼▼▼ ESTE ES EL BLOQUE QUE NECESITAS AÑADIR ▼▼▼
# --- INICIALIZACIÓN DE SESSION STATE ---
# Esto asegura que las variables de sesión existan desde el principio.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
# ▲▲▲ FIN DEL BLOQUE A AÑADIR ▲▲▲


# --- CONFIGURACIÓN GLOBAL ---
ASTRA_DB_COLLECTION_NAME = "vc_assistant"
ADMIN_USERS = ["openlab_admin"] 

print("Streamlit App Started")


# --- INICIALIZACIÓN DE SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "debug_messages" not in st.session_state:
    st.session_state.debug_messages = []


# --- CLASES Y FUNCIONES ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

def check_password():
    def login_form():
        with st.form("credentials"):
            st.text_input('Username', key='username')
            st.text_input('Password', type='password', key='password')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        if 'passwords' in st.secrets and st.session_state.get('username') in st.secrets['passwords'] and hmac.compare_digest(st.session_state.get('password', ''), st.secrets.passwords[st.session_state.get('username', '')]):
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
            if 'password' in st.session_state: del st.session_state['password']
        else:
            st.session_state['password_correct'] = False

    if st.session_state.get('password_correct', False):
        return True
    login_form()
    if "password_correct" in st.session_state and not st.session_state['password_correct']:
        st.error('😕 Usuario desconocido o contraseña incorrecta')
    return False

def logout():
    keys_to_delete = list(st.session_state.keys())
    for key in keys_to_delete: del st.session_state[key]
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

def vectorize_text(uploaded_files, vectorstore, lang_dict):
    if not vectorstore:
        st.error(lang_dict.get('vectorstore_not_ready_admin', "Vectorstore not ready for upload."))
        return
    with st.spinner("Processing files... This may take a moment."):
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                st.session_state.debug_messages.append(f"Admin: Processing file {uploaded_file.name}")
                try:
                    docs = []
                    if uploaded_file.name.endswith('.pdf'):
                        loader = PyPDFLoader(tmp_file_path)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["source"] = uploaded_file.name
                    elif uploaded_file.name.endswith('.csv'):
                        loader = CSVLoader(tmp_file_path, encoding='utf-8')
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["source"] = uploaded_file.name
                    elif uploaded_file.name.endswith('.txt'):
                        from langchain.schema import Document
                        with open(tmp_file_path, 'r', encoding='utf-8') as f_txt:
                            docs = [Document(page_content=f_txt.read(), metadata={"source": uploaded_file.name})]
                    else:
                        st.warning(f"Unsupported file type: {uploaded_file.name}")
                        continue
                    
                    if docs:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                        pages = text_splitter.split_documents(docs)
                        vectorstore.add_documents(pages)
                        st.info(f"✅ {uploaded_file.name} processed ({len(pages)} segments).")
                except Exception as e:
                    st.error(f"Error processing file {uploaded_file.name}: {e}")
                finally:
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

def vectorize_url(urls, vectorstore, lang_dict):
    if not vectorstore:
        st.error(lang_dict.get('vectorstore_not_ready_admin', "Vectorstore not ready for URL load."))
        return
    with st.spinner("Processing URLs..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        for url in urls:
            url = url.strip()
            if not url: continue
            try:
                st.session_state.debug_messages.append(f"Admin: Loading from URL: {url}")
                loader = WebBaseLoader(url)
                docs = loader.load()
                pages = text_splitter.split_documents(docs)
                vectorstore.add_documents(pages)
                st.info(f"✅ URL processed: {url}")
            except Exception as e:
                st.error(f"Error loading from URL {url}: {e}")

def generate_follow_up_question(question, answer, model):
    """
    Hace una segunda llamada a la IA para generar una pregunta de seguimiento.
    """
    prompt_template = """Basado en la pregunta original del usuario y la respuesta que ha dado la IA, genera una única y concisa pregunta de seguimiento para invitar al usuario a profundizar.
Devuelve únicamente el texto de la pregunta, sin saludos, prefijos ni nada más.

Pregunta del Usuario: "{user_question}"

Respuesta de la IA: "{ai_answer}"

Pregunta de Seguimiento Sugerida:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Creamos una cadena simple solo para esta tarea
    chain = prompt | model | StrOutputParser()
    
    try:
        # Usamos .invoke() porque no necesitamos streaming para esta llamada corta
        suggested_question = chain.invoke({
            "user_question": question,
            "ai_answer": answer,
        })
        return suggested_question
    except Exception:
        return None # Si algo falla, simplemente no devolvemos nada

def get_prompt(type, custom_prompt, language):
    base_template = f"""Use the following context to answer the question:
{{context}}

Use the following chat history to answer the question:
{{chat_history}}

Question:
{{question}}

Answer in {language}:"""
    
    if type == 'Extended results':
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. You prefer to use bulletpoints to summarize.
If you don't know the answer, just say 'I do not know the answer'.
{base_template}"""
    elif type == 'Short results':
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
You answer in an exceptionally brief way.
If you don't know the answer, just say 'I do not know the answer'.
{base_template}"""
    else: # 'Custom'
        template = custom_prompt if custom_prompt else base_template
    
    return ChatPromptTemplate.from_messages([("system", template)])

def load_retriever(vectorstore, top_k_vectorstore):
    print(f"""load_retriever with top_k_vectorstore='{top_k_vectorstore}'""")
    return vectorstore.as_retriever(search_kwargs={"k": top_k_vectorstore})

def generate_queries(model, language):
    prompt = f"""You are a helpful assistant that generates multiple search queries based on a single input query in language {language}.
Generate multiple search queries related to: {{original_query}}
OUTPUT (4 queries):"""
    return ChatPromptTemplate.from_messages([("system", prompt)]) | model | StrOutputParser() | (lambda x: x.split("\n"))

def reciprocal_rank_fusion(results: list[list], k=60):
    from langchain.load import dumps, loads
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            try:
                doc_str = dumps(doc)
                if doc_str not in fused_scores: fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)
            except Exception: continue
    return [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]


# --- Funciones de Carga de Recursos Cacheadas ---
@st.cache_data()
def load_localization(locale):
    try:
        df = pd.read_csv("./customizations/localization.csv", encoding='utf-8')
        df_lang = df.query(f"locale == '{locale}'")
        if df_lang.empty: df_lang = df.query("locale == 'en_US'")
        return pd.Series(df_lang.value.values, index=df_lang.key).to_dict()
    except Exception: return {"assistant_welcome": "Welcome (localization file not found)."}

@st.cache_data()
def load_rails(username):
    try:
        df = pd.read_csv("./customizations/rails.csv", encoding='utf-8')
        df_user = df.query(f"username == '{username}'")
        return pd.Series(df_user.value.values, index=df_user.key).to_dict()
    except Exception: return {}

@st.cache_resource(show_spinner="Loading embeddings...")
def load_embedding_rc():
    return OpenAIEmbeddings(openai_api_key=st.secrets.get("OPENAI_API_KEY"))

@st.cache_resource(show_spinner="Loading vector store...")
def load_vectorstore_rc(username, _embedding): # Underscore to prevent hashing
    return AstraDB(
        embedding=_embedding,
        collection_name=ASTRA_DB_COLLECTION_NAME,
        token=st.secrets["ASTRA_TOKEN"],
        api_endpoint=st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT")),
        namespace=st.secrets.get("ASTRA_KEYSPACE")
    )

@st.cache_resource(show_spinner="Loading chat history...")
def load_chat_history_rc(username, session_id):
    return AstraDBChatMessageHistory(
        session_id=f"{username}_{str(session_id)}",
        api_endpoint=st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT")),
        token=st.secrets["ASTRA_TOKEN"],
        collection_name="historial_chat_asistente"
    )

@st.cache_resource(show_spinner="Loading AI model...")
def load_model_rc():
    return ChatOpenAI(temperature=0.3, model='gpt-4o', streaming=True, verbose=False)

@st.cache_resource()
def load_memory_rc(_chat_history, top_k_history):
    if not _chat_history: return None
    return ConversationBufferWindowMemory(
        chat_memory=_chat_history, return_messages=True, k=top_k_history,
        memory_key="chat_history", input_key="question", output_key='answer')

# --- FLUJO PRINCIPAL DE LA APP ---

# 1. Login
if not check_password():
    st.stop()

# 2. Inicialización Post-Login
username = st.session_state.user
language = st.secrets.get("languages", {}).get(username, "es_ES")
lang_dict = load_localization(language)

embedding = load_embedding_rc()
vectorstore = load_vectorstore_rc(username, embedding) if embedding else None
chat_history = load_chat_history_rc(username, st.session_state.session_id) if vectorstore else None
model = load_model_rc()

# 3. Configuración de Parámetros (Condicional según el usuario)
if username != "demo":
    with st.sidebar:
        # Logo, Logout, Rails...
        st.image('./customizations/logo/default.svg', use_column_width="always")
        st.markdown(f"""{lang_dict.get('logout_caption','Logged in as')} :orange[{username}]""")
        if st.button(lang_dict.get('logout_button','Logout')): logout()
        st.divider()

        rails_dict = load_rails(username)
        st.subheader(lang_dict.get('rails_1', "Suggestions"))
        st.caption(lang_dict.get('rails_2', "Try asking:"))
        if rails_dict:
            for i in sorted(rails_dict.keys()): st.markdown(f"{i}. {rails_dict[i]}")
        st.divider()

        # Opciones de Chat
        st.subheader(lang_dict.get('options_header', "Chat Options"))
        
        user_defaults = st.secrets.get("DEFAULT_SETTINGS", {}).get(username, {})
        
        disable_chat_history = st.toggle(lang_dict.get('disable_chat_history', "Disable Chat History"), value=False)
        top_k_history = st.slider(lang_dict.get('k_chat_history', "K for Chat History"), 1, 10, user_defaults.get("TOP_K_HISTORY", 5), disabled=disable_chat_history)
        disable_vector_store = st.toggle(lang_dict.get('disable_vector_store', "Disable Vector Store?"), value=False)
        top_k_vectorstore = st.slider(lang_dict.get('top_k_vectorstore', "Top-K for Vector Store"), 1, 10, user_defaults.get("TOP_K_VECTORSTORE", 5), disabled=disable_vector_store)
        
        rag_strategies = ('Basic Retrieval', 'Maximal Marginal Relevance', 'Fusion')
        default_strategy = user_defaults.get("RAG_STRATEGY", 'Basic Retrieval')
        strategy_idx = rag_strategies.index(default_strategy) if default_strategy in rag_strategies else 0
        strategy = st.selectbox(lang_dict.get('rag_strategy', "RAG Strategy"), rag_strategies, index=strategy_idx, disabled=disable_vector_store)
        
        prompt_options = ('Short results', 'Extended results', 'Custom')
        default_prompt_type = user_defaults.get("PROMPT_TYPE", 'Custom')
        prompt_idx = prompt_options.index(default_prompt_type) if default_prompt_type in prompt_options else 2

        try:
            custom_prompt_text_val = Path(f"./customizations/prompt/{username}.txt").read_text(encoding='utf-8')
        except:
            try:
                custom_prompt_text_val = Path("./customizations/prompt/default.txt").read_text(encoding='utf-8')
            except:
                custom_prompt_text_val = "Error: No se encontró el archivo de prompt default.txt"

        prompt_type = st.selectbox(lang_dict.get('system_prompt', "System Prompt"), prompt_options, index=prompt_idx)
        custom_prompt = st.text_area(lang_dict.get('custom_prompt', "Custom Prompt"), value=custom_prompt_text_val, disabled=(prompt_type != 'Custom'))
        st.divider()

        # Herramientas de Administración
        if username in ADMIN_USERS:
            st.subheader(lang_dict.get('admin_tools_header', "Admin Tools"))
            with st.expander("Upload Files"):
                uploaded_files = st.file_uploader("Upload TXT, PDF, CSV files", type=['txt', 'pdf', 'csv'], accept_multiple_files=True)
                if st.button("Process Files"):
                    if uploaded_files: vectorize_text(uploaded_files, vectorstore, lang_dict)
            with st.expander("Load from URLs"):
                urls_text = st.text_area("Enter URLs (comma-separated)")
                if st.button("Process URLs"):
                    urls = [url.strip() for url in urls_text.split(',') if url.strip()]
                    if urls: vectorize_url(urls, vectorstore, lang_dict)
else: # Si el usuario es 'demo'
    # Inyectar CSS para ocultar la sidebar
    st.markdown("""<style>[data-testid="stSidebar"] {display: none}</style>""", unsafe_allow_html=True)
    
    # Definir valores por defecto para que la app no falle
    user_defaults = st.secrets.get("DEFAULT_SETTINGS", {}).get(username, {})
    disable_chat_history = True
    top_k_history = 0
    disable_vector_store = False
    top_k_vectorstore = user_defaults.get("TOP_K_VECTORSTORE", 5)
    strategy = user_defaults.get("RAG_STRATEGY", 'Basic Retrieval')
    prompt_type = user_defaults.get("PROMPT_TYPE", 'Custom')
    try:
        # Intenta cargar el prompt para 'demo' o el default
        user_prompt_file = Path(f"./customizations/prompt/demo.txt")
        if user_prompt_file.is_file():
            custom_prompt = user_prompt_file.read_text(encoding='utf-8')
        else:
            custom_prompt = Path("./customizations/prompt/default.txt").read_text(encoding='utf-8')
    except:
        custom_prompt = "Responde basándote en el contexto proporcionado."

# Inicializar memoria
memory = load_memory_rc(chat_history, top_k_history if not disable_chat_history else 0) if chat_history else None

# --- COPIA Y PEGA ESTE BLOQUE COMPLETO HASTA EL FINAL DEL ARCHIVO ---

# --- Interfaz Principal del Chat (Visible para TODOS los usuarios) ---

# 1. Inyectamos el CSS final para centrar el layout y el logo
st.markdown("""
    <style>
        /* Contenedor principal para los mensajes y el encabezado */
        section[data-testid="st.main"] .block-container {
            max-width: 55% !important; /* <-- AÑADIDO !important PARA FORZAR LA REGLA */
            margin: 0 auto !important;
        }
        /* Contenedor del campo de texto del chat */
        [data-testid="stChatInputContainer"] {
            max-width: 55% !important; /* <-- AÑADIDO !important PARA FORZAR LA REGLA */
            margin: 0 auto !important;
        }
    </style>
""", unsafe_allow_html=True)


# 2. Definimos y mostramos el encabezado de forma robusta
# Función para codificar la imagen a base64 (así la podemos meter en el HTML)
import base64
def get_image_as_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

# Obtenemos la imagen del logo en base64
logo_base64 = get_image_as_base64("./customizations/logo/anim-logo-1fps-verde.gif")

# Un solo bloque de HTML para todo el encabezado, asegurando el centrado
st.markdown(f"""
    <div style="text-align: center;">
        <img src="data:image/gif;base64,{logo_base64}" alt="Logo" width="150">
        <h1>Agente Experto IA para Fondos</h1>
        <p>Por OPENLAB VENTURES, S.L. ®</p>
        <p style="color: #9c9d9f; font-size: 0.9rem;">Tu consultor virtual especializado en la introducción estratégica de la Inteligencia Artificial en los procesos internos de Venture Capital y Private Equity.</p>
    </div>
""", unsafe_allow_html=True)

st.divider()


# 3. Lógica de visualización del chat
# Mensajes de bienvenida y botones de sugerencia (si es una sesión nueva)
if not st.session_state.messages:
    st.info(lang_dict.get('assistant_welcome', "¡Hola! ¿En qué puedo ayudarte hoy?"))
    rails_dict = load_rails(username)
    if rails_dict:
        st.write("O intenta con alguna de estas preguntas:")
        cols = st.columns(len(rails_dict) if len(rails_dict) <= 4 else 4)
        for i, (key, value) in enumerate(list(rails_dict.items())[:4]):
            if cols[i].button(value, key=f"rail_{key}"):
                st.session_state.question_from_button = value
                st.rerun()

# Mostrar todo el historial de chat en cada ejecución
for message in st.session_state.messages:
    avatar_icon = "🤖" if message.type == "ai" else "🧑‍💻"
    with st.chat_message(message.type, avatar=avatar_icon):
        st.markdown(message.content)
        # Si la respuesta es de la IA y tiene fuentes, las mostramos
        # if message.type == "ai" and message.additional_kwargs.get("sources"):
        #    with st.expander("Ver fuentes utilizadas"):
        #        for i, doc in enumerate(message.additional_kwargs["sources"]):
        #            source = doc.metadata.get('source', 'N/A')
        #            content_preview = doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
        #            st.info(f"**Fuente {i+1}:** `{source}`")
        #            st.write(content_preview)

# 5. Lógica para mostrar la pregunta sugerida como un botón
if "suggested_question" in st.session_state and st.session_state.suggested_question:
    # Usamos st.button para crear el botón. Si se pulsa, se ejecuta el bloque.
    if st.button(f"Sugerencia: *{st.session_state.suggested_question}*"):
        # Preparamos la pregunta para que el siguiente ciclo la procese
        st.session_state.question_from_button = st.session_state.suggested_question
        # Limpiamos la sugerencia para que no vuelva a aparecer
        del st.session_state.suggested_question 
        st.rerun()

# 4. Lógica para recibir y procesar una nueva pregunta
# Se captura la pregunta, ya sea de un botón o del campo de texto
question = st.session_state.pop("question_from_button", None)
if not question:
    # Usamos el operador walrus (:=) para asignar y comprobar en la misma línea
    if user_query := st.chat_input(lang_dict.get('assistant_question', "Pregunta lo que quieras...")):
        question = user_query

# Si tenemos una pregunta válida, la procesamos
if question:
    # Añadir el mensaje del usuario al historial y mostrarlo en la UI
    st.session_state.messages.append(HumanMessage(content=question))
    with st.chat_message('human', avatar="🧑‍💻"):
        st.markdown(question)

    # Preparar y ejecutar la cadena RAG para obtener la respuesta de la IA
    with st.chat_message('assistant', avatar="🤖"):
        response_placeholder = st.empty()
        
        # Comprobación de que los componentes de la IA están listos
        if not model or (not disable_vector_store and not vectorstore):
            response_placeholder.markdown("Lo siento, el asistente no está completamente configurado.")
        else:
            # Este bloque 'else' asegura que todo lo de abajo solo se ejecuta si la IA está lista
            
            # Paso 1: Recuperar documentos relevantes de Astra DB
            relevant_documents = []
            if not disable_vector_store:
                if strategy == 'Maximal Marginal Relevance':
                    relevant_documents = vectorstore.max_marginal_relevance_search(query=question, k=top_k_vectorstore)
                else: # Basic Retrieval por defecto
                    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_vectorstore})
                    relevant_documents = retriever.get_relevant_documents(query=question)
            
            # Paso 2: Cargar el historial de memoria del chat
            memory = load_memory_rc(chat_history, top_k_history if not disable_chat_history else 0)
            history = memory.load_memory_variables({}).get('chat_history', [])
            
            # Paso 3: Construir la cadena LangChain. 'chain' se define aquí.
            rag_chain_inputs = {'context': lambda x: x['context'], 'chat_history': lambda x: x['chat_history'], 'question': lambda x: x['question']}
            current_prompt_obj = get_prompt(prompt_type, custom_prompt, language)
            chain = RunnableMap(rag_chain_inputs) | current_prompt_obj | model

            # Paso 4: Ejecutar la cadena y mostrar la respuesta
# Bloque nuevo con la llamada a la función de seguimiento
try:
    # Paso 1: Obtener y mostrar la respuesta principal en streaming
    response = chain.invoke(
        {'question': question, 'chat_history': history, 'context': relevant_documents},
        config={'callbacks': [StreamHandler(response_placeholder)]}
    )
    final_content = response.content

    if memory: 
        memory.save_context({'question': question}, {'answer': final_content})

    st.session_state.messages.append(AIMessage(content=final_content))

    # Paso 2: Generar la pregunta de seguimiento y guardarla en la sesión
    with st.spinner("Generando sugerencia..."):
        suggested_question = generate_follow_up_question(question, final_content, model)
        if suggested_question:
            st.session_state.suggested_question = suggested_question

    st.rerun()

except Exception as e:
    st.error(f"Error durante la generación de la respuesta: {e}")

# --- FIN DEL ARCHIVO ---

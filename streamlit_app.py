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

# --- CONFIGURACIÓN DE PÁGINA (DEBE SER LO PRIMERO) ---
st.set_page_config(page_title=" ", page_icon='./customizations/logo/anim-logo-1fps-verde.gif', layout="wide")

# --- INICIALIZACIÓN DE SESSION STATE (ÚNICA Y CORRECTA) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "header_drawn" not in st.session_state: # Añadido para controlar el dibujado del encabezado
    st.session_state.header_drawn = False


# --- CONFIGURACIÓN GLOBAL ---
ASTRA_DB_COLLECTION_NAME = "vc_assistant"
ADMIN_USERS = ["openlab_admin"]

print("Streamlit App Started") # Esto se muestra en los logs de Streamlit Cloud

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
        try:
            user_creds = st.secrets.get('passwords', {})
            if st.session_state.get('username') in user_creds and hmac.compare_digest(st.session_state.get('password', ''), user_creds[st.session_state.get('username')]):
                st.session_state['password_correct'] = True
                st.session_state.user = st.session_state['username']
                if 'password' in st.session_state: del st.session_state['password']
            else:
                st.session_state['password_correct'] = False
        except Exception:
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

# --- Función para codificar la imagen a base64 (Movida aquí para definirse una sola vez) ---
def get_image_as_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

import hashlib
from langchain.schema import Document

def vectorize_text(uploaded_files, vectorstore, lang_dict):
    if not vectorstore:
        st.error(lang_dict.get('vectorstore_not_ready_admin',
                               "Vectorstore not ready for upload."))
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", ";", " "],
    )

    total_chunks = 0

    for up_file in uploaded_files:
        ext = Path(up_file.name).suffix.lower()

        # 1. Guardamos el archivo en un temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(up_file.getvalue())
            tmp_path = tmp.name  # ruta física en disco

        try:
            # 2. Seleccionamos loader según tipo
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext == ".csv":
                loader = CSVLoader(tmp_path, encoding="utf-8")
            elif ext == ".txt":
                with open(tmp_path, "r", encoding="utf-8") as f:
                    text = f.read()
                loader = None
                docs = [Document(page_content=text, metadata={"file_name": up_file.name})]
            else:
                st.warning(f"Tipo {ext} no soportado: {up_file.name}")
                continue

            docs = loader.load() if loader else docs

            # 3. Añadimos metadatos básicos
            for d in docs:
                d.metadata["file_name"] = up_file.name

            # 4. Chunking
            chunks = splitter.split_documents(docs)

            # 5. Hash + metadatos únicos
            for c in chunks:
                sha = hashlib.sha1(c.page_content.encode()).hexdigest()
                c.metadata["sha1"] = sha

            # 6. Insertamos en vectorstore
            vectorstore.add_documents(chunks)
            total_chunks += len(chunks)
            st.success(f"{up_file.name}: {len(chunks)} chunks cargados.")

        except Exception as e:
            st.error(f"Error procesando {up_file.name}: {e}")
        finally:
            os.remove(tmp_path)  # limpiamos el temp file

    st.info(f"Proceso completado. {total_chunks} chunks añadidos.")

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
                loader = WebBaseLoader(url)
                docs = loader.load()
                pages = text_splitter.split_documents(docs)
                vectorstore.add_documents(pages)
                st.info(f"✅ URL processed: {url}")
            except Exception as e:
                st.error(f"Error cargando desde URL {url}: {e}")

def generate_follow_up_question(question, answer, model):
    """
    Hace una segunda llamada a la IA para generar una pregunta de seguimiento.
    """
    _template = """Basado en la pregunta original del usuario y la respuesta que ha dado la IA, genera una única y concisa pregunta de seguimiento para invitar al usuario a profundizar.
Devuelve únicamente el texto de la pregunta, sin saludos, prefijos ni nada más.

Pregunta del Usuario: "{user_question}"

Respuesta de la IA: "{ai_answer}"

Pregunta de Seguimiento Sugerida:"""
    
    prompt = ChatPromptTemplate.from_template(_template)
    
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

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def get_prompt(type_param, custom_prompt, language):
    """
    Devuelve un ChatPromptTemplate con los campos {context}, {chat_history} y
    {question} listos para ser rellenados por RunnableMap.
    """
    # ---------------------- Extended results ----------------------
    if type_param == "Extended results":
        system_prompt = """### ROL Y PERSONALIDAD ###
    Actúa como un Analista Estratégico Senior especializado en la intersección de Inteligencia Artificial, Venture Capital y Private Equity. Tu nombre es "Asistente Experto IA". Tu tono es profesional, analítico y basado en datos. Te diriges a un usuario experto que valora respuestas concisas pero profundas.

    ### DIRECTIVA PRINCIPAL ###
    Tu única función es responder a las preguntas del usuario utilizando EXCLUSIVAMENTE la información proporcionada en la sección 'Contexto'. No inventes. Si no hay suficiente información, reconoce las limitaciones.

    ### REGLAS FUNDAMENTALES (NO MODIFICABLES) ###
    1. ❌ No alucines  
    2. ❌ No uses conocimiento externo  
    3. ✅ Usa el historial si aporta contexto  
    4. ✅ Extrae valor y evita vaguedades  
    5. Si el ‘Contexto’ contiene MÚLTIPLES casos o ejemplos, incluye **todos los que aparezcan**, resumiendo cada uno en 3-4 bullets con datos cuantificables.


    ### ESTILO Y ESTRUCTURA ADAPTATIVA ###
    - Comienza siempre con una **introducción de 1–2 frases** que anticipe lo más importante.
    - Luego adapta la respuesta en función del tipo de pregunta:
      - Si es un análisis comparativo o de casos → usa **bloques con títulos** (ej. "✅ 1. EQT Ventures – Motherbrain").
      - Si es conceptual o técnico → usa **viñetas o pasos secuenciales**.
      - Si es una decisión o estrategia → ofrece pros/contras o alternativas claras.
    - Resalta nombres propios, cifras o impactos tangibles si están presentes.
    - Si hay limitaciones en el contexto, **reconócelas con claridad**.

    ---
    **Contexto Relevante de los Documentos:**  
    {context}

    **Historial de Chat:**  
    {chat_history}

    **Pregunta del Usuario:**  
    {question}

    **Respuesta del Asistente Experto IA:**"""

        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

    # ---------------------- Short results ----------------------
    elif type_param == "Short results":
        system_prompt = """Eres un asistente que responde de forma muy breve.
Si no conoces la respuesta basándote en el contexto, di claramente
'No conozco la respuesta basada en los documentos proporcionados'.

Contexto:  
{context}

Historial:  
{chat_history}

Pregunta:  
{question}
"""
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

    # ---------------------- Custom ----------------------
    else:
        # Si el usuario ha proporcionado un prompt personalizado, úsalo tal cual.
        template = custom_prompt if custom_prompt else "{question}"
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(template),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

from langchain.retrievers.multi_query import MultiQueryRetriever
from collections import defaultdict
from langchain.schema.document import Document

def load_retriever(vectorstore, llm, top_k):
    """
    Devuelve una función `fused(query)` que combina:
      • Recuperación multiconsulta generada con LLM
      • Coincidencia BM25 clásica
    y aplica Reciprocal Rank Fusion (RRF) para mejorar cobertura y precisión.
    """

    # MultiQueryRetriever genera variantes de la pregunta para mejor cobertura
    retriever_mqr = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        llm=llm,
        prompt="Reescribe esta pregunta de tres formas distintas para mejorar la búsqueda en español e inglés:"
    )

    # BM25 clásico
    def retr_bm25(query: str):
        return vectorstore.similarity_search(query, k=top_k)

    # RRF: fusiona resultados dando prioridad a la posición en cada lista
    def rrf(lists, k=60):
        scores = defaultdict(float)
        doc_map = {}

        for docs in lists:
            for rank, doc in enumerate(docs):
                key = doc.page_content
                scores[key] += 1 / (rank + 1)
                doc_map[key] = doc

        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [doc_map[content] for content, _ in top_docs]

    # Función final que recibe la pregunta del usuario
    def fused(query: str):
        docs_mqr = retriever_mqr.get_relevant_documents(query)
        docs_bm25 = retr_bm25(query)
        return rrf([docs_mqr, docs_bm25], k=top_k)

    return fused
    
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

def list_document_sources(vectorstore):
    """
    Realiza una búsqueda genérica en el vectorstore para obtener los metadatos
    de los documentos y devuelve una lista de las fuentes únicas.
    """
    if not vectorstore:
        st.error("El Vector Store no está disponible.")
        return []

    try:
        # Hacemos una búsqueda de similitud con un término genérico para obtener documentos.
        # Pedimos un número alto (k=1000) para intentar obtener una muestra representativa.
        results = vectorstore.similarity_similarity_search("*", k=1000)
        
        # Usamos un set para guardar solo los nombres de archivo únicos
        sources = set()
        for doc in results:
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])
        
        return sorted(list(sources))

    except Exception as e:
        st.error(f"No se pudieron obtener los documentos: {e}")
        return []


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

@st.cache_data()
def get_custom_prompt(username):
    prompt_path = Path(f"./customizations/prompt/{username}.txt")
    if not prompt_path.is_file():
        prompt_path = Path("./customizations/prompt/default.txt")
    
    try:
        return prompt_path.read_text(encoding='utf-8')
    except Exception:
        return "Responde a la pregunta basándote en el contexto y la conversación previa."

@st.cache_resource(show_spinner="Conectando con la IA...")
def load_embedding_rc():
    return OpenAIEmbeddings(openai_api_key=st.secrets.get("OPENAI_API_KEY"))

@st.cache_resource(show_spinner="Cargando base de conocimiento...")
def load_vectorstore_rc(_embedding):
    return AstraDB(
        embedding=_embedding,
        collection_name=ASTRA_DB_COLLECTION_NAME,
        token=st.secrets["ASTRA_TOKEN"],
        api_endpoint=st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT")),
        namespace=st.secrets.get("ASTRA_KEYSPACE")
    )

@st.cache_resource(show_spinner="Cargando historial de chat...")
def load_chat_history_rc(username, session_id):
    return AstraDBChatMessageHistory(
        session_id=f"{username}_{str(session_id)}",
        api_endpoint=st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT")),
        token=st.secrets["ASTRA_TOKEN"],
        collection_name="historial_chat_asistente"
    )

@st.cache_resource(show_spinner="Cargando modelo de lenguaje...")
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
user_defaults = st.secrets.get("DEFAULT_SETTINGS", {}).get(username, {})

embedding = load_embedding_rc()
vectorstore = load_vectorstore_rc(embedding) if embedding else None
chat_history = load_chat_history_rc(username, st.session_state.session_id) if vectorstore else None
model = load_model_rc()

# 3. Configuración de Parámetros (Condicional según el usuario)
if username != "demo":
    with st.sidebar:
        # Logo, Logout, Rails...
        # La imagen del logo por defecto ya se renderiza por el CSS global, este es el logo antiguo
        # st.image('./customizations/logo/default.svg', use_column_width="always") 
        st.markdown(f"""{lang_dict.get('logout_caption','Logged in as')} :orange[{username}]""")
        if st.button(lang_dict.get('logout_button','Logout')): logout()
        st.divider()

        rails_dict = load_rails(username)
        # Este bloque de rails es para el sidebar, si no lo quieres, puedes borrarlo
        st.subheader(lang_dict.get('rails_1', "Suggestions"))
        st.caption(lang_dict.get('rails_2', "Try asking:"))
        if rails_dict:
            for i in sorted(rails_dict.keys()): st.markdown(f"{i}. {rails_dict[i]}")
        st.divider()
        
        # Sidebar organizada con pestañas.
        st.header("Configuración")
        chat_tab, admin_tab = st.tabs(["⚙️ Opciones de Chat", "🗂️ Admin de Datos"])

        with chat_tab:
            st.markdown("##### Configuración de RAG y Memoria")
            disable_vector_store = st.toggle(lang_dict.get('disable_vector_store', "Desactivar RAG"), value=False)
            top_k_vectorstore = st.slider(
                lang_dict.get('top_k_vectorstore', "Documentos a recuperar (K)"),
                min_value=1,
                max_value=25,
                value=user_defaults.get("TOP_K_VECTORSTORE", 10),
                disabled=disable_vector_store    
            )

            rag_strategies = ('Basic Retrieval', 'Maximal Marginal Relevance', 'Fusion')
            default_strategy = user_defaults.get("RAG_STRATEGY", 'Basic Retrieval')
            strategy = st.selectbox(lang_dict.get('rag_strategy', "RAG Strategy"), rag_strategies, index=rag_strategies.index(default_strategy) if default_strategy in rag_strategies else 0, disabled=disable_vector_store)
            
            st.markdown("---")
            disable_chat_history = st.toggle(lang_dict.get('disable_chat_history', "Desactivar Memoria"), value=False)
            top_k_history = st.slider(lang_dict.get('k_chat_history', "Mensajes a recordar (K)"), 1, 10, user_defaults.get("TOP_K_HISTORY", 5), disabled=disable_chat_history)

        with admin_tab:
            if username in ADMIN_USERS:
                st.markdown("##### Carga de Contenido")
                with st.expander("Subir Archivos"):
                    uploaded_files = st.file_uploader("Subir archivos TXT, PDF, CSV", type=['txt', 'pdf', 'csv'], accept_multiple_files=True)
                    if st.button("Process Files"):
                        if uploaded_files: vectorize_text(uploaded_files, vectorstore, lang_dict)
                with st.expander("Cargar desde URLs"):
                    urls_text = st.text_area("Enter URLs (comma-separated)")
                    if st.button("Process URLs"):
                        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
                        if urls: vectorize_url(urls, vectorstore, lang_dict)
                
                st.markdown("---")
                with st.expander("Ver Documentos Cargados"):
                    st.caption("Muestra las fuentes de los documentos actualmente en la base de datos.")
                    if st.button("Listar documentos en Astra DB"):
                        with st.spinner("Buscando..."):
                            document_sources = list_document_sources(vectorstore)
                            if document_sources:
                                st.write("Se han encontrado las siguientes fuentes:")
                                for source_name in document_sources:
                                    st.markdown(f"- `{source_name}`")
                            else:
                                st.warning("No se encontraron documentos en la base de datos.")
            else:
                st.info("No tienes permisos de administrador para cargar datos.")
        
        # Opciones de Prompt fuera de las pestañas pero en la sidebar
                st.divider()
                st.header("Personalidad del Asistente")
                st.caption("El asistente usa siempre el modo 'Extended results' (prompt fijo).")
                prompt_type = "Extended results"   # forzamos estilo
                custom_prompt = ""                 # no se utiliza

else: # Si el usuario es 'demo'
    st.markdown("""<style>[data-testid="stSidebar"] {display: none}</style>""", unsafe_allow_html=True)
    # Definimos aquí los parámetros para el usuario 'demo'. Más limpio.
    user_defaults = st.secrets.get("DEFAULT_SETTINGS", {}).get(username, {})
    disable_chat_history = True
    top_k_history = 0
    disable_vector_store = False
    top_k_vectorstore = user_defaults.get("TOP_K_VECTORSTORE", 5)
    strategy = user_defaults.get("RAG_STRATEGY", 'Basic Retrieval')
    prompt_type = "Extended results"   # prompt fijo también para demo
    custom_prompt = ""


# Inicializar memoria
memory = load_memory_rc(chat_history, top_k_history if not disable_chat_history else 0) if chat_history else None


# --- COPIA Y PEGA ESTE BLOQUE COMPLETO HASTA EL FINAL DEL ARCHIVO ---

# --- Interfaz Principal del Chat (Visible para TODOS los usuarios) ---

# 1. Inyectamos el CSS final para centrar el layout y el logo
# Protegemos este bloque para que solo se dibuje una vez por sesión
st.markdown("""
    <style>
        /* Contenedor principal de toda la aplicación Streamlit */
        div[data-testid="stAppViewContainer"] {
            max-width: 55% !important; 
            margin: 0 auto !important;
        }
        /* Contenedor del campo de texto del chat */
        [data-testid="stChatInputContainer"] {
            max-width: 55% !important; 
            margin: 0 auto !important;
        }
    </style>
""", unsafe_allow_html=True)

# 2️⃣ Sólo una vez: dibujar header (nivel 0 también)
if not st.session_state.header_drawn:
    logo_base64 = get_image_as_base64("./customizations/logo/anim-logo-1fps-verde.gif")
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/gif;base64,{logo_base64}" alt="Logo" width="150">
            <h1>Agente Experto IA para Fondos</h1>
            <p>Por OPENLAB VENTURES, S.L. ®</p>
            <p style="color: #9c9d9f; font-size: 0.9rem;">
                Tu consultor virtual especializado…
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.session_state.header_drawn = True


# 3. Lógica de visualización del chat
# Mensajes de bienvenida y botones de sugerencia (si es una sesión nueva)
if not st.session_state.messages:
    # Mostramos el nuevo texto de bienvenida que querías
    st.info("Escribe tu pregunta en el cuadro de abajo o selecciona alguna de las siguientes sugerencias de preguntas")
    
    # --- AÑADIMOS LAS PREGUNTAS FIJAS (CON EL TEXTO ACTUALIZADO) ---
    PREGUNTAS_SUGERIDAS = [
        "¿Qué pasos iniciales debo dar para introducir IA en mi fondo?",
        "Háblame de casos de éxito de fondos que ya usan la IA",
        "¿Cómo está la IA transformando el deal sourcing en VC y PE?"
    ]
    
    # Creamos 3 columnas para los 3 botones
    cols = st.columns(len(PREGUNTAS_SUGERIDAS))
    
    # Creamos un botón en cada columna
    for i, pregunta in enumerate(PREGUNTAS_SUGERIDAS):
        if cols[i].button(pregunta, key=f"rail_fijo_{i}"):
            st.session_state.question_from_button = pregunta
            st.rerun()
    
# Mostrar todo el historial de chat en cada ejecución
for message in st.session_state.messages:
    avatar_icon = "🤖" if message.type == "ai" else "🧑‍💻"
    with st.chat_message(message.type, avatar=avatar_icon): # Corregido: type a message.type
        st.markdown(message.content)

# Lógica para mostrar la pregunta sugerida dinámica (después de una respuesta)
if "suggested_question" in st.session_state and st.session_state.suggested_question:
    if st.button(f"Sugerencia: *{st.session_state.suggested_question}*"):
        st.session_state.question_from_button = st.session_state.suggested_question
        del st.session_state.suggested_question 
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Lógica para recibir una nueva pregunta del usuario
# ──────────────────────────────────────────────────────────────────────────────
question = st.session_state.pop("question_from_button", None)
if not question:
    if user_query := st.chat_input(lang_dict.get('assistant_question',
                                                 "Pregunta lo que quieras...")):
        question = user_query

# 🚦 Verificación inmediata: si sigue sin haber pregunta válida, detenemos flujo
if question is None or not str(question).strip():
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 5. Lógica para procesar la pregunta
# ──────────────────────────────────────────────────────────────────────────────
st.session_state.messages.append(HumanMessage(content=question))
with st.chat_message("human", avatar="🧑‍💻"):
    st.markdown(question)

with st.chat_message("assistant", avatar="🤖"):
    response_placeholder = st.empty()

    # Si faltan modelo o vectorstore, mostramos error y salimos
    if not model or (not disable_vector_store and not vectorstore):
        response_placeholder.markdown(
            "Lo siento, el asistente no está completamente configurado."
        )
        st.stop()

    # ───────── Recuperamos los documentos relevantes ─────────
    relevant_documents = []
    if not disable_vector_store:
        retriever = load_retriever(vectorstore, model, top_k_vectorstore)
        relevant_documents = retriever(question)   # ahora retriever es una función

    # ────────── Bloque de depuración: mostrar chunks ──────────
    if not disable_vector_store:
        with st.sidebar.expander("📝 Chunks recuperados", expanded=False):
            for i, doc in enumerate(relevant_documents, start=1):
                src = doc.metadata.get("source", "sin_fuente")
                preview = doc.page_content[:200].replace("\n", " ")
                st.markdown(f"**{i}. {src}**  \n{preview}…")

    # ────────── Preparamos memoria y prompt ──────────
    memory = load_memory_rc(
        chat_history,
        top_k_history if not disable_chat_history else 0,
    )
    history = memory.load_memory_variables({}).get("chat_history", [])

    print("\n\n========== CONTEXTO PASADO AL PROMPT ==========\n")
    for i, doc in enumerate(relevant_documents):
        print(f"[Chunk {i+1}]\n{doc.page_content}\n")
    print("========== FIN CONTEXTO ==========\n\n")
    
    rag_chain_inputs = {
        "context": lambda x: x["context"],
        "chat_history": lambda x: x["chat_history"],
        "question": lambda x: x["question"],
    }
    current_prompt_obj = get_prompt(prompt_type, custom_prompt, language)
    chain = RunnableMap(rag_chain_inputs) | current_prompt_obj | model

    # ────────── Ejecutamos el chain con streaming ──────────
    try:
        response = chain.invoke(
            {
                "question": question,
                "chat_history": history,
                "context": relevant_documents,
            },
            config={"callbacks": [StreamHandler(response_placeholder)]},
        )
        final_content = response.content

        # Guardamos en memoria
        if memory:
            memory.save_context({"question": question}, {"answer": final_content})

        # Añadimos al historial visible
        st.session_state.messages.append(AIMessage(content=final_content))

        # Generamos una pregunta de seguimiento sugerida
        with st.spinner("Generando sugerencia..."):
            suggested_question = generate_follow_up_question(
                question, final_content, model
            )
            if suggested_question:
                st.session_state.suggested_question = suggested_question

        # Rerun para refrescar UI
        st.rerun()

    except Exception as e:
        st.error(f"Error durante la generación de la respuesta: {e}")

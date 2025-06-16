import streamlit as st

# --- Prueba 1: Eliminar el texto superior ---
# Ponemos un título de página que es solo un espacio en blanco.
st.set_page_config(page_title=" ", layout="wide")

st.header("Fase de Diagnóstico")
st.info("El objetivo es ver si podemos centrar el contenido de abajo.")

# --- Prueba 2: Centrar un bloque de HTML ---
# Definimos un estilo muy simple para centrar texto dentro de un contenedor
# y le ponemos un borde rojo para poder VER el contenedor.
st.markdown("""
    <style>
    .contenedor-de-prueba {
        border: 2px solid red;  /* Borde rojo para visualizar la 'caja' */
        text-align: center;     /* La regla de centrado que queremos probar */
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Creamos el contenedor con nuestro estilo y le metemos texto.
st.markdown('<div class="contenedor-de-prueba"><h1>Texto de Prueba</h1><p>Si este texto, incluido el título, aparece centrado dentro de una caja roja, entonces el método funciona.</p></div>', unsafe_allow_html=True)

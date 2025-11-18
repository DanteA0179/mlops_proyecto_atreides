"""
Página del copiloto conversacional con IA.

Esta página proporciona una interfaz de chat para interactuar con un asistente
inteligente que ayuda con análisis energético y simulaciones.

User Story: US-032.3 - Copiloto Conversacional
"""

import streamlit as st
import requests
from datetime import datetime

# Importar constantes del módulo
from src.streamlit_app.pages import API_BASE_URL, ICONS, MESSAGES, DEFAULT_TIMEOUT


def render():
    """Renderiza la página del copiloto conversacional."""
    
    st.title("🤖 Copiloto Conversacional de Energía")
    
    st.markdown("""
        Pregúntame cualquier cosa sobre optimización energética, análisis de datos,
        o simulaciones "what-if". Utilizo IA Generativa para ayudarte a tomar mejores decisiones.
    """)
    
    st.markdown("---")
    
    # Inicializar historial de chat en session_state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """👋 ¡Hola! Soy tu copiloto de optimización energética.

Puedo ayudarte con:
- 📊 Análisis de patrones de consumo
- 🔮 Simulaciones "what-if"
- 💡 Recomendaciones de optimización
- 📈 Explicación de drivers de consumo
- ⚙️ Análisis de eficiencia operacional

¿En qué puedo ayudarte hoy?"""
            }
        ]
    
    # Mostrar historial de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta del asistente
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("🤔 Analizando..."):
                try:
                    # Llamar a la API del chatbot
                    response = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={
                            "message": prompt,
                            "history": st.session_state.messages[:-1]  # Excluir el último mensaje
                        },
                        timeout=DEFAULT_TIMEOUT
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        assistant_response = result.get("response", "Lo siento, no pude generar una respuesta.")
                        
                        # Mostrar respuesta
                        message_placeholder.markdown(assistant_response)
                        
                        # Agregar al historial
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": assistant_response
                        })
                        
                        # Mostrar metadata si está disponible
                        if "metadata" in result:
                            with st.expander("📊 Información Adicional"):
                                st.json(result["metadata"])
                    
                    else:
                        error_msg = f"❌ Error {response.status_code}: {response.text}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                
                except requests.exceptions.ConnectionError:
                    error_msg = MESSAGES['api_offline']
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                
                except Exception as e:
                    error_msg = f"❌ Error inesperado: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Sidebar con ejemplos de preguntas
    with st.sidebar:
        st.markdown("### 💡 Ejemplos de Preguntas")
        
        example_questions = [
            "¿Cuáles son los principales drivers de consumo energético?",
            "¿Qué pasaría si aumento el factor de potencia a 0.95?",
            "¿Cómo puedo reducir el consumo en horas pico?",
            "Explícame el patrón de consumo de los últimos días",
            "¿Cuál es el impacto del tipo de carga en las emisiones de CO2?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{hash(question)}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
        
        st.markdown("---")
        
        # Botón para limpiar conversación
        if st.button("🗑️ Limpiar Conversación", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]  # Mantener solo el mensaje de bienvenida
            st.rerun()

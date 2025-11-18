"""
Chat/Copilot endpoints for Energy Optimization API.

This module provides conversational AI endpoints for the copilot.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    history: List[ChatMessage] = Field(
        default=[],
        description="Conversation history"
    )
    mode: Optional[str] = Field(
        default="conversational",
        description="Chat mode"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default={},
        description="Additional parameters"
    )


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Assistant response")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Response metadata"
    )


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with AI Copilot",
    description="Send a message to the AI copilot for conversational analysis"
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint for conversational AI.
    
    This is a mock implementation. In production, this should integrate
    with an LLM service (Ollama, OpenAI, etc.)
    
    Parameters
    ----------
    request : ChatRequest
        Chat request with message and history
        
    Returns
    -------
    ChatResponse
        AI response with metadata
    """
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")
        
        # Mock response based on keywords
        response_text = generate_mock_response(request.message, request.history)
        
        # Generate metadata
        metadata = {
            "model": "mock-llm",
            "tokens": len(response_text.split()),
            "latency_ms": 100,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Chat response generated successfully")
        
        return ChatResponse(
            response=response_text,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


def generate_mock_response(message: str, history: List[ChatMessage]) -> str:
    """
    Generate a mock response based on the message.
    
    This is a simple implementation for demo purposes.
    In production, integrate with actual LLM.
    
    Parameters
    ----------
    message : str
        User message
    history : List[ChatMessage]
        Conversation history
        
    Returns
    -------
    str
        Generated response
    """
    message_lower = message.lower()
    
    # Drivers de consumo
    if any(word in message_lower for word in ["driver", "factor", "influy", "principal"]):
        return """Basándome en el análisis de feature importance del modelo, los principales drivers de consumo energético son:

📊 **Top 5 Factores Más Influyentes:**

1. **CO2 (tCO2)** - 35% de importancia
   - Mayor impacto en el consumo
   - Correlación directa con la intensidad operacional

2. **Factor de Potencia Retrasada** - 28% de importancia
   - Eficiencia en el uso de energía
   - Valores óptimos: > 0.90

3. **NSM (Hora del Día)** - 15% de importancia
   - Patrones temporales significativos
   - Consumo pico: 8:00-15:00

4. **Tipo de Carga** - 12% de importancia
   - Maximum Load: mayor consumo
   - Light Load: operación eficiente

5. **Día de la Semana** - 10% de importancia
   - Fin de semana: ~40% menos consumo
   - Días laborales: operación completa

💡 **Recomendación:** Optimizar el factor de potencia y planificar cargas pesadas fuera de horas pico."""

    # What-if scenarios
    elif any(word in message_lower for word in ["qué pasaría", "what if", "simula", "escenario"]):
        return """Para realizar una simulación "what-if", necesito algunos parámetros. Pero puedo darte un ejemplo:

🔮 **Escenario: Aumentar Factor de Potencia de 0.80 a 0.95**

**Impacto Esperado:**
- 📉 Reducción de consumo: ~8-12%
- 💰 Ahorro estimado: $15,000-20,000/año
- 🌍 Reducción CO2: ~5-7 tCO2/año
- ⚡ Mejora en eficiencia: Clase "Excelente"

**Implementación:**
1. Instalar bancos de capacitores
2. Monitoreo continuo del factor de potencia
3. Ajuste de cargas inductivas

**ROI:** 18-24 meses

¿Te gustaría simular otro escenario específico? Por ejemplo:
- Cambio en tipo de carga
- Optimización de horarios
- Impacto de mantenimiento preventivo"""

    # Reducción de consumo
    elif any(word in message_lower for word in ["reduc", "disminui", "ahorro", "optimiz"]):
        return """💡 **Estrategias para Reducir Consumo en Horas Pico**

**1. Gestión de Demanda (15-20% reducción)**
- Programar cargas pesadas fuera de horario pico (8-15h)
- Usar horario valle (22:00-6:00) para procesos no urgentes
- Implementar sistema de programación automática

**2. Optimización de Factor de Potencia (8-12% reducción)**
- Instalar compensación reactiva
- Mantener factor > 0.92
- Monitoreo en tiempo real

**3. Eficiencia Operacional (10-15% reducción)**
- Mantenimiento predictivo de equipos
- Actualización de motores a alta eficiencia
- Sistemas de control avanzados

**4. Gestión de Cargas (5-10% reducción)**
- Balance de cargas entre fases
- Reducción de cargas stand-by
- Apagado automático de equipos inactivos

**Ahorro Total Potencial:** 35-45% en horas pico
**Inversión:** $50,000-80,000
**ROI:** 2-3 años

¿Te gustaría profundizar en alguna estrategia específica?"""

    # Patrones de consumo
    elif any(word in message_lower for word in ["patrón", "horario", "hora", "tendencia"]):
        return """📈 **Análisis de Patrones de Consumo**

**Patrón Típico Diario:**

🌅 **Madrugada (00:00-06:00)**
- Consumo: 35-40 kWh
- Estado: Operación mínima
- Oportunidad: Cargas programables

☀️ **Mañana (06:00-12:00)**
- Consumo: 55-75 kWh
- Estado: Rampa de producción
- Pico: 11:00-12:00 (75 kWh)

🌞 **Tarde (12:00-18:00)**
- Consumo: 65-70 kWh
- Estado: Producción sostenida
- Eficiencia: Moderada

🌙 **Noche (18:00-24:00)**
- Consumo: 45-55 kWh
- Estado: Cierre de operaciones
- Reducción gradual

**Patrón Semanal:**
- Lunes-Viernes: 100% capacidad
- Sábado: 60% capacidad
- Domingo: 30% capacidad

**Recomendaciones:**
1. Cargas pesadas: 22:00-06:00
2. Mantenimiento: Domingos
3. Procesos batch: Madrugada"""

    # Impacto CO2
    elif any(word in message_lower for word in ["co2", "emisiones", "ambiental", "carbono"]):
        return """🌍 **Análisis de Impacto Ambiental - CO2**

**Relación Consumo-Emisiones:**

**Factores de Emisión por Tipo de Carga:**
- Maximum Load: 0.0012 tCO2/kWh
- Medium Load: 0.0009 tCO2/kWh  
- Light Load: 0.0006 tCO2/kWh

**Impacto Anual Estimado:**
- Consumo total: ~450,000 kWh/año
- Emisiones: ~405 tCO2/año
- Equivalente: 180 autos o 8,100 árboles

**Estrategias de Reducción:**

1. **Optimización Operacional (-15%)**
   - Reducción: 60 tCO2/año
   - Costo: Mínimo
   - Plazo: Inmediato

2. **Eficiencia Energética (-25%)**
   - Reducción: 100 tCO2/año
   - Inversión: $50,000
   - ROI: 2 años

3. **Energías Renovables (-50%)**
   - Reducción: 200 tCO2/año
   - Inversión: $200,000
   - ROI: 5 años

**Objetivo Net Zero:**
Combinando las 3 estrategias: -90% emisiones para 2030

¿Te interesa un plan de acción específico?"""

    # Default response
    else:
        return f"""Gracias por tu pregunta: "{message}"

Como copiloto de optimización energética, puedo ayudarte con:

📊 **Análisis de Datos**
- Patrones de consumo
- Drivers de energía
- Correlaciones entre variables

🔮 **Simulaciones**
- Escenarios "what-if"
- Impacto de cambios operacionales
- Proyecciones de ahorro

💡 **Recomendaciones**
- Estrategias de reducción
- Mejores prácticas
- Optimización de operaciones

📈 **Análisis de Eficiencia**
- Factor de potencia
- Impacto ambiental (CO2)
- Oportunidades de mejora

¿Podrías ser más específico sobre qué aspecto te gustaría explorar?"""

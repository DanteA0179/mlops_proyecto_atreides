"""Cliente HTTP para la API de FastAPI"""
import requests
from typing import Dict, Any, Optional, List
import streamlit as st


class APIClient:
    """Cliente para consumir la API de predicción"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.timeout = 30
        
    def health_check(self) -> bool:
        """Verifica que la API esté disponible"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def get_models(self) -> Optional[List[str]]:
        """Obtiene lista de modelos disponibles"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            st.error(f"Error obteniendo modelos: {e}")
            return None
    
    def get_metrics(self, model: Optional[str] = None) -> Optional[Dict]:
        """Obtiene métricas del modelo"""
        try:
            params = {"model": model} if model else {}
            response = requests.get(
                f"{self.base_url}/metrics",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error obteniendo métricas: {e}")
            return None
    
    def predict(self, features: Dict[str, Any]) -> Optional[Dict]:
        """Realiza predicción"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=features,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error en predicción: {e}")
            return None
    
    def simulate(self, base_features: Dict[str, Any], variations: List[Dict[str, Any]], model: Optional[str] = None) -> Optional[Dict]:
        """Ejecuta simulaciones what-if"""
        try:
            payload = {
                "base_features": base_features,
                "variations": variations,
                "model": model
            }
            response = requests.post(
                f"{self.base_url}/simulate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error en simulación: {e}")
            return None

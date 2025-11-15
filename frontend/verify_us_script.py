#!/usr/bin/env python3
"""
Script de Verificación para US-032 y US-033
Sistema de Optimización Energética - Atreides

Verifica que todos los componentes de las User Stories estén implementados correctamente.
"""

import os
import sys
from pathlib import Path
import subprocess
import requests
from typing import List, Tuple
import importlib.util

# Colores para terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Imprime un header destacado"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")

def print_success(text: str):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text: str):
    """Imprime mensaje de error"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_warning(text: str):
    """Imprime mensaje de advertencia"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_info(text: str):
    """Imprime mensaje informativo"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")

def check_file_exists(filepath: str) -> bool:
    """Verifica que un archivo exista"""
    path = Path(filepath)
    if path.exists():
        print_success(f"Archivo encontrado: {filepath}")
        return True
    else:
        print_error(f"Archivo faltante: {filepath}")
        return False

def check_python_module(module_name: str) -> bool:
    """Verifica que un módulo de Python esté instalado"""
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        print_success(f"Módulo instalado: {module_name}")
        return True
    else:
        print_error(f"Módulo faltante: {module_name}")
        return False

def check_service_running(url: str, service_name: str, optional: bool = False) -> bool:
    """Verifica que un servicio esté corriendo"""
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            print_success(f"{service_name} está corriendo en {url}")
            return True
        else:
            if optional:
                print_warning(f"{service_name} no responde correctamente (opcional)")
            else:
                print_error(f"{service_name} no responde correctamente")
            return False
    except requests.exceptions.RequestException:
        if optional:
            print_warning(f"{service_name} no está corriendo en {url} (opcional)")
        else:
            print_error(f"{service_name} no está corriendo en {url}")
        return False

def check_docker_image(image_name: str) -> bool:
    """Verifica que una imagen de Docker exista"""
    try:
        result = subprocess.run(
            ['docker', 'images', '-q', image_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            print_success(f"Imagen Docker encontrada: {image_name}")
            return True
        else:
            print_warning(f"Imagen Docker no encontrada: {image_name} (construir con docker build)")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_warning("Docker no está instalado o no responde")
        return False

def verify_us032() -> Tuple[int, int]:
    """
    Verifica US-032: Streamlit UI - Página de predicción
    
    Returns:
        Tuple[int, int]: (checks_passed, total_checks)
    """
    print_header("US-032: Streamlit UI - Página de Predicción")
    
    checks = []
    
    # 1. Verificar archivos principales
    print_info("Verificando archivos principales...")
    checks.append(check_file_exists("streamlit_app.py"))
    checks.append(check_file_exists("pages/1_🔮_Predicción.py"))
    checks.append(check_file_exists("pages/2_💬_Copiloto_IA.py"))
    
    # 2. Verificar configuración
    print_info("\nVerificando configuración...")
    checks.append(check_file_exists(".streamlit/config.toml"))
    
    # secrets.toml es opcional en desarrollo
    if not Path(".streamlit/secrets.toml").exists():
        print_warning("Archivo .streamlit/secrets.toml no encontrado (usar template)")
        checks.append(False)
    else:
        checks.append(True)
    
    # 3. Verificar dependencias
    print_info("\nVerificando dependencias de Python...")
    required_modules = [
        'streamlit',
        'plotly',
        'pandas',
        'requests'
    ]
    
    for module in required_modules:
        checks.append(check_python_module(module))
    
    # 4. Verificar estructura de páginas
    print_info("\nVerificando estructura de páginas...")
    
    # Verificar que streamlit_app.py tenga 3 tabs
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'st.tabs' in content and '["🏠 Inicio", "🔮 Predicción", "💬 Copiloto IA"]' in content:
                print_success("App principal tiene 3 tabs configurados")
                checks.append(True)
            else:
                print_error("App principal no tiene los 3 tabs requeridos")
                checks.append(False)
    except FileNotFoundError:
        print_error("No se pudo verificar contenido de streamlit_app.py")
        checks.append(False)
    
    # 5. Verificar página de predicción
    print_info("\nVerificando funcionalidades de predicción...")
    
    try:
        with open('pages/1_🔮_Predicción.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Verificar formulario
            has_form = all([
                'st.number_input' in content,
                'st.slider' in content,
                'st.selectbox' in content or 'st.radio' in content,
                'st.button' in content
            ])
            
            if has_form:
                print_success("Formulario de predicción implementado")
                checks.append(True)
            else:
                print_error("Formulario de predicción incompleto")
                checks.append(False)
            
            # Verificar validación
            has_validation = 'validate' in content.lower() or 'error' in content.lower()
            if has_validation:
                print_success("Validación de inputs implementada")
                checks.append(True)
            else:
                print_warning("Validación de inputs no detectada")
                checks.append(False)
            
            # Verificar visualizaciones
            has_gauges = 'gauge' in content.lower() and 'plotly' in content
            if has_gauges:
                print_success("Gauges de visualización implementados")
                checks.append(True)
            else:
                print_error("Gauges de visualización no implementados")
                checks.append(False)
            
    except FileNotFoundError:
        print_error("No se pudo verificar página de predicción")
        checks.extend([False, False, False])
    
    # 6. Verificar copiloto conversacional
    print_info("\nVerificando copiloto conversacional...")
    
    try:
        with open('pages/2_💬_Copiloto_IA.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Verificar chat interface
            has_chat = 'st.chat_input' in content or 'messages' in content
            if has_chat:
                print_success("Interfaz de chat implementada")
                checks.append(True)
            else:
                print_error("Interfaz de chat no implementada")
                checks.append(False)
            
            # Verificar integración con Ollama
            has_ollama = 'ollama' in content.lower() or 'llama' in content.lower()
            if has_ollama:
                print_success("Integración con Ollama implementada")
                checks.append(True)
            else:
                print_warning("Integración con Ollama no detectada")
                checks.append(False)
    
    except FileNotFoundError:
        print_error("No se pudo verificar copiloto conversacional")
        checks.extend([False, False])
    
    # 7. Verificar servicios (opcional)
    print_info("\nVerificando servicios (opcional)...")
    api_running = check_service_running("http://localhost:8000/health", "API Backend", optional=True)
    ollama_running = check_service_running("http://localhost:11434/api/tags", "Ollama", optional=True)
    
    # No cuentan para el score pero son informativos
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\n{Colors.BOLD}Resultado US-032: {passed}/{total} checks pasados{Colors.RESET}")
    
    return passed, total

def verify_us033() -> Tuple[int, int]:
    """
    Verifica US-033: Deployment de Streamlit
    
    Returns:
        Tuple[int, int]: (checks_passed, total_checks)
    """
    print_header("US-033: Deployment de Streamlit")
    
    checks = []
    
    # 1. Verificar Dockerfile
    print_info("Verificando Dockerfile...")
    checks.append(check_file_exists("Dockerfile.streamlit"))
    
    # 2. Verificar docker-compose
    print_info("\nVerificando docker-compose...")
    checks.append(check_file_exists("docker-compose.streamlit.yml"))
    
    # 3. Verificar requirements.txt
    print_info("\nVerificando requirements.txt...")
    checks.append(check_file_exists("requirements.txt"))
    
    # 4. Verificar scripts de deployment
    print_info("\nVerificando scripts de deployment...")
    checks.append(check_file_exists("scripts/deploy_streamlit_cloudrun.sh"))
    checks.append(check_file_exists("scripts/deploy_streamlit_cloudrun.ps1"))
    
    # 5. Verificar documentación
    print_info("\nVerificando documentación de deployment...")
    checks.append(check_file_exists("docs/STREAMLIT_DEPLOYMENT.md"))
    checks.append(check_file_exists("QUICKSTART.md") or check_file_exists("docs/QUICKSTART.md"))
    
    # 6. Verificar que Dockerfile esté bien formado
    print_info("\nVerificando contenido de Dockerfile...")
    try:
        with open('Dockerfile.streamlit', 'r') as f:
            content = f.read()
            
            has_base_image = 'FROM python:' in content
            has_workdir = 'WORKDIR' in content
            has_copy = 'COPY' in content
            has_expose = 'EXPOSE 8501' in content
            has_cmd = 'CMD' in content and 'streamlit' in content
            
            if all([has_base_image, has_workdir, has_copy, has_expose, has_cmd]):
                print_success("Dockerfile correctamente configurado")
                checks.append(True)
            else:
                print_error("Dockerfile incompleto o mal configurado")
                checks.append(False)
    except FileNotFoundError:
        print_error("No se pudo verificar Dockerfile")
        checks.append(False)
    
    # 7. Verificar imagen Docker (opcional)
    print_info("\nVerificando imagen Docker (opcional)...")
    image_exists = check_docker_image("energy-optimizer-ui")
    # No cuenta para el score
    
    # 8. Verificar que docker-compose esté bien formado
    print_info("\nVerificando docker-compose.yml...")
    try:
        with open('docker-compose.streamlit.yml', 'r') as f:
            content = f.read()
            
            has_streamlit = 'streamlit:' in content
            has_ports = '8501:8501' in content
            has_env = 'environment:' in content or 'env_file:' in content
            
            if all([has_streamlit, has_ports, has_env]):
                print_success("docker-compose.yml correctamente configurado")
                checks.append(True)
            else:
                print_error("docker-compose.yml incompleto")
                checks.append(False)
    except FileNotFoundError:
        print_error("No se pudo verificar docker-compose.yml")
        checks.append(False)
    
    # 9. Verificar permisos de scripts
    print_info("\nVerificando permisos de scripts...")
    bash_script = Path("scripts/deploy_streamlit_cloudrun.sh")
    if bash_script.exists():
        is_executable = os.access(bash_script, os.X_OK)
        if is_executable:
            print_success("Script bash tiene permisos de ejecución")
            checks.append(True)
        else:
            print_warning("Script bash no tiene permisos de ejecución (chmod +x)")
            checks.append(False)
    else:
        checks.append(False)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\n{Colors.BOLD}Resultado US-033: {passed}/{total} checks pasados{Colors.RESET}")
    
    return passed, total

def print_summary(us032_passed: int, us032_total: int, us033_passed: int, us033_total: int):
    """Imprime resumen final"""
    print_header("RESUMEN FINAL")
    
    total_passed = us032_passed + us033_passed
    total_checks = us032_total + us033_total
    percentage = (total_passed / total_checks * 100) if total_checks > 0 else 0
    
    print(f"US-032 (Streamlit UI):       {us032_passed}/{us032_total} ({'✅ PASS' if us032_passed == us032_total else '⚠️  PARTIAL'})")
    print(f"US-033 (Deployment):         {us033_passed}/{us033_total} ({'✅ PASS' if us033_passed == us033_total else '⚠️  PARTIAL'})")
    print(f"\n{Colors.BOLD}Total:                       {total_passed}/{total_checks} ({percentage:.1f}%){Colors.RESET}")
    
    if total_passed == total_checks:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ¡TODAS LAS VERIFICACIONES PASARON!{Colors.RESET}")
        print(f"{Colors.GREEN}Las User Stories US-032 y US-033 están completas.{Colors.RESET}")
    elif percentage >= 80:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  CASI COMPLETO{Colors.RESET}")
        print(f"{Colors.YELLOW}Revisa los items faltantes arriba.{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ INCOMPLETO{Colors.RESET}")
        print(f"{Colors.RED}Varios componentes faltan. Revisa los errores arriba.{Colors.RESET}")
    
    print("\n" + "="*80)

def main():
    """Función principal"""
    print_header("VERIFICACIÓN DE USER STORIES US-032 Y US-033")
    print_info("Sistema de Optimización Energética - Atreides\n")
    
    # Verificar que estemos en el directorio correcto
    if not Path("streamlit_app.py").exists() and not Path("pyproject.toml").exists():
        print_error("Este script debe ejecutarse desde la raíz del proyecto")
        sys.exit(1)
    
    # Verificar US-032
    us032_passed, us032_total = verify_us032()
    
    # Verificar US-033
    us033_passed, us033_total = verify_us033()
    
    # Imprimir resumen
    print_summary(us032_passed, us032_total, us033_passed, us033_total)
    
    # Exit code basado en resultado
    total_passed = us032_passed + us033_passed
    total_checks = us032_total + us033_total
    
    if total_passed == total_checks:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()

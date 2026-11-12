from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.environment import EnvironmentToolset
from google.adk.environment import LocalEnvironment
import os

WORKING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "environment_agent")
os.makedirs(WORKING_DIR, exist_ok=True)

agent_01 = Agent(
    model=LiteLlm(
        model="ollama_chat/qwen3:8b",
        reasoning_effort="none"
    ),
    name='agent_01',
    description='Crea, lee, modifica y borra archivos en la carpeta de trabajo environment_agent.',
    instruction="""Eres un agente que gestiona archivos dentro de tu carpeta de trabajo (environment_agent).

    Puedes crear, leer, modificar y borrar archivos usando tus herramientas.

    Reglas:
    - Trabaja solo dentro de tu carpeta de trabajo, nunca fuera de ella.
    - Usa rutas relativas a tu carpeta de trabajo.
    - Antes de modificar o borrar un archivo, verifica que existe.
    - No sobrescribas un archivo existente sin que el usuario lo haya pedido.
    - Al terminar, informa en español qué archivo creaste, modificaste o borraste.
    """,
    tools=[EnvironmentToolset(
        environment=LocalEnvironment(
            working_dir=WORKING_DIR
        )
    )],
)
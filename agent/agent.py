from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from duckduckgo_search import DDGS
from tools.tools import index_data, list_collections, retrieval, delete_collection, index_file, get_collection_info, web_search
from agent.subagents.subagents_01 import agent_01

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

root_agent = Agent(
    model=LiteLlm(
        model="ollama_chat/qwen3:8b",
        reasoning_effort="none"
    ),
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction="""Eres un asistente que responde preguntas. 
    utiliza la herramienta index_data para almacenar las respuesta encontradas en la web
    
        para responder una pregunta sigue la siguiente secuencia:
        1. Lista las colecciones
        2. utiliza la herramienta retrieval para responder

        usa get_collection_info para ver el contenido de una colección antes de consultarla.
        usa index_file para indexar un archivo existente.
        usa delete_collection solo si el usuario pide explícitamente borrar una colección.

        ###importante:
        cada vez que se necesita crear modificar o borrar un archivo usa el agente agent_01
        """,
    tools=[web_search, index_data, list_collections, retrieval, delete_collection, index_file, get_collection_info],
    sub_agents=[agent_01]
)
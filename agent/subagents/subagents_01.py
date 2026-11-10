from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.environment import EnvironmentToolset
from google.adk.environment import LocalEnvironment


agent_01 = Agent(
    model=LiteLlm(
        model="ollama_chat/nemotron-3-nano:30b-cloud",
        reasoning_effort="low"
    ),
    name='agent_01',
    description='A helpful assistant for user questions.',
    instruction="""
    Eres un agente que crea archivos en el enviroments LocalEnviroment
        """,
    tools=[EnvironmentToolset(
        environment=LocalEnvironment(
            working_dir="/home/ipf-2026/clase_agentes/enviroment_agent"
        )
    )],
)
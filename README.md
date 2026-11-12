# Clase Agentes

Asistente conversacional construido con [Google ADK](https://google.github.io/adk-docs/) y modelos locales de Ollama (vía LiteLLM). Combina búsqueda web, una base de conocimiento vectorial y un subagente que crea archivos.

## Qué hace

- **`root_agent`** (`agent/agent.py`): responde preguntas. Busca en la web (DuckDuckGo), indexa lo encontrado y consulta la base vectorial para responder.
- **`agent_01`** (`agent/subagents/subagents_01.py`): subagente que crea/modifica/borra archivos en `environment_agent/`.
- **Herramientas** (`tools/tools.py`): `web_search`, `index_data`, `index_file` (txt/md/pdf), `list_collections`, `get_collection_info`, `retrieval`, `delete_collection`. Usan ChromaDB (persistida en `vector_store/`) y embeddings `nomic-embed-text`.

## Requisitos

- Python 3.10+
- [Ollama](https://ollama.com/) corriendo en `http://localhost:11434`
- Modelos usados:
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen3:8b
  ```

## Cómo levantarlo

```bash
# 1. Entorno virtual e instalación de dependencias
python -m venv env
source env/bin/activate        # en fish: source env/bin/activate.fish
pip install -r requirements.txt

# 2. Asegurate de que Ollama esté corriendo
ollama serve

# 3. Levantar el agente (desde la raíz del proyecto)
adk web        # interfaz web en http://localhost:8000
# o bien
adk run agent  # modo terminal
```

## Notas

- `vector_store/`, `environment_agent/`, `env/` y `.env` están ignorados por git.

from langchain_chroma import Chroma
import chromadb
import os
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store")

def web_search(query: str, max_results: int = 5) -> list | str:
    """
    Busca información en la web utilizando DuckDuckGo.
    Args:
        query: str -> El término o pregunta a buscar en internet.
        max_results: int -> Número máximo de resultados a devolver.
    Returns:
        Lista de resultados de la búsqueda, o el mensaje de error.
    """
    try:
        # Se instancia DDGS correctamente
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        return str(e)

def index_data(text: str, c_name: str) -> list:
    """
    Divide un texto en fragmentos y los almacena en la base de datos vectorial.
    Args:
        text: str -> Texto que se desea indexar.
        c_name: str -> Nombre de la colección donde se guardarán los fragmentos.
    Returns:
        Lista con los IDs de los documentos agregados, o el mensaje de error.
    """
    try:
        vector_store= Chroma(
            collection_name= c_name,
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=OllamaEmbeddings(
                model="nomic-embed-text"
            )
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        docs = splitter.create_documents([text])

        results =vector_store.add_documents(documents=docs)
        return results

    except Exception as e:
        return str(e)

def list_collections() -> list:
    """
    Lista los nombres de las colecciones existentes en la base de datos vectorial.
    Returns:
        Lista de nombres de colecciones, o el mensaje de error.
    """
    try:
        client= chromadb.PersistentClient(
            path=VECTOR_STORE_DIR
        )
        collections= client.list_collections()
        results = [getattr(collection, "name", collection) for collection in collections]
        return results
    except Exception as e:
        return str(e)

def retrieval(query:str, collection_name:str, k:int=5)-> list:
    """
    Herramienta para consultar la base de datos vectorial.
    Args:
        query: str -> Pregunta o texto a buscar.
        collection_name: str -> Nombre de la colección a consultar.
        k: int -> Número de resultados a devolver.
    Returns:
        Lista de tuplas (documento, puntaje de relevancia), o el mensaje de error.
    """

    try:
        vector_store= Chroma(
            collection_name= collection_name,
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=OllamaEmbeddings(
                model="nomic-embed-text"
            )
        )

        return vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k
        )

    except Exception as e:
        return str(e)


def delete_collection(c_name: str) -> str:
    """
    Elimina una colección completa de la base de datos vectorial
    Args:
        c_name: str -> Nombre de la colección a eliminar.
    Returns:
        Mensaje de confirmación, o el mensaje de error.
    """
    try:
        client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        client.delete_collection(name=c_name)
        return f"Colección '{c_name}' eliminada."
    except Exception as e:
        return str(e)


def index_file(path: str, c_name: str) -> list:
    """
    Lee un archivo (.txt, .md o .pdf) y lo indexa en la base de datos vectorial.
    Args:
        path: str -> Ruta del archivo a indexar.
        c_name: str -> Nombre de la colección donde se guardarán los fragmentos.
    Returns:
        Lista con los IDs de los documentos agregados, o el mensaje de error.
    """
    try:
        if path.lower().endswith(".pdf"):
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        if not text.strip():
            return "El archivo no contiene texto para indexar."

        return index_data(text, c_name)
    except Exception as e:
        return str(e)


def get_collection_info(c_name: str, n_samples: int = 3) -> dict | str:
    """
    Devuelve información de una colección: cuántos documentos tiene y algunos ejemplos.
    Args:
        c_name: str -> Nombre de la colección a inspeccionar.
        n_samples: int -> Número de fragmentos de ejemplo a mostrar.
    Returns:
        Diccionario con el nombre, el total de documentos y los ejemplos, o el mensaje de error.
    """
    try:
        client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        collection = client.get_collection(name=c_name)
        samples = collection.peek(limit=n_samples)
        return {
            "name": c_name,
            "count": collection.count(),
            "samples": samples["documents"],
        }
    except Exception as e:
        return str(e)

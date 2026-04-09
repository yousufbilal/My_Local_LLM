from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

# 1. Load both CSVs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")

df = pd.read_csv(os.path.join(DATA_DIR, "gen9_pokemon.csv"))
df_movie = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# --- POKEMON DATABASE SECTION ---
pk_location = os.path.join(STORAGE_DIR, "chroma_pokemon_db")
add_pokemon = not os.path.exists(pk_location)

if add_pokemon:
    Pokemons = []
    ids = []
    for i, row in df.iterrows():
        pokemon = Document(
            page_content=f"Name: {row['Name']} | Type: {row['Type1']} | Region: {row['Region']}",
            metadata={"name": row["Name"], "type1": row["Type1"]},
            id=str(i)
        )
        ids.append(str(i))
        Pokemons.append(pokemon)

vector_store = Chroma(
    collection_name="gen9_pokemon",
    persist_directory=pk_location,
    embedding_function=embeddings
)

if add_pokemon:
    vector_store.add_documents(documents=Pokemons, ids=ids)


# --- MOVIES DATABASE SECTION ---
movies_location = os.path.join(STORAGE_DIR, "chroma_movies_db")
add_movies = not os.path.exists(movies_location)

if add_movies:
    Movies = []
    m_ids = []
    for i, row in df_movie.iterrows():
        movie = Document(
            page_content=f"Movie Name: {row['Movie Name']} | Director: {row['Director']} | Main Actors: {row['Main Actors']} | Plot Summary: {row['Plot Summary']}",
            # FIXED: Used 'Movie Name' and 'Director' instead of Title/Genre
            metadata={"title": row['Movie Name'], "director": row['Director']},
            id=str(i)
        )
        m_ids.append(str(i))
        Movies.append(movie)

movies_vector_store = Chroma(
    collection_name="movies_collection",
    persist_directory=movies_location,
    embedding_function=embeddings
)

if add_movies:
    movies_vector_store.add_documents(documents=Movies, ids=m_ids)


# --- RETRIEVERS ---
# We create the two specific ones
pokemon_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
movies_retriever = movies_vector_store.as_retriever(search_kwargs={"k": 10})

# # ADD THIS LINE: This fixes the 'ImportError' in your main.py
# # By default, we will point 'retriever' to the pokemon one
# retriever = pokemon_retriever

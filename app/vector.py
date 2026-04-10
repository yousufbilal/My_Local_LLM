from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

# 1. Load both CSVs
DATA_DIR = "./data"
STORAGE_DIR = "./storage"

#read both the files
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
            page_content=f"Name: {row['Name']} | Type1: {row['Type1']} | Attack: {row['Attack']} | Total: {row['Total']} | Category: {row['Category']}",
            
            metadata={"name": row["Name"], "type1": row["Type1"],"category": row["Category"], "is_legendary": row["Legendary/Mythical"], "source": "pokemon_db"},
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
    for i, row in df_movie.iterrows():        # FIXED: indented inside if block
        movie = Document(
            page_content=(f"MovieID: {row['MovieID']} | Title: {row['Title']} | Genre1: {row['Genre1']} | Year: {row['Year']} | Director: {row['Director']} | LeadActor: {row['LeadActor']} | LeadActress: {row['LeadActress']} | Budget(M$): {row['Budget(M$)']}"),
            metadata={"id": row['MovieID'], "title": row['Title'], "director": row['Director'], "genre": row['Genre1'], "year": int(row['Year']), "source": "movies_db"},
            id=str(i)
        )
        m_ids.append(str(i))                  # FIXED: already correct indent
        Movies.append(movie)                  # FIXED: already correct indent

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


from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

# df = pd.read_csv("realistic_restaurant_reviews.csv")
df = pd.read_csv("gen9_pokemon.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# db_location = "./chrome_langchain_db"
# add_documents = not os.path.exists(db_location)
pk_location = "./chroma_pokemon_db"
add_pokemon = not os.path.exists(pk_location)


# if add_documents:
if add_pokemon:
    # documents=[]
    Pokemons=[]
    ids=[]

    for i, row in df.iterrows():
        pokemon = Document(
        # document = Document(
            # page_content=row["Title"] + " " + row["Review"],
            # metadata={"rating": row["Rating"], "date": row["Date"]},
            page_content=row["Name"] + " " + row["Type1"] +  " " + row['Region'],
            metadata={"name": row["Name"], "type1": row["Type1"]},
            id=str(i)
        )
        # ids.append(str(i))
        # documents.append(document)
        ids.append(str(i))
        Pokemons.append(pokemon)
        

vector_store = Chroma(
    # collection_name="resturant_reviews",
    collection_name="gen9_pokemon",
    persist_directory= pk_location,
    embedding_function=embeddings
)

if add_pokemon:
    vector_store.add_documents(documents=Pokemons, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)
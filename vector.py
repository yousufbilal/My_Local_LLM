from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os


df = pd.read_csv("gen9_pokemon.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")


pk_location = "./chroma_pokemon_db"
add_pokemon = not os.path.exists(pk_location)

if add_pokemon:
    Pokemons=[]
    ids=[]

    for i, row in df.iterrows():
        pokemon = Document(

            # Labeling data ensures the Retriever finds the correct context for the AI.            page_content = f"Name: {row['Name']} | Type: {row['Type1']} | Region: {row['Region']}",
            metadata={"name": row["Name"], "type1": row["Type1"]},
            id=str(i)
        )
        # ids.append(str(i))
        # documents.append(document)
        ids.append(str(i))
        Pokemons.append(pokemon)
        

vector_store = Chroma(
    collection_name="gen9_pokemon",
    persist_directory= pk_location,
    embedding_function=embeddings
)

if add_pokemon:
    vector_store.add_documents(documents=Pokemons, ids=ids)

#K low so the AI search dont lag 
retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)
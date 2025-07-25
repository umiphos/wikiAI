# main.py
import os
from lxml import etree
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess

# === CONFIG ===
WIKI_XML = 'enwiki-20250601-pages-articles-multistream.xml'
MAX_ARTICLES = 1000
MODEL_NAME = 'all-MiniLM-L6-v2'

def parse_wikipedia(xml_path, max_articles=MAX_ARTICLES):
    context = etree.iterparse(xml_path, events=('end',), tag='{*}page')
    articles = []
    for _, elem in tqdm(context):
        ns = elem.findtext('{*}ns')
        if ns != '0':
            continue
        title = elem.findtext('{*}title')
        text_elem = elem.find('{*}revision')
        if text_elem is not None:
            text = text_elem.findtext('{*}text') or ""
            if len(text.strip()) > 0:
                articles.append({'title': title, 'text': text})
        elem.clear()
        if len(articles) >= max_articles:
            break
    return articles

# === INDEXER ===
def build_index(articles, model):
    docs = [f"{a['title']}. {a['text'][:500]}" for a in articles]
    embeddings = model.encode(docs, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    return index, docs

# === SEARCH ===
def search(query, model, index, docs, k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [docs[i] for i in I[0]]

# === OLLAMA ===
def ask_ollama(query, context):
    prompt = f"""
Contexto extraído de Wikipedia:

{context}

Pregunta: {query}
Respuesta:"""
    result = subprocess.run(
        ['ollama', 'run', 'mistral'],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()


if __name__ == '__main__':
    print("🚀 Parseando artículos...")
    articles = parse_wikipedia(WIKI_XML)

    print("🔎 Indexando...")
    model = SentenceTransformer(MODEL_NAME)
    index, docs = build_index(articles, model)

    while True:
        query = input("\n📝 Escribe una pregunta (o 'salir'): ")
        if query.lower() == 'salir':
            break
        results = search(query, model, index, docs)

        keywords = query.lower().split()
        filtered_results = [r for r in results if any(k in r.lower() for k in keywords)]

        context = "\n\n".join(filtered_results if filtered_results else results)

        answer = ask_ollama(query, context)
        print(f"\n💬 Respuesta:\n{answer}\n")

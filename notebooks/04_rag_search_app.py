%pip install sentence-transformers
dbutils.library.restartPython()

"""
04 - Engine de Busca Semântica (RAG) - Script de Teste
Este script recebe uma consulta em linguagem natural, converte-a num embedding,
e realiza a busca por similaridade vetorial na camada Gold do Delta Lake.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Carregar o modelo de embeddings (o mesmo utilizado na Camada Gold)
print("A carregar modelo de IA...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Ler os dados vetorizados da Camada Gold
print("A carregar base de dados vetorizada (Gold)...")
df_gold = spark.table("delta_brewery_gold_embeddings")
pdf_gold = df_gold.toPandas()

# Converter a coluna de embeddings de volta para matriz NumPy
matrix_embeddings = np.array(pdf_gold['embedding'].tolist())


def buscar_cervejarias_rag(query: str, top_k: int = 5):
    """
    Função que recebe uma pergunta e retorna as cervejarias mais relevantes por similaridade de cosseno.
    """
    print(f"\n--- Consulta efetuada: '{query}' ---")
    
    # Gerar embedding para a pergunta do utilizador
    query_vector = model.encode([query])
    
    # Calcular a similaridade por cosseno entre a pergunta e todas as cervejarias
    similarities = cosine_similarity(query_vector, matrix_embeddings)[0]
    
    # Obter os índices das top_k cervejarias mais similares
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    resultados = []
    for idx in top_indices:
        score = similarities[idx]
        item = pdf_gold.iloc[idx]
        resultados.append({
            "score_similaridade": round(float(score), 4),
            "nome": item['name'],
            "tipo": item['brewery_type'],
            "cidade": item['city'],
            "estado": item['state_province'],
            "pais": item['country'],
            "website": item['website_url'],
            "contexto": item['context_text']
        })
        
    return resultados

# ---------------------------------------------------------
# Exemplo de Teste de Busca Semântica
# ---------------------------------------------------------
pergunta_teste = "Microcervejarias artesanais ou brewpubs localizados na Alemanha com site oficial"
resultados = buscar_cervejarias_rag(pergunta_teste, top_k=3)

for r in resultados:
    print(f"\n[Score: {r['score_similaridade']}] {r['nome']} ({r['tipo']})")
    print(f"Localização: {r['cidade']}, {r['estado']} - {r['pais']}")
    print(f"Website: {r['website']}")
    print(f"Contexto: {r['contexto']}")
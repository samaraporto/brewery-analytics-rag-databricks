%pip install sentence-transformers
dbutils.library.restartPython()

"""
03 - Camada Gold: Enriquecimento e Embeddings para RAG
Este script lê a tabela Silver limpa, gera frases de contexto para cada cervejaria,
calcula os embeddings vetoriais, grava na tabela Gold e exporta para o Streamlit.
"""

from pyspark.sql.functions import col, concat_ws, lit
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# 1. Ler os dados limpos da Camada Silver
df_silver = spark.table("delta_brewery_silver")

df_for_embeddings = df_silver.select(
    "id", "name", "brewery_type", "city", "state_province", "country", "website_url"
)

print(f"Total de registros carregados para vetorização: {df_for_embeddings.count()}")

# 2. Criar a frase descritiva contextual (Context Generation para o RAG)
df_context = df_for_embeddings.withColumn(
    "context_text",
    concat_ws(
        ", ",
        col("name"),
        concat_ws(" ", lit("é uma cervejaria do tipo"), col("brewery_type")),
        concat_ws(" ", lit("localizada em"), col("city")),
        col("state_province"),
        col("country")
    )
)

display(df_context.limit(5))

# 3. Geração de Embeddings usando Sentence-Transformers
print("A carregar modelo de embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Converter para Pandas para vetorização em lote
pdf = df_context.toPandas()

print("Gerando embeddings vetoriais...")
embeddings = model.encode(pdf['context_text'].tolist(), show_progress_bar=True)

# Adicionar os vetores gerados de volta ao DataFrame pandas
pdf['embedding'] = embeddings.tolist()

# Converter de volta para Spark DataFrame
df_gold = spark.createDataFrame(pdf)

# 4. Salvando na Camada Gold (Delta Lake)
table_gold_name = "delta_brewery_gold_embeddings"

df_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(table_gold_name)

print(f"Dados vetorizados e salvos com sucesso na tabela Delta Gold: {table_gold_name}")

# 5. Validação rápida da Gold
display(spark.sql(f"SELECT id, name, context_text FROM {table_gold_name} LIMIT 5"))

# 6. Exportar para o Repositório (Para a Demo do Streamlit)
os.makedirs("data", exist_ok=True)
caminho_arquivo = "data/gold_breweries.parquet"
pdf.attrs = {}
pdf.to_parquet(caminho_arquivo, index=False)

print(f"Arquivo exportado com sucesso para: {caminho_arquivo}")
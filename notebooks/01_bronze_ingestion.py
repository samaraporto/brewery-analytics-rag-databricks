"""
01 - Camada Bronze: Ingestão de Dados Brutos
Este script extrai os dados paginados da API Open Brewery DB
e armazena o JSON bruto na tabela Delta Lake Bronze.
"""

import urllib.request
import json
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from pyspark.sql.functions import current_timestamp

# 1. Configurações da API com cabeçalho de navegador (User-Agent) para evitar bloqueio 403
url = "https://api.openbrewerydb.org/v1/breweries"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

page = 1
per_page = 200
all_breweries = []

print("Iniciando a extração dos dados da API Open Brewery DB...")

# Loop de paginação para capturar todos os registros disponíveis
while True:
    req = urllib.request.Request(f"{url}?page={page}&per_page={per_page}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            status_code = response.status
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Erro ao acessar a API na página {page}: {e}")
        break
        
    if status_code != 200:
        print(f"Erro ao acessar a API na página {page}: {status_code}")
        break
    
    # Se a lista vier vazia, terminamos a paginação
    if not data or len(data) == 0:
        break
        
    all_breweries.extend(data)
    print(f"Página {page} capturada com sucesso ({len(data)} registros).")
    page += 1

print(f"Extração concluída! Total de registros obtidos: {len(all_breweries)}")

# 2. Verificação de segurança e conversão para DataFrame Spark
if len(all_breweries) > 0:
    raw_data = [Row(raw_json=json.dumps(item)) for item in all_breweries]
    df_bronze = spark.createDataFrame(raw_data)
    df_bronze = df_bronze.withColumn("ingestion_timestamp", current_timestamp())
else:
    schema = StructType([
        StructField("raw_json", StringType(), True),
        StructField("ingestion_timestamp", TimestampType(), True)
    ])
    df_bronze = spark.createDataFrame([], schema)
    raise ValueError("Atenção: Nenhum dado foi retornado pela API da Open Brewery DB.")

# 3. Salvando na Camada Bronze (Delta Lake)
table_name = "delta_brewery_bronze"

df_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"Dados salvos com sucesso na tabela Delta Bronze: {table_name}")
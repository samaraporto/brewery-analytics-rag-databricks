"""
02 - Camada Silver: Limpeza e Transformação (MERGE / Upsert)
Este script lê a tabela Bronze, extrai os campos do JSON bruto,
limpa as impurezas (espaços, quebras de linha, nulos) e faz o Upsert na tabela Silver.
"""

from pyspark.sql.functions import col, trim, lower, when, regexp_replace, get_json_object
from delta.tables import DeltaTable

# 1. Ler os dados crus da tabela Bronze
df_bronze = spark.table("delta_brewery_bronze")

# 2. Extração dos campos do JSON bruto utilizando get_json_object
df_extracted = df_bronze.select(
    get_json_object(col("raw_json"), "$.id").alias("id"),
    get_json_object(col("raw_json"), "$.name").alias("name"),
    get_json_object(col("raw_json"), "$.brewery_type").alias("brewery_type"),
    get_json_object(col("raw_json"), "$.address_1").alias("address_1"),
    get_json_object(col("raw_json"), "$.city").alias("city"),
    get_json_object(col("raw_json"), "$.state_province").alias("state_province"),
    get_json_object(col("raw_json"), "$.postal_code").alias("postal_code"),
    get_json_object(col("raw_json"), "$.country").alias("country"),
    get_json_object(col("raw_json"), "$.longitude").alias("longitude"),
    get_json_object(col("raw_json"), "$.latitude").alias("latitude"),
    get_json_object(col("raw_json"), "$.website_url").alias("website_url"),
    col("ingestion_timestamp")
)

# 3. Limpeza rigorosa e normalização dos dados (Tratando quebras de linha, espaços e nulos)
df_silver_staging = df_extracted \
    .withColumn("id", trim(col("id"))) \
    .withColumn("name", trim(regexp_replace(col("name"), "[\n\r]", " "))) \
    .withColumn("brewery_type", lower(trim(col("brewery_type")))) \
    .withColumn("city", trim(regexp_replace(col("city"), "[\n\r]", " "))) \
    .withColumn("state_province", trim(regexp_replace(col("state_province"), "[\n\r]", " "))) \
    .withColumn("country", trim(regexp_replace(col("country"), "[\n\r]", " "))) \
    .filter(col("id").isNotNull() & col("name").isNotNull()) \
    .dropDuplicates(["id"])

# Padronizando tipos nulos ou vazios para 'unknown'
df_silver_staging = df_silver_staging.withColumn(
    "brewery_type", 
    when((col("brewery_type") == "") | col("brewery_type").isNull(), "unknown").otherwise(col("brewery_type"))
)

# 4. Lógica de merge / upsert no Delta Lake para a Camada Silver
table_silver_name = "delta_brewery_silver"

# Verifica se a tabela Silver já existe, se não existir, cria na primeira execução
if not spark.catalog.tableExists(table_silver_name):
    print(f"Tabela {table_silver_name} não existe. Criando a tabela inicial...")
    df_silver_staging.write.format("delta").saveAsTable(table_silver_name)
else:
    print(f"Executando MERGE (Upsert) na tabela {table_silver_name}...")
    deltaTableSilver = DeltaTable.forName(spark, table_silver_name)
    
    deltaTableSilver.alias("target").merge(
        df_silver_staging.alias("source"),
        "target.id = source.id"
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()

print("Processamento da Camada Silver concluído com sucesso!")

# 5. Validação rápida dos dados limpos
display(spark.sql("SELECT brewery_type, count(*) as total FROM delta_brewery_silver GROUP BY brewery_type ORDER BY total DESC LIMIT 10"))
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da Página
st.set_page_config(
    page_title="Brewery RAG Search | Portfólio",
    page_icon="🍺",
    layout="wide"
)

st.title("🍺 Assistente Inteligente de Cervejarias (RAG)")
st.markdown("Explore mais de 11.000 cervejarias globais usando **Busca Semântica por IA** (Processado no Databricks com Delta Lake e Sentence-Transformers).")

# Carregar modelo e dados com cache para otimizar a performance
@st.cache_resource
def carregar_recursos():
    # Carrega o modelo leve de embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Carrega o arquivo Parquet gerado pela Camada Gold
    df = pd.read_parquet("data/gold_breweries.parquet")
    
    # Converte os embeddings para matriz NumPy
    embeddings = np.array(df['embedding'].tolist())
    
    return model, df, embeddings

with st.spinner("A carregar modelo de IA e base de dados de cervejarias..."):
    model, df, matrix_embeddings = carregar_recursos()

# Barra de pesquisa interativa
query = st.text_input(
    "O que procuras?", 
    value="Microcervejarias artesanais ou brewpubs localizados na Califórnia",
    help="Escreve o que quiseres em linguagem natural!"
)

top_k = st.slider("Número de resultados:", min_value=1, max_value=10, value=5)

if st.button("🔍 Pesquisar com IA", type="primary"):
    if not query.strip():
        st.warning("Por favor, escreve uma pergunta válida.")
    else:
        with st.spinner("A calcular similaridade vetorial..."):
            # Gerar embedding da pergunta do utilizador
            query_vector = model.encode([query])
            
            # Calcular similaridade de cosseno
            similarities = cosine_similarity(query_vector, matrix_embeddings)[0]
            
            # Obter os melhores resultados
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            st.success(idados_encontrados := f"Encontradas as {top_k} cervejarias mais relevantes!")
            
            # Exibir resultados em formato de cartões visuais
            for idx in top_indices:
                score = similarities[idx]
                item = df.iloc[idx]
                
                with st.expander(f"⭐ {item['name']} ({item['brewery_type'].title()}) — Relevância: {score:.1%}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Cidade:** {item['city']}")
                        st.write(f"**Estado/Província:** {item['state_province']}")
                        st.write(f"**País:** {item['country']}")
                    with col2:
                        if item['website_url']:
                            st.markdown(f"**Website:** [Aceder ao site]({item['website_url']})")
                        else:
                            st.write("**Website:** Não disponível")
                    
                    st.info(f"**Contexto analisado pela IA:** {item['context_text']}")

# Rodapé profissional
st.markdown("---")
st.markdown("👨‍💻 Projeto desenvolvido para portfólio de Engenharia de Dados & IA Generativa.")
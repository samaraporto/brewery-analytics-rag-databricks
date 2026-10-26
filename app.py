import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da Página
st.set_page_config(
    page_title="Brewery AI Chat | Portfólio",
    page_icon="🍺",
    layout="wide"
)

st.title("🍺 Chat Inteligente de Cervejarias (RAG)")
st.markdown("Converse com a IA para explorar mais de 11.000 cervejarias globais processadas via Databricks e Delta Lake.")

# 1. Carregar recursos com cache
@st.cache_resource
def carregar_recursos():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_parquet("notebooks/data/gold_breweries.parquet")
    embeddings = np.array(df['embedding'].tolist())
    return model, df, embeddings

with st.spinner("A carregar modelo de IA e base de dados..."):
    model, df_base, matrix_embeddings = carregar_recursos()

# 2. Histórico de mensagens do chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Sou o teu assistente especializado em cervejarias. O que gostarias de encontrar hoje? (Ex: *Microcervejarias na Alemanha*)"}
    ]

# Exibir mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Input do Chat
if query := st.chat_input("Escreve a tua pergunta aqui..."):
    # Adicionar mensagem do utilizador ao histórico
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("A pesquisar e a analisar o contexto..."):
            
            # --- FILTRAGEM INTELIGENTE POR PAÍS (Garante Assertividade) ---
            query_lower = query.lower()
            df_filtrado = df_base.copy()
            
            # Dicionário simples de deteção de países na pergunta
            paises_map = {
                "alemanha": "germany",
                "portugal": "portugal",
                "estados unidos": "united states",
                "eua": "united states",
                "espanha": "spain",
                "frança": "france",
                "reino unido": "united kingdom",
                "brasil": "brazil"
            }
            
            pais_detectado = None
            for pt_nome, en_nome in paises_map.items():
                if pt_nome in query_lower:
                    pais_detectado = en_nome
                    break
            
            # Se a pergunta mencionar um país, filtramos estritamente a base antes da IA
            if pais_detectado:
                df_filtrado = df_base[df_base['country'].str.lower() == pais_detectado].reset_index(drop=True)
                if len(df_filtrado) == 0:
                    # Fallback caso não ache exato
                    df_filtrado = df_base.copy()
            
            # Se houver dados filtrados, recalculamos os embeddings locais para o subset
            if len(df_filtrado) < len(df_base):
                sub_embeddings = np.array(df_filtrado['embedding'].tolist())
            else:
                sub_embeddings = matrix_embeddings
            
            # Executar busca por similaridade vetorial
            query_vector = model.encode([query])
            similarities = cosine_similarity(query_vector, sub_embeddings)[0]
            
            top_k = min(4, len(df_filtrado))
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Construir resposta conversacional da IA
            resposta_ia = f"Encontrei as seguintes opções mais relevantes para a tua busca:\n\n"
            
            for idx in top_indices:
                score = similarities[idx]
                item = df_filtrado.iloc[idx]
                
                site_txt = f"[Aceder ao site]({item['website_url']})" if item['website_url'] else "Não disponível"
                
                resposta_ia += f"### 🍻 {item['name']} ({item['brewery_type'].title()})\n"
                resposta_ia += f"- **Localização:** {item['city']}, {item['state_province']} — **{item['country']}**\n"
                resposta_ia += f"- **Website:** {site_txt}\n"
                resposta_ia += f"- *Relevância:* {score:.1%}\n\n"
            
            if len(df_filtrado) == 0 or top_k == 0:
                resposta_ia = "Desculpa, não encontrei nenhuma cervejaria com esses critérios exatos na base de dados."

            st.markdown(resposta_ia)
            st.session_state.messages.append({"role": "assistant", "content": resposta_ia})

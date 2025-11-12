# commodity_news_ai.py
import random
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
import json
import time
import os
from transformers import pipeline

#  CONFIG 
st.set_page_config(page_title="Commodity News AI - Moatez", layout="wide")
st.title("Commodity News AI Dashboard")
st.markdown("**PROJECT Demo – Moatez DHIEB** | EPI SUP | DNEXT Project #1")

#  SIDEBAR 
st.sidebar.header("Paramètres")
commodity = st.sidebar.selectbox("Commodity", ["corn", "wheat", "soybean", "coffee"])
num_articles = st.sidebar.slider("Articles à analyser", 1, 20, 5)

#  HEADERS 
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

#  FICHIER JSON 
JSON_FILE = "scraped_articles.json"

#  HUGGING FACE MODEL (FINANCIAL NEWS - 100% FONCTIONNEL) 
@st.cache_resource
def load_sentiment_model():
    try:
        model = pipeline(
            "sentiment-analysis",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            return_all_scores=False
        )
        st.success("IA Hugging Face (Financial News) chargée !")
        return model
    except Exception as e:
        st.error(f"Erreur chargement HF : {e}")
        st.warning("→ Fallback simulation activée")
        return None

#  SCRAPING RÉEL SUR BING NEWS RSS 
@st.cache_data(ttl=1800)
def scrape_bing_news(query, n=5):
    url = f"https://www.bing.com/news/search?q={query}+price&format=rss"
    articles = []
    try:
        time.sleep(1)
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            st.warning(f"HTTP {response.status_code} → Fallback")
            return []
        
        soup = BeautifulSoup(response.text, 'xml')
        items = soup.find_all('item')[:n]
        
        if not items:
            st.warning("Aucun article Bing → Fallback")
            return []
        
        for item in items:
            title = item.find('title').text if item.find('title') else "No title"
            link = item.find('link').text if item.find('link') else "#"
            articles.append({
                "title": title,
                "link": link,
                "source": "Bing News",
                "commodity": commodity,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        st.success(f"Scraping Bing News → {len(articles)} articles sauvés dans `{JSON_FILE}`")
        return articles

    except Exception as e:
        st.error(f"Erreur Bing : {e} → Fallback")
        return []

#  DONNÉES SIMULÉES (fallback) 
fallback_articles = [
    {"title": f"{commodity.title()} prices fall due to strong harvest", "link": "https://reuters.com", "source": "Simulé"},
    {"title": f"Brazil boosts {commodity} exports", "link": "https://bloomberg.com", "source": "Simulé"},
    {"title": f"EU imposes new tariffs on {commodity}", "link": "https://euronews.com", "source": "Simulé"}
]

#  ANALYSE IA (HF FINANCIER) 
def analyze_sentiment(title, model):
    if model:
        try:
            res = model(title)[0]
            label = res['label'].lower()
            score = res['score']
            if label == "negative": 
                score = -score
                sentiment_fr = "Négative"
            elif label == "positive": 
                sentiment_fr = "Positive"
            else:  # neutral
                score = 0
                sentiment_fr = "Neutre"
            return sentiment_fr, round(score, 3)
        except Exception as e:
            st.error(f"Erreur analyse HF : {e}")
            return "Erreur", 0
    else:
        # Fallback simulation
        pos = ["rise", "boost", "record", "strong", "surge", "high"]
        neg = ["fall", "drop", "drought", "threat", "tariff", "low"]
        t = title.lower()
        if any(w in t for w in pos): 
            return "Positive", round(random.uniform(0.6, 0.95), 3)
        if any(w in t for w in neg): 
            return "Négative", round(random.uniform(-0.95, -0.6), 3)
        return "Neutre", 0.0

if st.button("Lancer l'analyse IA"):
    with st.spinner("Scraping Bing News + Analyse IA (Hugging Face Financial)..."):
        real_articles = scrape_bing_news(commodity, num_articles)
        articles = real_articles if real_articles else fallback_articles[:num_articles]
        
        model = load_sentiment_model()
        
        results = []
        for art in articles:
            sentiment, score = analyze_sentiment(art["title"], model)
            results.append({
                "Titre": art["title"][:100] + "..." if len(art["title"]) > 100 else art["title"],
                "Sentiment": sentiment,
                "Score": score,
                "Lien": art["link"]
            })
        
        df = pd.DataFrame(results)
        
        #  DASHBOARD  STYLE 
        st.markdown("---")
        
        #  HEADER STATS 
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric(
                label="Articles analysés",
                value=len(df),
                delta=f"{commodity.title()}"
            )
        with col_stats2:
            pos_count = len(df[df["Sentiment"] == "Positive"])
            st.metric(
                label="Sentiment Positif",
                value=f"{pos_count}/{len(df)}",
                delta=f"+{round((pos_count/len(df)*100) if len(df)>0 else 0, 1)}%"
            )
        with col_stats3:
            neg_count = len(df[df["Sentiment"] == "Négative"])
            st.metric(
                label="Sentiment Négatif",
                value=f"{neg_count}/{len(df)}",
                delta=f"-{round((neg_count/len(df)*100) if len(df)>0 else 0, 1)}%"
            )

        st.markdown("---")

        #  CHART + TABLEAU 
        col_chart, col_table = st.columns([2, 1], gap="large")

        with col_chart:
            st.subheader("Tendance du Sentiment (IA Hugging Face)")
            fig = px.bar(
                df, 
                x="Titre", 
                y="Score", 
                color="Sentiment",
                color_discrete_map={"Positive": "#00D26A", "Négative": "#FF4B4B", "Neutre": "#A0A0A0", "Erreur": "#FFAA00"},
                title=f"Analyse IA sur les prix du {commodity.title()}",
                hover_data={"Lien": True}
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Score IA",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", size=12),
                legend_title="Sentiment"
            )
            fig.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y}<br><extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.subheader("Détail des Résultats IA")
            # Style tableau
            df_display = df.copy()
            df_display["Lien"] = df_display["Lien"].apply(
                lambda x: f"[Voir l'article]({x})" if x != "#" and "http" in x else "Lien non disponible"
            )
            st.dataframe(
                df_display,
                use_container_width=True,
                column_config={
                    "Titre": st.column_config.TextColumn("Titre", width="medium"),
                    "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                    "Score": st.column_config.NumberColumn("Score IA", format="%.3f"),
                    "Lien": st.column_config.LinkColumn("Source", width="small")
                }
            )

        st.markdown("---")

        #  JSON CHECK 
        with st.expander("Check technique : Données brutes (scraping)", expanded=False):
            if os.path.exists(JSON_FILE):
                with open(JSON_FILE, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                st.success(f"{len(json_data)} articles scrapés avec succès")
                st.json(json_data, expanded=False)
            else:
                st.warning("Aucun fichier JSON trouvé → Scraping échoué")

    st.success("Analyse IA terminée !")

#  FOOTER
st.markdown("---")
col_footer1, col_footer2 = st.columns([3, 1])
with col_footer1:
    st.markdown(
        """
        **Commodity News AI** – Analyse en temps réel des prix agricoles  
        *Scraping Bing News + IA Hugging Face (modèle financier)*  
        **Développé par Moatez DHIEB** – Étudiant en Ingénierie Informatique (EPI SUP)  
        """
    )
with col_footer2:
    st.markdown("**Projet Demo**  \nDNEXT Intelligence SA  \n*Fév  2026*")

#  MON CV INTÉGRÉ 
with st.expander("Voir mon CV complet (clique pour télécharger)", expanded=False):
    col_cv1, col_cv2 = st.columns([1, 2])
    with col_cv1:
        st.markdown("""**MediConnect – Plateforme de télémédecine intelligente**
                         **Scanne le QR code, découvre une plateforme prête pour la production.**
                                             et apprécier le travail """)
        st.image("MediConnectQrCode.jpg", use_container_width=True)
    with col_cv2:
        st.markdown("### **Moatez DHIEB**")
        st.markdown("**Software Engineer | Data Science | Full-Stack**")
        st.markdown("Sousse, Tunisie | +216 24500607 | dhiebmoatez@gmail.com")
        st.markdown("[GitHub](https://github.com/MoatezDh) | [LinkedIn](www.linkedin.com/in/moatez-dhieb)")
        if os.path.exists("CV_Moatez_DHIEB.pdf"):
            with open("CV_Moatez_DHIEB.pdf", "rb") as f:
                st.download_button(
                    label="Télécharger CV (PDF)",
                    data=f.read(),
                    file_name="CV_Moatez_DHIEB.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:

            st.warning("Fichier PDF manquant → Ajoute `CV_Moatez_DHIEB.pdf`")

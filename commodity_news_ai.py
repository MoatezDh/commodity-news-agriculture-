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
from datetime import datetime

# CONFIG
st.set_page_config(page_title="Commodity News AI - Moatez", layout="wide")
st.title("Commodity News AI Dashboard")
st.markdown("**PROJECT Demo – Moatez DHIEB** | EPI SUP | DNEXT Project #1")

# SIDEBAR
st.sidebar.header("Paramètres")
commodity = st.sidebar.selectbox("Commodity", ["corn", "wheat", "soybean", "coffee"])
num_articles = st.sidebar.slider("Articles à analyser", 1, 50, 10)  # Increased for more data points

# NEW: Date Filters for Historical/Recent Scraping
st.sidebar.subheader("Filtre de Date (pour scraping & plot)")
start_date = st.sidebar.date_input("Date de début", value=datetime.now().date() - pd.DateOffset(months=12))  # Default last year
end_date = st.sidebar.date_input("Date de fin", value=datetime.now().date())
granularity = st.sidebar.selectbox("Granularité du Plot", ["Daily", "Monthly", "Yearly"])  # User chooses

# HEADERS
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

# FICHIER JSON
JSON_FILE = "scraped_articles.json"

# HUGGING FACE MODEL (FINANCIAL NEWS - 100% FONCTIONNEL)
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

# ENHANCED SCRAPING: Add date parsing + optional date filters (Bing supports approx. via query)
@st.cache_data(ttl=1800)
def scrape_bing_news(query, n=10, start_date=None, end_date=None):
    # Build query with dates if provided (Bing RSS approx. supports "since:YYYY-MM-DD until:YYYY-MM-DD")
    date_filter = ""
    if start_date and end_date:
        date_filter = f' since:{start_date.strftime("%Y-%m-%d")} until:{end_date.strftime("%Y-%m-%d")}'
    url = f"https://www.bing.com/news/search?q={query}+price{date_filter}&format=rss"
    articles = []
    try:
        time.sleep(1)
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            st.warning(f"HTTP {response.status_code} → Fallback")
            return generate_historical_fallback(query, n)  # Use enhanced fallback
        
        soup = BeautifulSoup(response.text, features="lxml-xml")
        items = soup.find_all('item')[:n]
        
        if not items:
            st.warning("Aucun article Bing → Fallback historique")
            return generate_historical_fallback(query, n)
        
        for item in items:
            title = item.find('title').text if item.find('title') else "No title"
            link = item.find('link').text if item.find('link') else "#"
            pubdate_tag = item.find('pubDate')
            date_str = pubdate_tag.text if pubdate_tag else None
            if date_str:
                try:
                    # Parse RSS date: "Wed, 13 Nov 2025 12:00:00 GMT"
                    dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
                    parsed_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    parsed_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Fallback to now
            else:
                parsed_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            articles.append({
                "title": title,
                "link": link,
                "source": "Bing News",
                "commodity": query,
                "timestamp": parsed_date  # Parsed date with hour
            })
        
        # SAVE JSON with dates
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        st.success(f"Scraping Bing → {len(articles)} articles avec dates sauvés dans `{JSON_FILE}`")
        return articles

    except Exception as e:
        st.error(f"Erreur Bing : {e} → Fallback historique")
        return generate_historical_fallback(query, n)

# NEW: Generate Simulated Historical Data (for demo, like Air Passengers 1949-1960 style)
def generate_historical_fallback(commodity, n=20):
    # Simulate 5 years (2020-2024) of monthly news-like data
    dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS')  # Monthly start
    articles = []
    for i, date in enumerate(dates[:n]):
        # Simulate titles with varying sentiment
        sentiments = random.choices(['rise', 'fall', 'stable'], weights=[0.4, 0.4, 0.2], k=1)[0]
        if sentiments == 'rise':
            title = f"{commodity.title()} prices surge due to demand boom"
        elif sentiments == 'fall':
            title = f"{commodity.title()} prices drop amid oversupply"
        else:
            title = f"{commodity.title()} market stable this {date.month_name()}"
        
        # Random hour for granularity
        hour = random.randint(0, 23)
        full_date = date.replace(hour=hour)
        
        articles.append({
            "title": title,
            "link": f"https://news.com/article-{i}",
            "source": "Simulé Historique",
            "commodity": commodity,
            "timestamp": full_date.strftime("%Y-%m-%d %H:%M:%S")
        })
    st.info(f"Fallback activé : {len(articles)} articles historiques simulés (2020-2024)")
    return articles

# ANALYSE IA (HF FINANCIER) - Unchanged
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
            else:
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
        real_articles = scrape_bing_news(commodity, num_articles, start_date, end_date)
        articles = real_articles  # Fallback now historical
        
        model = load_sentiment_model()
        
        results = []
        for art in articles:
            sentiment, score = analyze_sentiment(art["title"], model)
            # Parse timestamp to datetime
            try:
                dt = pd.to_datetime(art["timestamp"])
            except:
                dt = pd.to_datetime('now')
            results.append({
                "Titre": art["title"][:100] + "..." if len(art["title"]) > 100 else art["title"],
                "Sentiment": sentiment,
                "Score": score,
                "Lien": art["link"],
                "Date": dt  # NEW: Datetime column
            })
        
        df = pd.DataFrame(results)
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]  # Filter by dates
        
        # DASHBOARD STYLE (existing bar chart)
        st.markdown("---")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric(label="Articles analysés", value=len(df), delta=f"{commodity.title()}")
        with col_stats2:
            pos_count = len(df[df["Sentiment"] == "Positive"])
            st.metric(label="Sentiment Positif", value=f"{pos_count}/{len(df)}", delta=f"+{round((pos_count/len(df)*100) if len(df)>0 else 0, 1)}%")
        with col_stats3:
            neg_count = len(df[df["Sentiment"] == "Négative"])
            st.metric(label="Sentiment Négatif", value=f"{neg_count}/{len(df)}", delta=f"-{round((neg_count/len(df)*100) if len(df)>0 else 0, 1)}%")

        st.markdown("---")

        # EXISTING: Bar Chart
        col_chart, col_table = st.columns([2, 1], gap="large")
        with col_chart:
            st.subheader("Distribution du Sentiment (IA Hugging Face)")
            fig_bar = px.bar(
                df, x="Titre", y="Score", color="Sentiment",
                color_discrete_map={"Positive": "#00D26A", "Négative": "#FF4B4B", "Neutre": "#A0A0A0", "Erreur": "#FFAA00"},
                title=f"Analyse IA sur les prix du {commodity.title()}",
                hover_data={"Lien": True}
            )
            fig_bar.update_layout(xaxis_title="", yaxis_title="Score IA", height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Arial", size=12), legend_title="Sentiment")
            fig_bar.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y}<br><extra></extra>")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_table:
            st.subheader("Détail des Résultats IA")
            df_display = df.copy()
            df_display["Lien"] = df_display["Lien"].apply(lambda x: f"[Voir l'article]({x})" if x != "#" and "http" in str(x) else "Lien non disponible")
            st.dataframe(df_display, use_container_width=True, column_config={
                "Titre": st.column_config.TextColumn("Titre", width="medium"),
                "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "Score": st.column_config.NumberColumn("Score IA", format="%.3f"),
                "Lien": st.column_config.LinkColumn("Source", width="small"),
                "Date": st.column_config.DateColumn("Date", width="medium")  # NEW
            })

        st.markdown("---")

        # NEW: Time Series Plot (like Air Passengers)
        if len(df) > 0:
            st.markdown("---")
            st.subheader("Analyse Temporelle : Tendance du Sentiment sur le Temps")
            st.write(f"Évolution du score moyen de sentiment pour le {commodity.title()} ({granularity.lower()})")

            # Prepare data for time series
            df_ts = df.copy()
            df_ts['Year'] = df_ts['Date'].dt.year
            df_ts['Month'] = df_ts['Date'].dt.month_name()

            # Aggregate by granularity
            if granularity == "Daily":
                df_agg = df_ts.groupby(df_ts['Date'].dt.date)['Score'].mean().reset_index()
                df_agg['Date'] = pd.to_datetime(df_agg['Date'])
                fig_ts = px.line(df_agg, x="Date", y="Score", title=f"Tendance Quotidienne du Sentiment ({commodity.title()})")
            elif granularity == "Monthly":
                df_monthly = df_ts.groupby([df_ts['Date'].dt.to_period('M'), 'Year', 'Month'])['Score'].mean().reset_index()
                df_monthly['Date'] = df_monthly['Date'].dt.to_timestamp()
                fig_ts = px.line(df_monthly, x="Month", y="Score", color="Year", title=f"Tendance Mensuelle par Année ({commodity.title()})",
                                 markers=True)
                fig_ts.update_xaxes(categoryorder="array", categoryarray=[m for m in pd.date_range(start='1-1-2020', periods=12, freq='MS').month_name()])
            else:  # Yearly
                df_yearly = df_ts.groupby('Year')['Score'].mean().reset_index()
                fig_ts = px.line(df_yearly, x="Year", y="Score", title=f"Tendance Annuelle du Sentiment ({commodity.title()})")

            fig_ts.update_layout(
                xaxis_title="Temps", yaxis_title="Score Moyen IA",
                height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", size=12)
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("---")

        # JSON CHECK
        with st.expander("Check technique : Données brutes (scraping)", expanded=False):
            if os.path.exists(JSON_FILE):
                with open(JSON_FILE, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                st.success(f"{len(json_data)} articles scrapés avec dates")
                st.json(json_data, expanded=False)
            else:
                st.warning("Aucun fichier JSON trouvé → Scraping/Fallback échoué")

    st.success("Analyse IA terminée !")

# FOOTER & CV (unchanged)
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
    st.markdown("**Projet Demo**  \nDNEXT Intelligence SA  \n*Fév 2026*")

with st.expander("Voir mon CV complet (clique pour télécharger)", expanded=False):
    col_cv1, col_cv2 = st.columns([1, 2])
    with col_cv1:
        st.markdown("""**MediConnect – Plateforme de télémédecine intelligente**
                         **Scanne le QR code, découvre une plateforme prête pour la production.**
                                             et apprécier le travail """)
        if os.path.exists("MediConnectQrCode.jpg"):
            st.image("MediConnectQrCode.jpg", use_container_width=True)
        else:
            st.warning("Fichier QR manquant → Ajoute `MediConnectQrCode.jpg`")
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

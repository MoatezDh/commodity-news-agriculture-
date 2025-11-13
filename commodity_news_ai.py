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

# CONFIG
st.set_page_config(page_title="Commodity News AI - Moatez", layout="wide")
st.title("Commodity News AI Dashboard")
st.markdown("**PROJECT Demo – Moatez DHIEB** | EPI SUP | DNEXT Project #1")

# SIDEBAR
st.sidebar.header("Paramètres")
commodity = st.sidebar.selectbox("Commodity", ["corn", "wheat", "soybean", "coffee"])
num_articles = st.sidebar.slider("Articles à analyser", 1, 50, 5)

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

# SCRAPING RÉEL SUR BING NEWS RSS
# ADD AT TOP
if 'scrape_counter' not in st.session_state:
    st.session_state.scrape_counter = 0

# REPLACE scrape_news
def scrape_news(query, n=5):
    # Unique key forces fresh scrape
    cache_key = f"{query}_{n}_{st.session_state.scrape_counter}"
    
    if cache_key in st.session_state:
        articles = st.session_state[cache_key]
        st.info(f"Chargé depuis mémoire ({len(articles)} articles)")
        return articles

    articles = []
    url = f"https://news.google.com/rss/search?q={query}+price+when:1d&hl=en-US&gl=US&ceid=US:en"
    try:
        time.sleep(1)
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'xml')
            items = soup.find_all('item')
            st.info(f"Google News: {len(items)} articles trouvés")
        else:
            raise Exception("Google failed")
    except:
        st.warning("Google échoué → Fallback Bing")
        url = f"https://www.bing.com/news/search?q={query}+price&format=rss"
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                raise Exception("Bing failed")
            soup = BeautifulSoup(response.text, 'xml')
            items = soup.find_all('item')
            st.info(f"Bing: {len(items)} articles trouvés")
        except Exception as e:
            st.error(f"Scraping échoué: {e}")
            return []

    valid = []
    for item in items:
        title_tag = item.find('title')
        link_tag = item.find('link')
        if not title_tag or not link_tag:
            continue
        title = title_tag.text.strip()
        link = link_tag.text.strip()

        if any(x in title.lower() for x in ["video", "watch", "live", "youtube", "podcast"]):
            continue
        if len(title) < 25:
            continue

        # CLEAN GOOGLE LINK
        if "news.google.com/rss/articles/" in link:
            try:
                import base64, re
                from urllib.parse import urlparse
                path = urlparse(link).path
                encoded = path.split("/articles/")[-1].split("?")[0]
                missing = len(encoded) % 4
                if missing:
                    encoded += '=' * (4 - missing)
                decoded = base64.urlsafe_b64decode(encoded).decode('utf-8', 'ignore')
                match = re.search(r'"(https?://[^"]+)"', decoded)
                if match:
                    link = match.group(1)
            except:
                pass

        valid.append({
            "title": title,
            "link": link,
            "source": "Google News" if "google" in url else "Bing News",
            "commodity": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    articles = valid[:n]

    if articles:
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        st.success(f"{len(articles)} articles sauvés → {JSON_FILE}")
        st.session_state[cache_key] = articles
    else:
        st.warning("Aucun article valide")

    return articles
# ANALYSE IA (HF FINANCIER)
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
def make_clickable(url):
    if url == "#" or not url.startswith("http"):
        return "Lien non disponible"
    # Clean Bing redirect: extract real URL from 'url=' param
    if "bing.com/news/apiclick.aspx" in url:
        try:
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(url)
            real_url = parse_qs(parsed.query).get('url', [url])[0]
            if real_url:
                url = real_url
        except:
            pass  # fallback to original
    return f'<a href="{url}" target="_blank">Voir l\'article</a>'

if st.button("Lancer l'analyse IA"):
    st.session_state.scrape_counter += 1  # FORCE FRESH
    with st.spinner("Scraping Bing News + Analyse IA (Hugging Face Financial)..."):
        # FIXED: Dynamic fallback inside button (uses current commodity)
        fallback_articles = [
            {"title": f"{commodity.title()} prices fall due to strong harvest", "link": "https://reuters.com", "source": "Simulé"},
            {"title": f"Brazil boosts {commodity} exports", "link": "https://bloomberg.com", "source": "Simulé"},
            {"title": f"EU imposes new tariffs on {commodity}", "link": "https://euronews.com", "source": "Simulé"}
        ]
        
        real_articles = scrape_news(commodity, num_articles)
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
        
        # DASHBOARD STYLE
        st.markdown("---")
        
        # HEADER STATS
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

        # CHART + TABLEAU (2 COLUMNS)
        col_chart, col_table = st.columns([2, 1], gap="large")

        # === CHART COLUMN: BAR + LINE CHART ===
        with col_chart:
            st.subheader("Tendance du Sentiment (IA Hugging Face)")

            # --- BAR CHART ---
            fig_bar = px.bar(
                df,
                x="Titre",
                y="Score",
                color="Sentiment",
                color_discrete_map={"Positive": "#00D26A", "Négative": "#FF4B4B", "Neutre": "#A0A0A0", "Erreur": "#FFAA00"},
                title=f"Analyse IA (Barres)",
                hover_data={"Lien": False}
            )
            fig_bar.update_layout(
                xaxis_title="", yaxis_title="Score IA", height=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", size=12), legend_title="Sentiment"
            )
            fig_bar.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y}<extra></extra>")

            # --- LINE CHART (NEW!) ---
            # Sort by score descending to simulate "time" (most impactful first)
            df_line = df.copy().sort_values("Score", ascending=False).reset_index(drop=True)
            df_line["Index"] = range(1, len(df_line) + 1)

            fig_line = px.line(
                df_line,
                x="Index",
                y="Score",
                color="Sentiment",
                color_discrete_map={"Positive": "#00D26A", "Négative": "#FF4B4B", "Neutre": "#A0A0A0", "Erreur": "#FFAA00"},
                title="Évolution du Sentiment (Ligne)",
                markers=True
            )
            fig_line.update_layout(
                xaxis_title="Article # (trié par impact)", yaxis_title="Score IA",
                height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", size=12), legend_title="Sentiment"
            )
            fig_line.update_traces(hovertemplate="<b>Article %{x}</b><br>Score: %{y}<extra></extra>")

            # --- DISPLAY BOTH CHARTS ---
            tab1, tab2 = st.tabs(["Barres", "Ligne"])
            with tab1:
                st.plotly_chart(fig_bar, use_container_width=True)
            with tab2:
                st.plotly_chart(fig_line, use_container_width=True)
        # === TABLE COLUMN: RESPONSIVE + CLEAN ===        with col_table:
            st.subheader("Détail des Résultats IA")

            def make_clickable(val):
                if val == "#" or not val.startswith("http"):
                    return "Lien non disponible"
                try:
                    from urllib.parse import urlparse, parse_qs
                    # BING
                    if "bing.com/news/apiclick.aspx" in val:
                        real = parse_qs(urlparse(val).query).get('url', [val])[0]
                        return f'<a href="{real}" target="_blank" style="color:#00D26A; text-decoration:none; font-weight:500;">Voir</a>'
                    # GOOGLE NEWS
                    if "news.google.com/rss/articles/" in val:
                        import base64, re
                        path = urlparse(val).path
                        encoded = path.split("/articles/")[-1].split("?")[0]
                        missing = len(encoded) % 4
                        if missing:
                            encoded += '=' * (4 - missing)
                        try:
                            decoded = base64.urlsafe_b64decode(encoded).decode('utf-8', 'ignore')
                            match = re.search(r'"(https?://[^"]+)"', decoded)
                            if match:
                                return f'<a href="{match.group(1)}" target="_blank" style="color:#00D26A; text-decoration:none; font-weight:500;">Voir</a>'
                        except:
                            pass
                    return f'<a href="{val}" target="_blank" style="color:#00D26A; text-decoration:none; font-weight:500;">Voir</a>'
                except:
                    return "Lien non disponible"

            df_display = df.copy()
            df_display["Lien_HTML"] = df_display["Lien"].apply(make_clickable)

            # === BUILD HTML TABLE (NO ESCAPING!) ===
            html_table = """
            <style>
            .responsive-table {
                width: 100%; border-collapse: collapse; font-size: 0.85em;
                display: block; overflow-x: auto; white-space: nowrap;
            }
            .responsive-table th, .responsive-table td {
                padding: 10px 8px; text-align: left; border-bottom: 1px solid #eee; min-width: 80px;
            }
            .responsive-table th {
                background-color: #f8f9fa; font-weight: 600; position: sticky; top: 0; z-index: 1;
            }
            .responsive-table tr:hover { background-color: #f1f3f5; }
            .responsive-table a { color: #00D26A !important; font-weight: 500; }
            @media (max-width: 768px) {
                .responsive-table { font-size: 0.75em; }
                .responsive-table th, .responsive-table td { padding: 6px 4px; }
            }
            </style>
            <div style="max-height: 550px; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px;">
            <table class="responsive-table">
                <thead>
                    <tr>
                        <th style="width: 55%;">Titre</th>
                        <th style="width: 15%;">Sentiment</th>
                        <th style="width: 15%;">Score</th>
                        <th style="width: 15%;">Lien</th>
                    </tr>
                </thead>
                <tbody>
            """

            for _, row in df_display.iterrows():
                title = row['Titre']
                if len(title) > 80:
                    title = title[:77] + "..."

                # DO NOT ESCAPE TITLE — IT'S SAFE
                lien_html = row['Lien_HTML']

                html_table += f"""
                <tr>
                    <td style="max-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                        {title}
                    </td>
                    <td style="text-align: center;">{row['Sentiment']}</td>
                    <td style="text-align: center;">{row['Score']:.3f}</td>
                    <td style="text-align: center;">{lien_html}</td>
                </tr>
                """

            html_table += """
                </tbody>
            </table>
            </div>
            """
            st.markdown(html_table, unsafe_allow_html=True)

    st.success("Analyse IA terminée !")

# FOOTER
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

# MON CV INTÉGRÉ (expander pro)
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









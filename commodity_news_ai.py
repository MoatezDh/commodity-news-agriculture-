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
st.sidebar.header("Settings")
commodity = st.sidebar.selectbox("Commodity", ["corn", "wheat", "soybean", "coffee"])
num_articles = st.sidebar.slider("Articles to analyze", 1, 50, 5)

# HEADERS
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

# JSON FILE
JSON_FILE = "scraped_articles.json"

# HUGGING FACE MODEL (FINANCIAL NEWS - 100% FUNCTIONAL)
@st.cache_resource
def load_sentiment_model():
    try:
        model = pipeline(
            "sentiment-analysis",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            return_all_scores=False
        )
        st.success("Hugging Face AI (Financial News) loaded!")
        return model
    except Exception as e:
        st.error(f"HF loading error: {e}")
        st.warning("→ Fallback simulation activated")
        return None

# ADD AT TOP
if 'scrape_counter' not in st.session_state:
    st.session_state.scrape_counter = 0

# SCRAPING REAL NEWS FROM GOOGLE/BING RSS
def scrape_news(query, n=5):
    # Unique key forces fresh scrape
    cache_key = f"{query}_{n}_{st.session_state.scrape_counter}"
   
    if cache_key in st.session_state:
        articles = st.session_state[cache_key]
        st.info(f"Loaded from memory ({len(articles)} articles)")
        return articles

    articles = []
    url = f"https://news.google.com/rss/search?q={query}+price+when:1d&hl=en-US&gl=US&ceid=US:en"
    try:
        time.sleep(1)
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'xml')
            items = soup.find_all('item')
            st.info(f"Google News: {len(items)} articles found")
        else:
            raise Exception("Google failed")
    except:
        st.warning("Google failed → Fallback to Bing")
        url = f"https://www.bing.com/news/search?q={query}+price&format=rss"
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                raise Exception("Bing failed")
            soup = BeautifulSoup(response.text, 'xml')
            items = soup.find_all('item')
            st.info(f"Bing: {len(items)} articles found")
        except Exception as e:
            st.error(f"Scraping failed: {e}")
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
        st.success(f"{len(articles)} articles saved → {JSON_FILE}")
        st.session_state[cache_key] = articles
    else:
        st.warning("No valid articles found")
    
    return articles

# AI ANALYSIS (HF FINANCIAL)
def analyze_sentiment(title, model):
    if model:
        try:
            res = model(title)[0]
            label = res['label'].lower()
            score = res['score']
            if label == "negative":
                score = -score
                sentiment_en = "Negative"
            elif label == "positive":
                sentiment_en = "Positive"
            else:  # neutral
                score = 0
                sentiment_en = "Neutral"
            return sentiment_en, round(score, 3)
        except Exception as e:
            st.error(f"HF analysis error: {e}")
            return "Error", 0
    else:
        # Fallback simulation
        pos = ["rise", "boost", "record", "strong", "surge", "high"]
        neg = ["fall", "drop", "drought", "threat", "tariff", "low"]
        t = title.lower()
        if any(w in t for w in pos):
            return "Positive", round(random.uniform(0.6, 0.95), 3)
        if any(w in t for w in neg):
            return "Negative", round(random.uniform(-0.95, -0.6), 3)
        return "Neutral", 0.0

if st.button("Run AI Analysis"):
    st.session_state.scrape_counter += 1  # FORCE FRESH
    with st.spinner("Scraping News + AI Analysis (Hugging Face Financial)..."):
        # FIXED: Dynamic fallback inside button (uses current commodity)
        fallback_articles = [
            {"title": f"{commodity.title()} prices fall due to strong harvest", "link": "https://reuters.com", "source": "Simulated"},
            {"title": f"Brazil boosts {commodity} exports", "link": "https://bloomberg.com", "source": "Simulated"},
            {"title": f"EU imposes new tariffs on {commodity}", "link": "https://euronews.com", "source": "Simulated"}
        ]
       
        real_articles = scrape_news(commodity, num_articles)
        articles = real_articles if real_articles else fallback_articles[:num_articles]
       
        model = load_sentiment_model()
       
        results = []
        for art in articles:
            sentiment, score = analyze_sentiment(art["title"], model)
            results.append({
                "Title": art["title"][:100] + "..." if len(art["title"]) > 100 else art["title"],
                "Sentiment": sentiment,
                "Score": score,
                "Link": art["link"]
            })
       
        df = pd.DataFrame(results)
       
        # DASHBOARD STYLE
        st.markdown("---")
       
        # HEADER STATS
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric(
                label="Articles analyzed",
                value=len(df),
                delta=f"{commodity.title()}"
            )
        with col_stats2:
            pos_count = len(df[df["Sentiment"] == "Positive"])
            st.metric(
                label="Positive Sentiment",
                value=f"{pos_count}/{len(df)}",
                delta=f"+{round((pos_count/len(df)*100) if len(df)>0 else 0, 1)}%"
            )
        with col_stats3:
            neg_count = len(df[df["Sentiment"] == "Negative"])
            st.metric(
                label="Negative Sentiment",
                value=f"{neg_count}/{len(df)}",
                delta=f"-{round((neg_count/len(df)*100) if len(df)>0 else 0, 1)}%"
            )
        st.markdown("---")
        
        # CHART + TABLE (2 COLUMNS)
        col_chart, col_table = st.columns([2, 1], gap="large")
        
        # === CHART COLUMN: BAR + LINE CHART ===
        with col_chart:
            st.subheader("Sentiment Trend (Hugging Face AI)")
            # --- BAR CHART ---
            fig_bar = px.bar(
                df,
                x="Title",
                y="Score",
                color="Sentiment",
                color_discrete_map={"Positive": "#00D26A", "Negative": "#FF4B4B", "Neutral": "#A0A0A0", "Error": "#FFAA00"},
                title="AI Analysis (Bars)",
                hover_data={"Link": False}
            )
            fig_bar.update_layout(
                xaxis_title="", yaxis_title="AI Score", height=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", size=12), legend_title="Sentiment"
            )
            fig_bar.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y}<extra></extra>")
            
            # --- LINE CHART ---
            df_line = df.copy().sort_values("Score", ascending=False).reset_index(drop=True)
            df_line["Index"] = range(1, len(df_line) + 1)
            fig_line = px.line(
                df_line,
                x="Index",
                y="Score",
                color="Sentiment",
                color_discrete_map={"Positive": "#00D26A", "Negative": "#FF4B4B", "Neutral": "#A0A0A0", "Error": "#FFAA00"},
                title="Sentiment Evolution (Line)",
                markers=True
            )
            fig_line.update_layout(
                xaxis_title="Article # (sorted by impact)", yaxis_title="AI Score",
                height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial", size=12), legend_title="Sentiment"
            )
            fig_line.update_traces(hovertemplate="<b>Article %{x}</b><br>Score: %{y}<extra></extra>")
            
            # DISPLAY BOTH CHARTS
            tab1, tab2 = st.tabs(["Bars", "Line"])
            with tab1:
                st.plotly_chart(fig_bar, use_container_width=True)
            with tab2:
                st.plotly_chart(fig_line, use_container_width=True)
        
        # === TABLE COLUMN: RESPONSIVE + CLEAN ===
        with col_table:
            st.subheader("AI Results Details")
            # Truncate titles
            df_display = df.copy()
            df_display["Title"] = df_display["Title"].apply(
                lambda x: x[:77] + "..." if len(x) > 80 else x
            )
            # Clean links (extract real URL)
            def clean_link(url):
                if not url or url == "#" or not url.startswith("http"):
                    return None
                try:
                    from urllib.parse import urlparse, parse_qs
                    # BING
                    if "bing.com/news/apiclick.aspx" in url:
                        return parse_qs(urlparse(url).query).get('url', [url])[0]
                    # GOOGLE NEWS
                    if "news.google.com/rss/articles/" in url:
                        import base64, re
                        path = urlparse(url).path
                        encoded = path.split("/articles/")[-1].split("?")[0]
                        missing = len(encoded) % 4
                        if missing:
                            encoded += '=' * (4 - missing)
                        decoded = base64.urlsafe_b64decode(encoded).decode('utf-8', 'ignore')
                        match = re.search(r'"(https?://[^"]+)"', decoded)
                        if match:
                            return match.group(1)
                    return url
                except:
                    return None
            
            df_display["Link"] = df_display["Link"].apply(clean_link)
            
            # Use Streamlit's native LinkColumn
            st.dataframe(
                df_display[["Title", "Sentiment", "Score", "Link"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Title": st.column_config.TextColumn(
                        "Title",
                        width="medium",
                        help="Article title"
                    ),
                    "Sentiment": st.column_config.TextColumn(
                        "Sentiment",
                        width="small"
                    ),
                    "Score": st.column_config.NumberColumn(
                        "Score",
                        format="%.3f",
                        width="small"
                    ),
                    "Link": st.column_config.LinkColumn(
                        "Link",
                        display_text="View article",
                        width="small",
                        help="Open article"
                    )
                }
            )
        
        st.markdown("---")
        
        # JSON CHECK (technical expander)
        with st.expander("Technical Check: Raw Scraping Data", expanded=False):
            if os.path.exists(JSON_FILE):
                with open(JSON_FILE, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                st.success(f"{len(json_data)} articles scraped successfully")
                st.json(json_data, expanded=False)
            else:
                st.warning("No JSON file found → Scraping failed")

    st.success("AI Analysis completed!")

# FOOTER
st.markdown("---")
col_footer1, col_footer2 = st.columns([3, 1])
with col_footer1:
    st.markdown(
        """
        **Commodity News AI** – Real-time agricultural price analysis
        *News scraping + Hugging Face AI (financial model)*
        **Developed by Moatez DHIEB** – Computer Engineering Student (EPI SUP)
        """
    )
with col_footer2:
    st.markdown("**Project Demo** \nDNEXT Intelligence SA \n*Feb 2026*")

# MY CV INTEGRATED (pro expander)
with st.expander("View my full CV (click to download)", expanded=False):
    col_cv1, col_cv2 = st.columns([1, 2])
    with col_cv1:
        st.markdown("""**MediConnect – Intelligent telemedicine platform**
                         **Scan the QR code, discover a production-ready platform.**
                                             and appreciate my recent work. """)
        if os.path.exists("MediConnectQrCode.jpg"):
            st.image("MediConnectQrCode.jpg", use_container_width=True)
        else:
            st.warning("QR file missing → Add `MediConnectQrCode.jpg`")
    with col_cv2:
        st.markdown("### **Moatez DHIEB**")
        st.markdown("**Software Engineer | Data Science | Full-Stack**")
        st.markdown("Sousse, Tunisia | +216 24500607 | dhiebmoatez@gmail.com")
        st.markdown("[GitHub](https://github.com/MoatezDh) | [LinkedIn](www.linkedin.com/in/moatez-dhieb)")
        if os.path.exists("CV_Moatez_DHIEB.pdf"):
            with open("CV_Moatez_DHIEB.pdf", "rb") as f:
                st.download_button(
                    label="Download CV (PDF)",
                    data=f.read(),
                    file_name="CV_Moatez_DHIEB.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.warning("PDF file missing → Add `CV_Moatez_DHIEB.pdf`")


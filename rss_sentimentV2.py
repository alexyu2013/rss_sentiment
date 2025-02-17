import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import feedparser
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already done
nltk.download('vader_lexicon', quiet=True)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker)
    return data

def add_ema(data, periods):
    for period in periods:
        data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

def add_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def add_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

@st.cache_data
def get_fundamental_metrics(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    metrics = {
        'P/E Ratio': info.get('trailingPE', 'N/A'),
        'ROE': info.get('returnOnEquity', 'N/A'),
        'ROA': info.get('returnOnAssets', 'N/A'),
        'Gross Margin': info.get('grossMargins', 'N/A'),
        'Profit Margin': info.get('profitMargins', 'N/A'),
        'Debt to Equity': info.get('debtToEquity', 'N/A'),
        'Current Ratio': info.get('currentRatio', 'N/A'),
        'Price to Book': info.get('priceToBook', 'N/A'),
        'Earnings Per Share': info.get('trailingEps', 'N/A'),
        'Dividend Yield': info.get('dividendYield', 'N/A')
    }
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            metrics[key] = round(value, 2)
        elif value == 'N/A':
            metrics[key] = 'N/A'
        else:
            try:
                metrics[key] = round(float(value), 2)
            except ValueError:
                metrics[key] = 'N/A'
    
    return metrics

@st.cache_data
def fetch_rss_feed(ticker):
    feed_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(feed_url)
    return feed

def get_vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

st.title('交互式智能股票分析：整合技术指标、基本面数据与新闻情绪')

# Sidebar for user inputs and news feed
st.sidebar.title('股票情绪智能分析')
ticker = st.sidebar.text_input('输入股票代码（目前仅支持美股）', 'TLSA').upper()

# Main content
data = load_data(ticker)

periods = st.slider('选择时间段（以天为单位）', 30, 365, 180)

selected_emas = st.multiselect('选择 EMA 周期', [200, 50, 20], default=[200, 50, 20])

add_rsi_plot = st.checkbox('添加 RSI 分析图')
add_macd_plot = st.checkbox('添加 MACD 分析图')

st.subheader('选择要显示的技术指标')
metrics = get_fundamental_metrics(ticker)
selected_metrics = st.multiselect('选择指标', list(metrics.keys()), default=['P/E Ratio', 'ROE', 'Profit Margin'])

if selected_metrics:
    st.subheader('基本指标')
    for i in range(0, len(selected_metrics), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(selected_metrics):
                metric = selected_metrics[i + j]
                cols[j].metric(label=metric, value=metrics[metric])

data = add_ema(data, selected_emas)

data_period = data[-periods:]

if add_rsi_plot:
    data_period = add_rsi(data_period)

if add_macd_plot:
    data_period = add_macd(data_period)

rows = 1 + add_rsi_plot + add_macd_plot

price_range = [data_period['Close'].min() * 0.95, data_period['Close'].max() * 1.05]
rsi_range = [0, 100]
macd_range = [0, 0]

if add_macd_plot:
    macd_range = [
        min(data_period['MACD'].min(), data_period['Signal Line'].min()) * 1.05,
        max(data_period['MACD'].max(), data_period['Signal Line'].max()) * 1.05
    ]

fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    vertical_spacing=0.15,
                    row_heights=[0.5] + [0.25] * (rows - 1),
                    subplot_titles=('Price', 'RSI', 'MACD')[:rows])

fig.add_trace(go.Candlestick(x=data_period.index,
                             open=data_period['Open'],
                             high=data_period['High'],
                             low=data_period['Low'],
                             close=data_period['Close'],
                             name='Candlesticks'), row=1, col=1)

for period in selected_emas:
    fig.add_trace(go.Scatter(x=data_period.index, y=data_period[f'EMA_{period}'], mode='lines', name=f'EMA_{period}'), row=1, col=1)

current_row = 2
if add_rsi_plot:
    fig.add_trace(go.Scatter(x=data_period.index, y=data_period['RSI'], mode='lines', name='RSI'), row=current_row, col=1)
    fig.update_yaxes(range=rsi_range, row=current_row, col=1, title='RSI')
    current_row += 1

if add_macd_plot:
    fig.add_trace(go.Scatter(x=data_period.index, y=data_period['MACD'], mode='lines', name='MACD'), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=data_period.index, y=data_period['Signal Line'], mode='lines', name='Signal Line'), row=current_row, col=1)
    fig.update_yaxes(range=macd_range, row=current_row, col=1, title='MACD')

fig.update_layout(
    title=f'{ticker} 股票价格和指标',
    xaxis_title='Date',
    yaxis_title='Price',
    height=400 + 200 * (rows - 1),
    margin=dict(l=50, r=50, t=50, b=50),
    legend=dict(x=0, y=1, traceorder='normal'),
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)

st.plotly_chart(fig)

# RSS feed in the sidebar with sentiment analysis
st.sidebar.subheader(f"基于智能情感分析的 {ticker} 股票情报")
st.sidebar.text("Loading news...")  # Simple loading message

feed = fetch_rss_feed(ticker)

if feed.entries:
    news_items = []

    for entry in feed.entries:
        sentiment = get_vader_sentiment(entry.title)
        compound_score = sentiment['compound']

        # Determine sentiment category
        if compound_score >= 0.05:
            sentiment_category = "积极"
            color = "red"
        elif compound_score <= -0.05:
            sentiment_category = "消极"
            color = "green"
        else:
            sentiment_category = "中性"
            color = "gray"

        news_items.append({
            'title': entry.title,
            'link': entry.link,
            'published': entry.published,
            'sentiment_category': sentiment_category,
            'compound_score': compound_score,
            'color': color
        })

    # Sort news items by published date (latest first) and take the latest 10
    latest_10_news = sorted(news_items, key=lambda x: x['published'], reverse=True)[:10]

    # Sort the latest 10 news by sentiment score (highest to lowest)
    latest_10_news.sort(key=lambda x: x['compound_score'], reverse=True)

    # Calculate total sentiment score for the latest 10 news items
    total_sentiment_score = sum(item['compound_score'] for item in latest_10_news)

    # Display total sentiment score in red
    st.sidebar.markdown(f"<h3 style='color: red;'>总情绪得分: {total_sentiment_score:.2f}</h3>", unsafe_allow_html=True)

    # Display sorted news items
    for item in latest_10_news:
        st.sidebar.markdown(f"**{item['title']}**")
        st.sidebar.markdown(f"[Read more]({item['link']})")
        st.sidebar.markdown(f"*发布时间: {item['published']}*")
        st.sidebar.markdown(f"情绪: <span style='color:{item['color']}'>{item['sentiment_category']}</span> (Score: {item['compound_score']:.2f})", unsafe_allow_html=True)
        st.sidebar.markdown("---")
else:
    st.sidebar.write("对不起，暂未找该股票代码的新闻.")

# Remove the loading message after fetching is complete
st.sidebar.empty()

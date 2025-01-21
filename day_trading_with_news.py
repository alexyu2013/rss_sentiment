
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
def load_data(ticker, interval='1m', period='1d'):
    data = yf.download(ticker, interval=interval, period=period)
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
def fetch_rss_feed(ticker):
    feed_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(feed_url)
    return feed

def get_vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def main():
    st.title('交互式智能股票分析：日内交易')

    # Sidebar for user inputs and news feed
    st.sidebar.title('股票情绪智能分析')
    ticker = st.sidebar.text_input('输入股票代码（目前仅支持美股）', 'AAPL').upper()
    
    interval = st.sidebar.selectbox('选择时间间隔', ['1m', '5m', '15m'], index=1)
    period = '1d'  # For day trading, we'll use 1-day period

    # Main content
    data = load_data(ticker, interval, period)

    selected_emas = st.multiselect('选择 EMA 周期', [9, 20, 50], default=[9, 20])

    add_rsi_plot = st.checkbox('添加 RSI 分析图', value=True)
    add_macd_plot = st.checkbox('添加 MACD 分析图', value=True)

    data = add_ema(data, selected_emas)

    if add_rsi_plot:
        data = add_rsi(data)

    if add_macd_plot:
        data = add_macd(data)

    rows = 1 + add_rsi_plot + add_macd_plot

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[0.5] + [0.25] * (rows - 1),
                        subplot_titles=('Price', 'RSI', 'MACD')[:rows])

    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlesticks'), row=1, col=1)

    for period in selected_emas:
        fig.add_trace(go.Scatter(x=data.index, y=data[f'EMA_{period}'], mode='lines', name=f'EMA_{period}'), row=1, col=1)

    current_row = 2
    if add_rsi_plot:
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'), row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1, title='RSI')
        current_row += 1

    if add_macd_plot:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Signal Line'], mode='lines', name='Signal Line'), row=current_row, col=1)
        fig.add_bar(x=data.index, y=data['MACD'] - data['Signal Line'], name='MACD Histogram', row=current_row, col=1)
        fig.update_yaxes(title='MACD', row=current_row, col=1)

    fig.update_layout(
        title=f'{ticker} 日内交易图表 ({interval} 间隔)',
        xaxis_title='Time',
        yaxis_title='Price',
        height=800,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # News feed with sentiment analysis
    st.sidebar.subheader(f"基于智能情感分析的 {ticker} 股票情报")
    
    feed = fetch_rss_feed(ticker)

    if feed.entries:
        news_items = []

        for entry in feed.entries[:10]:  # Limit to 10 latest news items
            sentiment = get_vader_sentiment(entry.title)
            compound_score = sentiment['compound']

            if compound_score >= 0.05:
                sentiment_category = "积极"
                color = "green"
            elif compound_score <= -0.05:
                sentiment_category = "消极"
                color = "red"
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

        # Calculate total sentiment score
        total_sentiment_score = sum(item['compound_score'] for item in news_items)

        # Display total sentiment score
        st.sidebar.markdown(f"<h3 style='color: {'green' if total_sentiment_score >= 0 else 'red'}'>总情绪得分： {total_sentiment_score:.2f}</h3>", unsafe_allow_html=True)

        # Display news items
        for item in news_items:
            st.sidebar.markdown(f"**{item['title']}**")
            st.sidebar.markdown(f"[Read more]({item['link']})")
            st.sidebar.markdown(f"*发布时间: {item['published']}*")
            st.sidebar.markdown(f"情绪: <span style='color:{item['color']}'>{item['sentiment_category']}</span> (Score: {item['compound_score']:.2f})", unsafe_allow_html=True)
            st.sidebar.markdown("---")
    else:
        st.sidebar.write("对不起，暂未找该股票代码的新闻.")

if __name__ == "__main__":
    main()

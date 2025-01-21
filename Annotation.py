import streamlit as st
from annotated_text import annotated_text
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import base64
# Ensure the required NLTK data is downloaded
nltk.download('punkt')
nltk.download('vader_lexicon', quiet=True)  # Optional for sentiment analysis

# Now you can safely use the sentence tokenizer
from nltk.tokenize import sent_tokenize

# Function to load the image and convert it to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to the locally stored QR code image
qr_code_path = "qrcode.png"  # Ensure the image is in your app directory

# Convert image to base64
qr_code_base64 = get_base64_of_bin_file(qr_code_path)

# Custom CSS to position the QR code close to the top-right corner under the "Deploy" area
st.markdown(
    f"""
    <style>
    .qr-code {{
        position: fixed;  /* Keeps the QR code fixed in the viewport */
        top: 10px;       /* Sets the distance from the top of the viewport */
        right: 10px;     /* Sets the distance from the right of the viewport */
        width: 200px;    /* Adjusts the width of the QR code */
        z-index: 100;    /* Ensures the QR code stays above other elements */
    }}
    </style>
    <img src="data:image/png;base64,{qr_code_base64}" class="qr-code">
    """,
    unsafe_allow_html=True
)


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon', quiet=True)

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Function to annotate words with sentiment
def annotate_word_with_sentiment(message):
    words = message.split()  # Split message into words
    annotated_parts = []

    for word in words:
        sentiment = sia.polarity_scores(word)
        sentiment_score = sentiment['compound']

        # Annotate words based on sentiment score
        if sentiment_score >= 0.5:
            annotated_parts.append((word, "Positive", "#8ef"))
        elif sentiment_score >= 0.05:
            annotated_parts.append((word, "Positive", "#aef"))
        elif sentiment_score <= -0.5:
            annotated_parts.append((word, "Negative", "#faa"))
        elif sentiment_score <= -0.05:
            annotated_parts.append((word, "Negative", "#f77"))
        else:
            annotated_parts.append((word, "Neutral", "#aaa"))

    return annotated_parts

# Function to annotate sentences with sentiment
def annotate_sentence_with_sentiment(message):
    sentences = nltk.sent_tokenize(message)  # Tokenize message into sentences
    annotated_parts = []

    for sentence in sentences:
        sentiment = sia.polarity_scores(sentence)
        sentiment_score = sentiment['compound']

        # Annotate sentences based on sentiment score
        if sentiment_score >= 0.5:
            annotated_parts.append((sentence, "Positive", "#8ef"))
        elif sentiment_score >= 0.05:
            annotated_parts.append((sentence, "Positive", "#aef"))
        elif sentiment_score <= -0.5:
            annotated_parts.append((sentence, "Negative", "#faa"))
        elif sentiment_score <= -0.05:
            annotated_parts.append((sentence, "Negative", "#f77"))
        else:
            annotated_parts.append((sentence, "Neutral", "#aaa"))

    return annotated_parts

# Function to annotate paragraphs with sentiment
def annotate_paragraph_with_sentiment(message):
    paragraphs = message.split('\n')  # Split the message into paragraphs
    annotated_parts = []

    for paragraph in paragraphs:
        if paragraph.strip():  # Skip empty paragraphs
            sentiment = sia.polarity_scores(paragraph)
            sentiment_score = sentiment['compound']

            # Annotate paragraphs based on sentiment score
            if sentiment_score >= 0.5:
                annotated_parts.append((paragraph, "Positive", "#8ef"))
            elif sentiment_score >= 0.05:
                annotated_parts.append((paragraph, "Positive", "#aef"))
            elif sentiment_score <= -0.5:
                annotated_parts.append((paragraph, "Negative", "#faa"))
            elif sentiment_score <= -0.05:
                annotated_parts.append((paragraph, "Negative", "#f77"))
            else:
                annotated_parts.append((paragraph, "Neutral", "#aaa"))

    return annotated_parts

# Streamlit app
def sentiment_annotation_app():
    st.title("Sentiment Annotation App")
    st.write("This app annotates words, sentences, and paragraphs with sentiment analysis. Enter some text below.")

    user_message = st.text_area("Enter your message:")

    if user_message:
        # Word-level sentiment annotation
        st.subheader("Word-level Sentiment Annotation")
        word_annotations = annotate_word_with_sentiment(user_message)
        annotated_text(*word_annotations)

        st.write("\n")  # Add space between sections

        # Sentence-level sentiment annotation
        st.subheader("Sentence-level Sentiment Annotation")
        sentence_annotations = annotate_sentence_with_sentiment(user_message)
        annotated_text(*sentence_annotations)

        st.write("\n")  # Add space between sections

        # Paragraph-level sentiment annotation
        st.subheader("Paragraph-level Sentiment Annotation")
        paragraph_annotations = annotate_paragraph_with_sentiment(user_message)
        annotated_text(*paragraph_annotations)

# Run the Streamlit app
if __name__ == "__main__":
    sentiment_annotation_app()

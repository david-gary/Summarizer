from utils import *
from analyzer import SentimentAnalyzer
import streamlit as st


def main():
    st.title("Sentiment Analysis Tool")

    # dropdown side menu for selecting dataset or custom text
    st.sidebar.title("Select Dataset")
    # default value is "Custom Text"

    dataset = st.sidebar.selectbox(
        "Select Dataset", ["Custom Text", "Multi-News", "Reddit", "S2ORC"])

    default_text = "The Russell/Norvig Book is amazing."
    if dataset == "Custom Text":
        text = st.text_area("Enter text to analyze:", default_text)
    elif dataset == "Multi-News":
        text = grab_random_multinews()[0]
    elif dataset == "Reddit":
        text = grab_random_reddit()[0]
    elif dataset == "S2ORC":
        text = grab_random_s2orc()[0]

    # button to run sentiment analysis
    if st.button("Run Sentiment Analysis"):
        analyzer = SentimentAnalyzer()
        sa_report = analyzer.predict(text)
        label, score = sa_report[0]['label'], sa_report[0]['score']
        st.write(f"Input Text: {text}")
        st.write(f"Label: {label}")
        st.write(f"Score: {score}")

    # # button to run all tests
    # if st.button("Run All Tests"):
    #     test_all_grabs()
    #     test_sentiment_analysis_basic()
    #     test_sentiment_analysis_input()


if __name__ == "__main__":
    main()

from analyzer import SentimentAnalyzer, XLNetBuilder


def test_sentiment_analysis_basic():
    analyzer = SentimentAnalyzer()
    love_score = analyzer.predict("I love you")
    hate_score = analyzer.predict("I hate you")
    print(f"Love score: {love_score}")
    print(f"Hate score: {hate_score}")


def test_sentiment_analysis_input():
    analyzer = SentimentAnalyzer()
    input_text = input("Enter text to analyze: ")
    sa_report = analyzer.predict(input_text)
    label, score = sa_report[0]['label'], sa_report[0]['score']
    print(f"Input Text: {input_text}")
    print(f"Label: {label}")
    print(f"Score: {score}")


def main():
    test_sentiment_analysis_basic()
    test_sentiment_analysis_input()

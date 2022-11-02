from transformers import pipeline, XLNetConfig, XLNetModel, XLNetTokenizer


class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline("sentiment-analysis")

    def predict(self, text):
        return self.model(text)


class XLNetBuilder:
    def __init__(self, vocab_size=32000, d_model=1024, n_layer=24):
        self.config = XLNetConfig(
            vocab_size=vocab_size, d_model=d_model, n_layer=n_layer)
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.model = XLNetModel(self.config)

    def build(self):
        return self.model, self.tokenizer

    def save_model(self, model_path):
        self.model.save_pretrained(model_path)

    def sentiment_analysis(self, text):
        return self.model(text)

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from summarizer import TransformerSummarizer
import pandas as pd


class SentimentAnalyzer:
    def __init__(self, max_input_size=1024):
        self.model = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment',
                              tokenizer='nlptown/bert-base-multilingual-uncased-sentiment', max_length=max_input_size)
        self.max_input_size = max_input_size

    def predict(self, text):
        if len(text) > self.max_input_size:
            text = text[:self.max_input_size]
        return self.model(text)


class SummarizationSuite:

    def __init__(self, model_type='pegasusx', max_length=100, min_length=5):
        self.model_type = model_type
        self.model_name = None
        self.description = None
        self.config = None
        self.texts = None
        self.tokenizer = None
        self.model = None
        self.max_length = max_length
        self.min_length = min_length

    def set_model_name(self):
        if self.model_type == 'xlnet':
            self.model_name = 'xlnet-base-cased'
        elif self.model_type == 'bart':
            self.model_name = 'facebook/bart-large-cnn'
        elif self.model_type == 'bartx':
            self.model_name = 'facebook/bart-large-xsum'
        elif self.model_type == 't5':
            self.model_name = 't5-base'
        elif self.model_type == 'pegasus':
            self.model_name = 'google/pegasus-large'
        elif self.model_type == 'pegasusx':
            self.model_name = 'google/pegasus-xsum'

    def set_config(self):
        """
        Loads and sets the model configuration using AutoModelForSeq2SeqLM and AutoConfig from the transformers library.
        """
        if self.model_type == 'xlnet':
            pass
        else:
            # should directly load from the model hub
            self.config = AutoConfig.from_pretrained(self.model_name)

    def set_description(self):
        """
        Sets the description of the summarization suite by referencing the model type used.
        Descriptions are found in txt files inside the `models/descriptions` folder.
        """

        if self.model_type == 'xlnet':
            self.description = open(
                'models/descriptions/xlnet.txt', 'r').read()
        elif self.model_type == 'bart':
            self.description = open(
                'models/descriptions/bart.txt', 'r').read()
        elif self.model_type == 'bartx':
            self.description = open(
                'models/descriptions/bartx.txt', 'r').read()
        elif self.model_type == 't5':
            self.description = open(
                'models/descriptions/t5.txt', 'r').read()
        elif self.model_type == 'pegasus':
            self.description = open(
                'models/descriptions/pegasus.txt', 'r').read()
        elif self.model_type == 'pegasusx':
            self.description = open(
                'models/descriptions/pegasusx.txt', 'r').read()

    def build_text_records(self):
        """
        Builds an empty dataframe with two columns: `text` and `summary`.
        """
        self.texts = pd.DataFrame(columns=['text', 'summary'])

    def build_model(self):
        """
        Builds the model for summarization from the following:
        - xlnet (custom made)
        - bart
        - bartx
        - t5
        - pegasus
        - pegasusx
        """
        self.set_model_name()
        self.set_config()
        self.set_description()

        if self.model_type == 'xlnet':
            self.model = TransformerSummarizer(
                transformer_type="XLNet", transformer_model_key="xlnet-base-cased")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, config=self.config)

    def build_tokenizer(self):
        """
        Builds the proper tokenizer for the summarization suite.
        """
        if self.model_type == 'xlnet':
            pass
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def save_model(self):
        """
        Saves the model to the `models` folder.
        """
        # check if there is already a saved model of the same type
        # if there is, delete it
        self.model.save_pretrained('models/saved/{}'.format(self.model_type))

    def load_model(self):
        """
        Loads the model from the `models` folder.
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            'models/saved/{}'.format(self.model_type))

    def summarization(self, input_text=None):
        """
        Generates a new summarization using the model already loaded.
        """

        if input_text is None:
            return "No input text provided."

        if self.model_type == 'xlnet':
            return self.model(input_text, min_length=self.min_length, max_length=self.max_length)
        else:
            input_ids = self.tokenizer.encode(
                input_text, return_tensors='pt', max_length=self.max_length)
            summary_ids = self.model.generate(
                input_ids, num_beams=4, max_length=self.max_length, min_length=self.min_length, early_stopping=True)
            summary = self.tokenizer.decode(
                summary_ids[0], skip_special_tokens=True)
            return summary

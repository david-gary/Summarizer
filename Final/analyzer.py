from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import pandas as pd


class SentimentAnalyzer:
    def __init__(self, tokenizer=None, model=None, device=None):
        self.tokenizer = tokenizer
        self.model = model or pipeline(
            'sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment', tokenizer='cardiffnlp/twitter-roberta-base-sentiment')

    def predict(self, text):
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
        - t5
        - pegasus
        - pegasusx
        """
        self.set_model_name()
        self.set_config()
        self.set_description()

        # Load the model from the HuggingFace model hub using AutoModelForSeq2SeqLM
        # requires credentials to be set up

        new_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, config=self.config)
        self.model = new_model

    def build_tokenizer(self):
        """
        Builds the proper tokenizer for the summarization suite.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def summarization(self, input_text=None):
        """
        Generates a new summarization using the model already loaded.
        """

        input_ids = self.tokenizer.encode(
            input_text, return_tensors='pt', max_length=self.max_length)
        summary_ids = self.model.generate(
            input_ids, num_beams=4, max_length=self.max_length, min_length=self.min_length, early_stopping=True)
        output = self.tokenizer.decode(
            summary_ids[0], skip_special_tokens=True)
        return output

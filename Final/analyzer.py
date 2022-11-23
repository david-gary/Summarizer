from transformers import pipeline, XLNetConfig, XLNetModel, XLNetTokenizer, BartTokenizer, BartModel, BartConfig, T5Config, T5Model, T5Tokenizer, PegasusConfig, PegasusModel, PegasusTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from torch import nn, optim, device, cuda
from rouge_score import rouge_scorer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances, euclidean_distances, manhattan_distances, cosine_distances, pairwise_kernels
import pandas as pd


class SentimentAnalyzer:
    def __init__(self, tokenizer, model=None):
        self.tokenizer = tokenizer
        self.model = model or pipeline(
            'sentiment-analysis', model='xlnet-base-cased', tokenizer=tokenizer)

    def predict(self, text):
        return self.model(text)


class SummarizationSuite:

    def __init__(self, vocab_size=32000, d_model=1024, n_layer=24, model_type='xlnet', model_path=None, tokenizer=None, batch_size=1, optimizer='adam', learning_rate=1e-5, epochs=1, max_length=512, num_beams=4, length_penalty=2.0, early_stopping=True, device='cpu'):
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.device = device
        self.model_type = model_type
        self.optimizer = optimizer
        self.model = self.build_model_type()
        self.tokenizer = self.build_tokenizer()
        self.texts = []
        self.summaries = []

    def build_model_type(self):
        """
        Acceptable model types for summarization are:
        - xlnet
        - bart
        - t5
        - pegasus
        - gpt2
        - pegasusx
        """
        if self.model_type == 'xlnet':
            self.model = XLNetModel(XLNetConfig(
                vocab_size=self.vocab_size, d_model=self.d_model, n_layer=self.n_layer))
        elif self.model_type == 'bart':
            self.model = BartModel(BartConfig(
                vocab_size=self.vocab_size, d_model=self.d_model, n_layer=self.n_layer))
        elif self.model_type == 't5':
            self.model = T5Model(T5Config(
                vocab_size=self.vocab_size, d_model=self.d_model, n_layer=self.n_layer))
        elif self.model_type == 'pegasus':
            self.model = PegasusModel(PegasusConfig(
                vocab_size=self.vocab_size, d_model=self.d_model, n_layer=self.n_layer))
        elif self.model_type == 'gpt2':
            self.model = GPT2Model(GPT2Config(
                vocab_size=self.vocab_size, n_layer=self.n_layer))
        elif self.model_type == 'pegasusx':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                'google/pegasus-xsum')

    def build_tokenizer(self):
        if self.model_type == 'xlnet':
            self.tokenizer = XLNetTokenizer.from_pretrained(
                'xlnet-base-cased')
        elif self.model_type == 'bart':
            self.tokenizer = BartTokenizer.from_pretrained(
                'facebook/bart-large')
        elif self.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(
                't5-base')
        elif self.model_type == 'pegasus':
            self.tokenizer = PegasusTokenizer.from_pretrained(
                'google/pegasus-large')
        elif self.model_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'gpt2')
        elif self.model_type == 'pegasusx':
            self.tokenizer = AutoTokenizer.from_pretrained(
                'google/pegasus-xsum')

    def build_optimizer(self):
        if self.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.learning_rate)

    def build_loss(self):
        self.loss = nn.CrossEntropyLoss()

    def perform_summarization(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=self.max_length, num_beams=self.num_beams,
                                     length_penalty=self.length_penalty, early_stopping=self.early_stopping)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def load_texts_summaries(self, texts, summaries):
        self.texts = texts
        self.summaries = summaries

    def rouge_score(self, text, summary):
        """
        ROUGE is an automatic evaluation metric for summarization tasks.
        ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation.
        ROUGE evaluates the similarity between a system summary and a set of reference summaries.
        - ROUGE-1: unigram
        - ROUGE-2: bigram
        - ROUGE-L: longest common subsequence

        Performs the ROUGE-N and ROUGE-L evaluation metrics.

        Args:
            text: The reference text.
            summary: The summary text.

        Returns:
            A dictionary containing the ROUGE-N and ROUGE-L scores.
        """
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        score_output = scorer.score(text, summary)

        score_dict = {
            'rouge1': score_output['rouge1'].fmeasure,
            'rouge2': score_output['rouge2'].fmeasure,
            'rougeL': score_output['rougeL'].fmeasure
        }

    def bleu_score(self, text, summary):
        """
        BLEU is an automatic evaluation metric for summarization tasks.
        BLEU stands for Bilingual Evaluation Understudy.
        BLEU evaluates the similarity between a system summary and a set of reference summaries.
        - BLEU-1: unigram
        - BLEU-2: bigram
        - BLEU-3: trigram
        - BLEU-4: 4-gram

        Performs the BLEU-N evaluation metric.

        Args:
            text: The reference text.
            summary: The summary text.

        Returns:
            A dictionary containing the BLEU-N score.
        """

        bleu1 = sentence_bleu(
            [text], summary, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
        bleu2 = sentence_bleu(
            [text], summary, weights=(0, 1, 0, 0), smoothing_function=SmoothingFunction().method1)
        bleu3 = sentence_bleu(
            [text], summary, weights=(0, 0, 1, 0), smoothing_function=SmoothingFunction().method1)
        bleu4 = sentence_bleu(
            [text], summary, weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1)

        return {'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4}

    def jaccard_score(self, text, summary):
        """
        Jaccard is an automatic evaluation metric for summarization tasks.
        Jaccard evaluates the similarity between a system summary and a set of reference summaries.
        - Jaccard: Jaccard similarity

        Performs the Jaccard evaluation metric.

        Args:
            text: The reference text.
            summary: The summary text.

        Returns:
            A dictionary containing the Jaccard score.
        """
        text = set(text.split())
        summary = set(summary.split())

        return {'jaccard': len(text.intersection(summary)) / len(text.union(summary))}

    def perplexity_score(self, text, summary):
        """
        Perplexity is an automatic evaluation metric for summarization tasks.
        Perplexity evaluates the similarity between a system summary and a set of reference summaries.
        - Perplexity: Perplexity

        Performs the Perplexity evaluation metric.

        Args:
            text: The reference text.
            summary: The summary text.

        Returns:
            A dictionary containing the Perplexity score.
        """

        perplexity = np.exp(
            len(text.split()) / len(summary.split())) if len(summary.split()) > 0 else 0

        return {'perplexity': perplexity}

    def cosine_score(self, text, summary):
        """
        Cosine is an automatic evaluation metric for summarization tasks.
        Cosine evaluates the similarity between a system summary and a set of reference summaries.
        - Cosine: Cosine similarity

        Performs the Cosine evaluation metric.

        Args:
            text: The reference text.
            summary: The summary text.

        Returns:
            A dictionary containing the Cosine score.
        """

        text = text.split()
        summary = summary.split()

        vectorizer = CountVectorizer().fit(text)
        vectorizer.transform(text)
        vectorizer.transform(summary)

        vectorizer = TfidfVectorizer(
            vocabulary=vectorizer.vocabulary_).fit(text)
        text_vector = vectorizer.transform(text)
        summary_vector = vectorizer.transform(summary)

        cosine = cosine_similarity(text_vector, summary_vector).mean() if len(
            summary) > 0 else 0

        return {'cosine': cosine_similarity}

    def full_score_report(self):

        self.rouge_scores = []
        self.bleu_scores = []
        self.jaccard_scores = []
        self.perplexity_scores = []
        self.cosine_scores = []

        for i in range(len(self.texts)):
            text = self.texts[i]
            summary = self.summaries[i]

            rouge_score = self.rouge_score(text, summary)
            bleu_score = self.bleu_score(text, summary)
            jaccard_score = self.jaccard_score(text, summary)
            perplexity_score = self.perplexity_score(text, summary)
            cosine_score = self.cosine_score(text, summary)

            self.rouge_scores.append(rouge_score)
            self.bleu_scores.append(bleu_score)
            self.jaccard_scores.append(jaccard_score)
            self.perplexity_scores.append(perplexity_score)
            self.cosine_scores.append(cosine_score)

        self.rouge_scores = pd.DataFrame(self.rouge_scores)
        self.bleu_scores = pd.DataFrame(self.bleu_scores)
        self.jaccard_scores = pd.DataFrame(self.jaccard_scores)
        self.perplexity_scores = pd.DataFrame(self.perplexity_scores)
        self.cosine_scores = pd.DataFrame(self.cosine_scores)

        self.rouge_scores = self.rouge_scores.mean()
        self.bleu_scores = self.bleu_scores.mean()
        self.jaccard_scores = self.jaccard_scores.mean()
        self.perplexity_scores = self.perplexity_scores.mean()
        self.cosine_scores = self.cosine_scores.mean()

        self.full_scores = pd.concat(
            [self.rouge_scores, self.bleu_scores, self.jaccard_scores, self.perplexity_scores, self.cosine_scores], axis=0)

        return self.full_scores

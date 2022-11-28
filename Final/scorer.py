from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances, euclidean_distances, manhattan_distances, cosine_distances, pairwise_kernels
import numpy as np
import pandas as pd


class ScoringSuite:

    def __init__(self, text, summary):
        self.text = text
        self.summary = summary

    def rouge_score(self, text: str, summary):
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
            'rouge1 fmeasure': score_output['rouge1'].fmeasure,
            'rouge1 precision': score_output['rouge1'].precision,
            'rouge1 recall': score_output['rouge1'].recall,
            'rouge2 fmeasure': score_output['rouge2'].fmeasure,
            'rouge2 precision': score_output['rouge2'].precision,
            'rouge2 recall': score_output['rouge2'].recall,
            'rougeL fmeasure': score_output['rougeL'].fmeasure,
            'rougeL precision': score_output['rougeL'].precision,
            'rougeL recall': score_output['rougeL'].recall
        }
        return score_dict

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

        return {'cosine': cosine}

    def full_score_report(self):

        rouge_score = self.rouge_score(self.text, self.summary)
        bleu_score = self.bleu_score(self.text, self.summary)
        jaccard_score = self.jaccard_score(self.text, self.summary)
        perplexity_score = self.perplexity_score(self.text, self.summary)
        cosine_score = self.cosine_score(self.text, self.summary)

        score_dict = {"Rouge Score": rouge_score, "Bleu Score": bleu_score,
                      "Jaccard Score": jaccard_score, "Perplexity Score": perplexity_score, "Cosine Score": cosine_score}

        return score_dict

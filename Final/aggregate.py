from analyzer import SummarizationSuite
from scorer import ScoringSuite
from utils import grab_random_xsum, grab_random_cnndm, grab_random_gigaword,\
    grab_random_reddit, grab_random_s2orc, grab_random_multinews
import pandas as pd


def generate_dataframes():
    """
    Generates dataframes to store the summarizations from each model.
    For each model, a score dataframe is generated with these columns:
        - input text
        - summary
        - rouge1 fmeasure
        - rouge1 precision
        - rouge1 recall
        - rouge2 fmeasure
        - rouge2 precision
        - rouge2 recall
        - rougeL fmeasure
        - rougeL precision
        - rougeL recall
        - bleu1
        - bleu2
        - bleu3
        - bleu4
        - jaccard
        - perplexity
        - cosine similarity
    """

    MODEL_TYPES = ['bart', 'bartx', 't5', 'pegasus', 'pegasusx']
    DATASETS = ['XSum', 'CNN/DM', 'Gigaword', 'Reddit', 'S2ORC', 'Multi-News']

    for model_type in MODEL_TYPES:
        # Create a dataframe to store the scores and summarizations from each model.
        df = pd.DataFrame(columns=['input_text', 'summary', 'rouge1_fmeasure', 'rouge1_precision', 'rouge1_recall', 'rouge2_fmeasure', 'rouge2_precision',
                          'rouge2_recall', 'rougeL_fmeasure', 'rougeL_precision', 'rougeL_recall', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'jaccard', 'perplexity', 'cosine_similarity'])

        summarizer = SummarizationSuite(model_type)
        summarizer.build_text_records()
        summarizer.build_model()
        summarizer.build_tokenizer()

        for dataset in DATASETS:
            for i in range(50):
                if dataset == "XSum":
                    text = grab_random_xsum()[0]
                elif dataset == "CNN/DM":
                    text = grab_random_cnndm()[0]
                elif dataset == "Gigaword":
                    text = grab_random_gigaword()[0]
                elif dataset == "Reddit":
                    text = grab_random_reddit()[0]
                elif dataset == "S2ORC":
                    text = grab_random_s2orc()[0]
                elif dataset == "Multi-News":
                    text = grab_random_multinews()[0]

                summary = summarizer.summarization(text)

                scoring_suite = ScoringSuite(text, summary)
                full_score_dictionary = scoring_suite.full_score_report()

                # scores should be parsed from the dictionary and added to the dataframe
                rouge1_fmeasure = full_score_dictionary['Rouge Score']['rouge1 fmeasure']
                rouge1_precision = full_score_dictionary['Rouge Score']['rouge1 precision']
                rouge1_recall = full_score_dictionary['Rouge Score']['rouge1 recall']
                rouge2_fmeasure = full_score_dictionary['Rouge Score']['rouge2 fmeasure']
                rouge2_precision = full_score_dictionary['Rouge Score']['rouge2 precision']
                rouge2_recall = full_score_dictionary['Rouge Score']['rouge2 recall']
                rougeL_fmeasure = full_score_dictionary['Rouge Score']['rougeL fmeasure']
                rougeL_precision = full_score_dictionary['Rouge Score']['rougeL precision']
                rougeL_recall = full_score_dictionary['Rouge Score']['rougeL recall']
                bleu1 = full_score_dictionary['Bleu Score']['bleu1']
                bleu2 = full_score_dictionary['Bleu Score']['bleu2']
                bleu3 = full_score_dictionary['Bleu Score']['bleu3']
                bleu4 = full_score_dictionary['Bleu Score']['bleu4']
                jaccard = full_score_dictionary['Jaccard Score']['jaccard']
                perplexity = full_score_dictionary['Perplexity Score']['perplexity']
                cosine_similarity = full_score_dictionary['Cosine Score']['cosine']

                # add all values to the dataframe in their respective columns
                df = df.append({'input_text': text, 'summary': summary, 'rouge1_fmeasure': rouge1_fmeasure, 'rouge1_precision': rouge1_precision, 'rouge1_recall': rouge1_recall, 'rouge2_fmeasure': rouge2_fmeasure, 'rouge2_precision': rouge2_precision, 'rouge2_recall': rouge2_recall,
                               'rougeL_fmeasure': rougeL_fmeasure, 'rougeL_precision': rougeL_precision, 'rougeL_recall': rougeL_recall, 'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4, 'jaccard': jaccard, 'perplexity': perplexity, 'cosine_similarity': cosine_similarity}, ignore_index=True)

                print(
                    f"Adding row {i} to dataframe for {model_type} on {dataset} dataset.")

        df.to_csv(f'./results/{model_type}_scores.csv', index=False)


def test():
    generate_dataframes()


if __name__ == "__main__":
    test()

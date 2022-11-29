from analyzer import SummarizationSuite
from scorer import ScoringSuite
from utils import grab_random_xsum, grab_random_cnndm, grab_random_gigaword,\
    grab_random_reddit, grab_random_multinews
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


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
    DATASETS = ['XSum', 'CNNDM', 'Gigaword', 'Reddit', 'Multi-News']
    base_df = pd.DataFrame(columns=['dataset', 'rouge1_fmeasure', 'rouge1_precision', 'rouge1_recall', 'rouge2_fmeasure', 'rouge2_precision',
                                    'rouge2_recall', 'rougeL_fmeasure', 'rougeL_precision', 'rougeL_recall', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'jaccard', 'perplexity', 'cosine_similarity'])

    for model_type in MODEL_TYPES:
        # Create a dataframe to store the scores and summarizations from each model.
        score_df = base_df.copy()

        summarizer = SummarizationSuite(model_type)
        summarizer.build_text_records()
        summarizer.build_model()
        summarizer.build_tokenizer()

        for dataset in DATASETS:
            for i in range(60):
                if dataset == "XSum":
                    text = grab_random_xsum()[0]
                elif dataset == "CNNDM":
                    text = grab_random_cnndm()[0]
                elif dataset == "Gigaword":
                    text = grab_random_gigaword()[0]
                elif dataset == "Reddit":
                    text = grab_random_reddit()[0]
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

                # create the row to add to the dataframe
                row = {'dataset': dataset, 'rouge1_fmeasure': rouge1_fmeasure, 'rouge1_precision': rouge1_precision, 'rouge1_recall': rouge1_recall, 'rouge2_fmeasure': rouge2_fmeasure, 'rouge2_precision': rouge2_precision,
                       'rouge2_recall': rouge2_recall, 'rougeL_fmeasure': rougeL_fmeasure, 'rougeL_precision': rougeL_precision, 'rougeL_recall': rougeL_recall, 'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4, 'jaccard': jaccard, 'perplexity': perplexity, 'cosine_similarity': cosine_similarity}

                # add the row to the dataframe
                score_df = pd.concat([score_df, pd.DataFrame([row])])

                print(
                    f"Adding row {i} to dataframe for {model_type} on {dataset} dataset.")

        score_df.to_csv(f'./results/{model_type}_scores.csv', index=False)
        print(f"Saved to {model_type}_scores.csv in the results folder.")


MODEL_TYPES = ['bart', 'bartx', 't5', 'pegasus', 'pegasusx']
DATASETS = ['XSum', 'CNNDM', 'Gigaword', 'Reddit', 'Multi-News']


def grab_model_csvs():
    """
    Grabs all the csv files from the results/model_results directory.
    """

    csv_files = os.listdir('results/model_results')
    csv_files = [file for file in csv_files if file.endswith('.csv')]
    return csv_files


def grab_dataset_csvs():
    """
    Grabs all the csv files from the results/dataset_results directory.
    """

    csv_files = os.listdir('results/dataset_results')
    csv_files = [file for file in csv_files if file.endswith('.csv')]
    return csv_files


def create_dataframe_by_dataset(dataset):
    """
    Creates a dataframe of model scores for a given dataset.
    """

    # grab csvs
    csv_files = grab_model_csvs()

    df = pd.DataFrame(columns=['model', 'rouge1_fmeasure', 'rouge1_precision', 'rouge1_recall', 'rouge2_fmeasure', 'rouge2_precision',
                               'rouge2_recall', 'rougeL_fmeasure', 'rougeL_precision', 'rougeL_recall', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'jaccard', 'perplexity', 'cosine_similarity'])

    for csv_file in csv_files:
        print(f"Opening {csv_file}")
        model = csv_file.split('_')[0]
        model_df = pd.read_csv('results/model_results/' + csv_file)
        model_df = model_df[model_df['dataset'] == dataset]
        model_df['model'] = model
        model_df = model_df.drop(columns=['dataset'])
        df = pd.concat([df, model_df])

    # save dataframe
    df.to_csv('results/dataset_results/' +
              dataset + '_scores.csv', index=False)


def plot_rouge_scores(dataset_result_path):
    """
    Creates a plot with three subplots, one for each rouge score.
    - dataset_result_path: path to the csv file containing the dataset results
    """

    # read in csv
    df = pd.read_csv(dataset_result_path)

    # create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # plot rouge1
    sns.barplot(x='model', y='rouge1_fmeasure', data=df, ax=axes[0])
    axes[0].set_title('ROUGE-1 F-Measure')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('ROUGE-1 F-Measure')

    # plot rouge2
    sns.barplot(x='model', y='rouge2_fmeasure', data=df, ax=axes[1])
    axes[1].set_title('ROUGE-2 F-Measure')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('ROUGE-2 F-Measure')

    # plot rougeL
    sns.barplot(x='model', y='rougeL_fmeasure', data=df, ax=axes[2])
    axes[2].set_title('ROUGE-L F-Measure')
    axes[2].set_xlabel('Model')
    axes[2].set_ylabel('ROUGE-L F-Measure')

    # save figure
    plt.savefig(dataset_result_path.split('.')[0] + '_rouge_scores.png')
    plt.close()


def test():
    generate_dataframes()


if __name__ == "__main__":
    test()

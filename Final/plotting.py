import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys


def clean_columns(file_path):

    df = pd.read_csv(file_path)
    df = df.drop(columns=['input_text', 'summary'])
    df = df.rename(columns={'rouge1_fmeasure': 'ROUGE-1 F1', 'rouge1_precision': 'ROUGE-1 Precision', 'rouge1_recall': 'ROUGE-1 Recall', 'rouge2_fmeasure': 'ROUGE-2 F1', 'rouge2_precision': 'ROUGE-2 Precision', 'rouge2_recall': 'ROUGE-2 Recall', 'rougeL_fmeasure': 'ROUGE-L F1',
                   'rougeL_precision': 'ROUGE-L Precision', 'rougeL_recall': 'ROUGE-L Recall', 'bleu1': 'BLEU-1', 'bleu2': 'BLEU-2', 'bleu3': 'BLEU-3', 'bleu4': 'BLEU-4', 'jaccard': 'Jaccard', 'perplexity': 'Perplexity', 'cosine_similarity': 'Cosine Similarity'})
    return df


# Heatmap of all models' scores on all datasets
def heatmap_plots(file_path):
    """
    Generates a heatmap of the scores from the summarizations.
    - file_path: the path to the csv file containing the scores.
    - output files are stored in the `figures` directory with a name that
      indicates the model and plot type.
    """

    df = clean_columns(file_path)
    # drop the dataset column
    df = df.drop(columns=['dataset'])
    # Generate a heatmap of the scores.
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns,
                yticklabels=corr.columns, annot=True)
    plt.title('Correlation Heatmap')
    model_type = file_path.split("/")[-1].split("_")[0]
    plt.savefig(f'figures/{model_type}_heatmap.png')
    plt.clf()


# Plot to show how well a model does on each dataset (average score)
def dataset_model_relation_plot(file_path):
    """
    Generates a plot of the average score of each model on each dataset.
    - file_path: the path to the csv file containing the scores.
    - output files are stored in the `figures` directory with a name that
      indicates the model and plot type.
    """

    df = clean_columns(file_path)
    # Plot the average score of each model on each dataset.
    df = df.groupby('dataset').mean()
    df = df.transpose()

    # use the seaborn barplot to plot the data
    sns.barplot(data=df)
    model_type = file_path.split("/")[-1].split("_")[0]
    plt.title(f"Average Score of {model_type} on Each Dataset")
    plt.savefig(f'figures/{model_type}_dataset_model_relation.png')
    plt.clf()


# Clustermap of all models' scores on all datasets
def clustermap_plots(file_path):
    """
    Generates a clustermap of the scores from the summarizations.
    - file_path: the path to the csv file containing the scores.
    - output files are stored in the `figures` directory with a name that
      indicates the model and plot type.
    """

    df = clean_columns(file_path)
    # Generate a clustermap of the scores.
    sns.clustermap(df.corr(), annot=True)
    plt.title('Correlation Clustermap')
    model_type = file_path.split("/")[-1].split("_")[0]
    plt.savefig(f'figures/{model_type}_clustermap.png')
    plt.clf()


def single_plot_set(file_path):
    heatmap_plots(file_path)
    dataset_model_relation_plot(file_path)
    clustermap_plots(file_path)


def generate_all_plots():

    # Grab the list of all csv files in the `results` directory.
    files = os.listdir('results')
    csv_files = [file for file in files if file.endswith('.csv')]
    for file in csv_files:
        file_path = f'results/{file}'
        single_plot_set(file_path)


def main():
    # use sys command line arguments to determine which plots to generate
    if len(sys.argv) == 1:
        generate_all_plots()
    elif len(sys.argv) == 2:
        file_path = sys.argv[1]
        single_plot_set(file_path)


if __name__ == '__main__':
    main()

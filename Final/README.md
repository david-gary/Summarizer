# Text Analyzer: Deliverable 1

## Project Description

The first stage of this project provides simple sentiment analysis of given texts using the `transformers` library. A variety of dataset loading scripts have been implemented, which can be used to supply sample text input for the analyzer to score. A frontend has been implemented to allow users to input text of their own, or to select details for the sample input they wish to use.

## Installation Requirements and Setup

This project requires Bash and Python 3.7.3 or higher, and many of the scripts require the use of the `python3` command. To install Python 3.7.3, follow the instructions on the [Python website](https://www.python.org/downloads/). The project also requires the use of the `pip` command to install the required packages. Run the command below to get your folder setup completed, grant the proper permissions to all scripts, and install the required packages.

```bash
bash init.sh
```

## Loading the Datasets

As additional training data for XLNet, three datasets have been provided and can be easily loaded using the `datasets-loader.py` script. The script takes in a command line argument for the dataset name and writes all relevant files to their respective directories.

Dataset arguments:

- `reddit`: Reddit TIFU dataset
- `multinews`: Multi-News dataset
- `s2orc`: S2ORC dataset **WARNING:** this dataset is very large (516GiB) and will take a long time to download
- `cnndm`: CNN/Daily Mail dataset
- `xsum`: XSum dataset
- `gigaword`: Gigaword dataset

```bash
python3 datasets_loader.py --dataset <dataset_name>
```

### Example

```bash
python3 datasets_loader.py --dataset cnndm
```

## Erasing all Datasets

If you would like to clear out all dataset subdirectories, run the following command:

```bash
./reset_datasets.sh
```

## Text Analysis Interface

The entire program is packed into a single interface for the user. Once the datasets have been loaded, the user can simply run the command below to start the interface.

```bash
streamlit run main_UI.py
```

This will display an address in the terminal which can be followed to the frontend display in the browser. The left side of the screen contains a set of customizations the user can make to change the source dataset, model used for summarization, output summary length range, and a description of the model used. The main area of the screen contains two buttons, one for sentiment analysis and one for summary generation.

The sentiment analysis button will run the `bert-base-uncased` model on the input text and display the sentiment score. The summary generation button will run whichever model has been selected, but the score report shown at the end will stay the same. This score report contains a series of metrics that are considered the standard for summarization evaluation. The metrics are:

- ROUGE-1 (Precision, Recall, F1)
- ROUGE-2 (Precision, Recall, F1)
- ROUGE-L (Precision, Recall, F1)
- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4
- Jaccard Similarity
- Perplexity
- Cosine Similarity

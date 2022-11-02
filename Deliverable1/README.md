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
- `multi-news`: Multi-News dataset
- `s2orc`: S2ORC dataset **WARNING:** this dataset is very large (516GiB) and will take a long time to download
- `all`: Downloads all three datasets

```bash
python3 datasets_loader.py --dataset <dataset_name>
```

## Erasing all Datasets

If you would like to clear out all dataset subdirectories, run the following command:

```bash
./reset_datasets.sh
```

## Sentiment Analysis Interface

## Sentiment Analysis CLI

## Unit Tests

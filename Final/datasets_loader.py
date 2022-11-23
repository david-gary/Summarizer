from datasets import load_dataset
from tqdm import tqdm
import argparse


def load_multi_news():
    dataset = load_dataset('multi_news')

    # for each document in dataset['train'], write it to a file
    print(
        f"Loading {len(dataset['train'])} files from the Multi-News train set")
    for i in tqdm(range(len(dataset['train']))):
        with open(f"datasets/multinews/training/trainDocument{i}.txt", "w") as f:
            f.write(dataset['train'][i]['document'])
        with open(f"datasets/multinews/training/trainSummary{i}.txt", "w") as f:
            f.write(dataset['train'][i]['summary'])

    print(
        f"Loading {len(dataset['validation'])} files from the Multi-News validation set")
    for i in tqdm(range(len(dataset['validation']))):
        with open(f"datasets/multinews/validation/valDocument{i}.txt", "w") as f:
            f.write(dataset['train'][i]['document'])
        with open(f"datasets/multinews/validation/valSummary{i}.txt", "w") as f:
            f.write(dataset['train'][i]['summary'])

    print(f"Loading {len(dataset['test'])} files from the Multi-News test set")
    for i in tqdm(range(len(dataset['test']))):
        with open(f"datasets/multinews/testing/testDocument{i}.txt", "w") as f:
            f.write(dataset['train'][i]['document'])
        with open(f"datasets/multinews/testing/testSummary{i}.txt", "w") as f:
            f.write(dataset['train'][i]['summary'])


def load_reddit_tifu():
    dataset = load_dataset('reddit_tifu', 'short')

    # for the reddit data, we need only the `documents` and `tldr` fields
    print(
        f"Loading {len(dataset['train'])} files from the Reddit TIFU dataset")
    for i in tqdm(range(len(dataset['train']))):
        # check that the tldr value is not an empty string
        if dataset['train'][i]['tldr'] != '':
            with open(f"datasets/reddit/documents/trainDocument{i}.txt", "w") as f:
                f.write(dataset['train'][i]['documents'])
            with open(f"datasets/reddit/summaries/trainSummary{i}.txt", "w") as f:
                f.write(dataset['train'][i]['tldr'])


def load_s2orc():

    dataset = load_dataset('s2orc')

    # for each document in dataset['train'], write it to a file
    print(f"Loading {len(dataset['train'])} files from the train set")
    for i in tqdm(range(len(dataset['train']))):
        with open(f"datasets/s2orc/training/traindoc{i}.txt", "w") as f:
            f.write(dataset['train'][i]['text'])

    print(
        f"Loading {len(dataset['validation'])} files from the validation set")
    for i in tqdm(range(len(dataset['validation']))):
        with open(f"datasets/s2orc/validation/valdoc{i}.txt", "w") as f:
            f.write(dataset['validation'][i]['text'])

    print(f"Loading {len(dataset['test'])} files from the test set")
    for i in tqdm(range(len(dataset['test']))):
        with open(f"datasets/s2orc/testing/testdoc{i}.txt", "w") as f:
            f.write(dataset['test'][i]['text'])


def load_xsum():
    dataset = load_dataset('xsum')

    # for each document in dataset['train'], write it to a file
    print(f"Loading {len(dataset['train'])} files from the train set")
    for i in tqdm(range(len(dataset['train']))):
        with open(f"datasets/xsum/training/traindoc{i}.txt", "w") as f:
            f.write(dataset['train'][i]['document'])

    print(
        f"Loading {len(dataset['validation'])} files from the validation set")
    for i in tqdm(range(len(dataset['validation']))):
        with open(f"datasets/xsum/validation/valdoc{i}.txt", "w") as f:
            f.write(dataset['validation'][i]['document'])

    print(f"Loading {len(dataset['test'])} files from the test set")
    for i in tqdm(range(len(dataset['test']))):
        with open(f"datasets/xsum/testing/testdoc{i}.txt", "w") as f:
            f.write(dataset['test'][i]['document'])


def load_cnn_dailymail():
    dataset = load_dataset('cnn_dailymail', '3.0.0')

    # for each document in dataset['train'], write it to a file
    print(f"Loading {len(dataset['train'])} files from the train set")
    for i in tqdm(range(len(dataset['train']))):
        with open(f"datasets/cnn_dailymail/training/traindoc{i}.txt", "w") as f:
            f.write(dataset['train'][i]['article'])

    print(
        f"Loading {len(dataset['validation'])} files from the validation set")
    for i in tqdm(range(len(dataset['validation']))):
        with open(f"datasets/cnn_dailymail/validation/valdoc{i}.txt", "w") as f:
            f.write(dataset['validation'][i]['article'])

    print(f"Loading {len(dataset['test'])} files from the test set")
    for i in tqdm(range(len(dataset['test']))):
        with open(f"datasets/cnn_dailymail/testing/testdoc{i}.txt", "w") as f:
            f.write(dataset['test'][i]['article'])


def load_gigaword():
    dataset = load_dataset('gigaword')

    # for each document in dataset['train'], write it to a file
    print(f"Loading {len(dataset['train'])} files from the train set")
    for i in tqdm(range(len(dataset['train']))):
        with open(f"datasets/gigaword/training/traindoc{i}.txt", "w") as f:
            f.write(dataset['train'][i]['text'])

    print(
        f"Loading {len(dataset['validation'])} files from the validation set")
    for i in tqdm(range(len(dataset['validation']))):
        with open(f"datasets/gigaword/validation/valdoc{i}.txt", "w") as f:
            f.write(dataset['validation'][i]['text'])

    print(f"Loading {len(dataset['test'])} files from the test set")
    for i in tqdm(range(len(dataset['test']))):
        with open(f"datasets/gigaword/testing/testdoc{i}.txt", "w") as f:
            f.write(dataset['test'][i]['text'])


def main():
    # get keyword arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='multinews')
    args = parser.parse_args()

    if args.dataset == 'multinews':
        load_multi_news()
    elif args.dataset == 'reddit':
        load_reddit_tifu()
    elif args.dataset == 's2orc':
        load_s2orc()
    elif args.dataset == 'xsum':
        load_xsum()
    elif args.dataset == 'cnn_dailymail':
        load_cnn_dailymail()
    elif args.dataset == 'gigaword':
        load_gigaword()

    elif args.dataset == 'all':
        load_multi_news()
        load_reddit_tifu()
        load_s2orc()
    else:
        print("Invalid dataset name. Please choose from multinews, reddit, s2orc, or all.\ne.g. python datasets_loader.py --dataset multinews")


if __name__ == '__main__':
    main()

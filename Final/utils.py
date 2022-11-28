import os
import random
import sys


# grab random news from testing folder
def grab_random_multinews(n=1):

    # join path to testing folder (cwd + datasets/multinews + testing)
    path = os.path.join(os.getcwd(), "datasets", "multinews", "testing")
    if len(os.listdir(path)) == 0:
        print("Multi-News testing folder is empty. Please run 'datasets_loader.py --dataset multinews to load the dataset.")
        sys.exit()
    else:
        # grab n random files from testing folder
        files = random.sample(os.listdir(path), n)

        # open each file and save the text to a list
        texts = []
        for file in files:
            with open(os.path.join(path, file), "r") as f:
                texts.append(f.read())

    return texts


def grab_random_reddit(n=1):

    # join path to testing folder (cwd + datasets/reddit + documents)
    path = os.path.join(os.getcwd(), "datasets", "reddit", "documents")
    if len(os.listdir(path)) == 0:
        print("Reddit documents folder is empty. Please run 'datasets_loader.py --dataset reddit' to load the dataset.")
        sys.exit()
    else:
        # grab n random files from testing folder
        files = random.sample(os.listdir(path), n)

        # open each file and save the text to a list
        texts = []
        for file in files:
            with open(os.path.join(path, file), "r") as f:
                texts.append(f.read())

    return texts


def grab_random_s2orc(n=1):

    # join path to testing folder (cwd + datasets/s2orc + abstracts)
    path = os.path.join(os.getcwd(), "datasets", "s2orc", "abstracts")
    if len(os.listdir(path)) == 0:
        print("S2ORC abstracts folder is empty. Please run 'datasets_loader.py --dataset s2orc' to load the dataset.")
        sys.exit()
    else:
        # grab n random files from testing folder
        files = random.sample(os.listdir(path), n)

        # open each file and save the text to a list
        texts = []
        for file in files:
            with open(os.path.join(path, file), "r") as f:
                texts.append(f.read())

    return texts


def grab_random_xsum(n=1):

    # join path to testing folder (cwd + datasets/xsum + testing)
    path = os.path.join(os.getcwd(), "datasets", "xsum", "testing")
    if len(os.listdir(path)) == 0:
        print("XSum documents folder is empty. Please run 'datasets_loader.py --dataset xsum' to load the dataset.")
        sys.exit()
    else:
        # grab n random files from testing folder
        files = random.sample(os.listdir(path), n)

        # open each file and save the text to a list
        texts = []
        for file in files:
            with open(os.path.join(path, file), "r") as f:
                texts.append(f.read())

    return texts


def grab_random_cnndm(n=1):

    # join path to testing folder (cwd + datasets/cnndm + testing)
    path = os.path.join(os.getcwd(), "datasets", "cnndm", "testing")
    if len(os.listdir(path)) == 0:
        print("CNN/DailyMail documents folder is empty. Please run 'datasets_loader.py --dataset cnn_dailymail' to load the dataset.")
        sys.exit()
    else:
        # grab n random files from testing folder
        files = random.sample(os.listdir(path), n)

        # open each file and save the text to a list
        texts = []
        for file in files:
            with open(os.path.join(path, file), "r") as f:
                texts.append(f.read())

    return texts


def grab_random_gigaword(n=1):

    # join path to testing folder (cwd + datasets + gigaword)
    path = os.path.join(os.getcwd(), "datasets", "gigaword", "testing")
    if len(os.listdir(path)) == 0:
        print("Gigaword folder is empty. Please run 'datasets_loader.py --dataset gigaword' to load the dataset.")
        sys.exit()
    else:
        # grab n random files from testing folder
        files = random.sample(os.listdir(path), n)

        # open each file and save the text to a list
        texts = []
        for file in files:
            with open(os.path.join(path, file), "r") as f:
                texts.append(f.read())

    return texts

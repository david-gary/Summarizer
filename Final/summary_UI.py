from utils import grab_random_xsum, grab_random_cnndm, grab_random_gigaword, grab_random_reddit, grab_random_s2orc
from analyzer import SummarizationSuite
import streamlit as st


def main():
    st.title("Summarization Tool")

    # dropdown side menu for selecting dataset or custom text
    st.sidebar.title("Select Dataset")
    # default value is "Custom Text"

    # datasets are limited to ones that have summaries provided
    # - XSum
    # - CNN/DM
    # - Gigaword
    # - Reddit
    # - S2ORC

    dataset = st.sidebar.selectbox(
        "Select Dataset", ["Custom Text", "XSum", "CNN/DM", "Gigaword", "Reddit", "S2ORC"])

    default_text = "The Russell/Norvig Book is amazing."

    if dataset == "Custom Text":
        text = st.text_area("Enter text to summarize:", default_text)
    elif dataset == "XSum":
        text = grab_random_xsum()[0]
    elif dataset == "CNN/DM":
        text = grab_random_cnndm()[0]
    elif dataset == "Gigaword":
        text = grab_random_gigaword()[0]
    elif dataset == "Reddit":
        text = grab_random_reddit()[0]
    elif dataset == "S2ORC":
        text = grab_random_s2orc()[0]

    # second dropdown side menu for selecting summarization model
    st.sidebar.title("Select Summarization Model")
    # default value is "XLNet"

    # Acceptable model types for summarization are:
    # - xlnet
    # - bart
    # - t5
    # - pegasus
    # - gpt2
    # - pegasusx
    model = st.sidebar.selectbox(
        "Select Summarization Model", ["XLNet", "BART", "T5", "Pegasus", "GPT2", "PegasusX"])

    if model == "XLNet":
        model_type = "xlnet"
    elif model == "BART":
        model_type = "bart"
    elif model == "T5":
        model_type = "t5"
    elif model == "Pegasus":
        model_type = "pegasus"
    elif model == "GPT2":
        model_type = "gpt2"
    elif model == "PegasusX":
        model_type = "pegasusx"

    # button to perform summarization and score results

    if st.button("Summarize"):
        st.write("Summarizing...")
        summarizer = SummarizationSuite(model_type)
        summary = summarizer.summarize(text)
        st.write(summary)
        # full score report
        st.write("Score Report")
        st.write(summarizer.full_score_report(text, summary))


if __name__ == "__main__":

    main()

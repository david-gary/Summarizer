from utils import grab_random_xsum, grab_random_cnndm, grab_random_gigaword,\
    grab_random_reddit, grab_random_s2orc, grab_random_multinews
from analyzer import SummarizationSuite
from scorer import ScoringSuite
import streamlit as st


def main():
    st.title("Summarization Tool")

    # dropdown side menu for selecting dataset or custom text
    st.sidebar.title("Select Dataset")

    dataset = st.sidebar.selectbox(
        "Select Dataset", ["CNN/DM", "Custom Text", "XSum", "Gigaword", "Reddit", "S2ORC", "Multi-News"])

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
    elif dataset == "Multi-News":
        text = grab_random_multinews()[0]

    # second dropdown side menu for selecting summarization model
    st.sidebar.title("Select Summarization Model")
    # default value is "XLNet"

    # Acceptable model types for summarization are:
    # - xlnet
    # - bart
    # - t5
    # - pegasus
    # - pegasusx
    model = st.sidebar.selectbox(
        "Model Settings", ["T5", "XLNet", "BART", "Pegasus", "PegasusX"])

    if model == "XLNet":
        model_type = "xlnet"
    elif model == "BART":
        model_type = "bart"
    elif model == "T5":
        model_type = "t5"
    elif model == "Pegasus":
        model_type = "pegasus"
    elif model == "PegasusX":
        model_type = "pegasusx"

    # third dropdown side menu to allow the user to set min and max length of the summary
    summary_length = st.sidebar.slider(
        "Set Summary Length", 10, 1000, (10, 100))
    min_length = summary_length[0]
    max_length = summary_length[1]

    # button to perform summarization and score results

    if st.button("Summarize"):

        summarizer = SummarizationSuite(model_type, max_length, min_length)
        summarizer.build_text_records()
        summarizer.build_model()
        summarizer.build_tokenizer()

        # sidebar text box for displaying the model description
        st.sidebar.text_area("Model Description", summarizer.description)

        # Add a heading for the input text details
        st.header("Input Text Details")
        # show the first 300 characters of the input text

        snippet = text[:300] if len(text) > 300 else text
        st.write(f"Snippet: {snippet}")

        # Show full text length in number of words
        st.write(f"Length: {len(text.split())} words")

        st.header("Summaries")

        summary = summarizer.summarization(text)
        st.write(f" Summary: {summary}")

        # full score report
        st.header("Score Report")
        scoring_suite = ScoringSuite(text, summary)
        st.write(scoring_suite.full_score_report())


if __name__ == "__main__":

    main()

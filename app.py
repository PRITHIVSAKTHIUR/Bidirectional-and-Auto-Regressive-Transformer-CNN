import nltk
import validators
import streamlit as st
from transformers import AutoTokenizer, pipeline


from extractive_summarizer.model_processors import Summarizer
from utils import (
    clean_text,
    fetch_article_text,
    preprocess_text_for_abstractive_summarization,
    read_text_from_file,
)

from rouge import Rouge

if __name__ == "__main__":

    st.title("Text Summarizer 📝")

    st.markdown("Model by: [BART CNN](https://huggingface.co/prithivMLmods/Ares-Bidirectional-and-Auto-Regressive-Transformer-CNN)")
    st.markdown(
        "Source: [GitHub Repository](https://github.com/PRITHIVSAKTHIUR/Bidirectional-and-Auto-Regressive-Transformer-CNN)"
    )
    summarize_type = st.sidebar.selectbox(
        "Summarization type", options=["Extractive", "Abstractive"]
    )

    st.markdown(
        "BART is designed to handle text generation tasks bidirectionally. This means it can generate text both left-to-right and right-to-left, allowing it to capture more contextual information. BART achieves this bidirectionality through a novel pretraining method that involves corrupting the input text with a technique called noising, which involves randomly masking, shuffling, and deleting tokens. By pretraining on this noisy data, BART learns to generate coherent text by reconstructing the original input from its corrupted version. BART has been shown to achieve state-of-the-art performance on various text generation tasks, including summarization, translation, and text classification."
    )
    st.markdown(
        """- Raw text in text box 
- URL of article/news to be summarized 
- .txt, .pdf, .docx file formats"""
    )
    st.markdown(
        """This app supports two type of summarization:

1. **Extractive Summarization**: The extractive approach involves picking up the most important phrases and lines from the documents. It then combines all the important lines to create the summary. So, in this case, every line and word of the summary actually belongs to the original document which is summarized.
2. **Abstractive Summarization**: The abstractive approach involves rephrasing the complete document while capturing the complete meaning of the document. This type of summarization provides more human-like summary"""
    )
    st.markdown("---")

    nltk.download("punkt")
    abs_tokenizer_name = "facebook/bart-large-cnn"
    abs_model_name = "facebook/bart-large-cnn"
    abs_tokenizer = AutoTokenizer.from_pretrained(abs_tokenizer_name)
    abs_max_length = 90
    abs_min_length = 30
    

    inp_text = st.text_input("Enter text or a url here")
    st.markdown(
        "<h3 style='text-align: center; color: green;'>OR</h3>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload a .txt, .pdf, .docx file for summarization"
    )

    is_url = validators.url(inp_text)
    if is_url:
        
        text, cleaned_txt = fetch_article_text(url=inp_text)
    elif uploaded_file:
        cleaned_txt = read_text_from_file(uploaded_file)
        cleaned_txt = clean_text(cleaned_txt)
    else:
        cleaned_txt = clean_text(inp_text)


    with st.expander("View input text"):
        if is_url:
            st.write(cleaned_txt[0])
        else:
            st.write(cleaned_txt)
    summarize = st.button("Summarize")

    
    if summarize:
        if summarize_type == "Extractive":
            if is_url:
                text_to_summarize = " ".join([txt for txt in cleaned_txt])
            else:
                text_to_summarize = cleaned_txt
        

            with st.spinner(
                text="Creating extractive summary. This might take a few seconds ..."
            ):
                ext_model = Summarizer()
                summarized_text = ext_model(text_to_summarize, num_sentences=5)
                

        elif summarize_type == "Abstractive":
            with st.spinner(
                text="Creating abstractive summary. This might take a few seconds ..."
            ):
                text_to_summarize = cleaned_txt
                abs_summarizer = pipeline(
                    "summarization", model=abs_model_name, tokenizer=abs_tokenizer_name
                )

                if is_url is False:
                    
                    text_to_summarize = preprocess_text_for_abstractive_summarization(
                        tokenizer=abs_tokenizer, text=cleaned_txt
                    )

                tmp_sum = abs_summarizer(
                    text_to_summarize,
                    max_length=abs_max_length,
                    min_length=abs_min_length,
                    do_sample=False,
                )

                summarized_text = " ".join([summ["summary_text"] for summ in tmp_sum])

      
        st.subheader("Summarized text")
        st.info(summarized_text)

        st.subheader("Rogue(Recall-Oriented Understudy for Gisting Evaluation)Scores")
        rouge_sc = Rouge()
        ground_truth = cleaned_txt[0] if is_url else cleaned_txt
        score = rouge_sc.get_scores(summarized_text, ground_truth, avg=True)
        st.code(score)

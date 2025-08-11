import streamlit as st
from transformers import pipeline

# Smaller model choice to reduce RAM usage
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model=MODEL_NAME)

summarizer = get_summarizer()

st.set_page_config(page_title="AI Text Summarizer", layout="centered")
st.title("🧠 AI Text Summarizer")
st.write("Paste a long article/paragraph and get a short summary.")

text_input = st.text_area("Enter text here:", height=300)

def summarize_text(text):
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]

if st.button("Summarize"):
    if not text_input.strip():
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Summarizing... this can take a bit on first run"):
            try:
                summary = summarize_text(text_input)
                st.success("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error during summarization: {e}")
                st.write("If this fails due to memory, ask me for an OpenAI-api alternative or a chunking version.")
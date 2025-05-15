# LLM Text Quality & Toxicity Checker

A simple web application for checking the quality and toxicity of English text.  
The tool uses BERT-based models ([unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert) and [dehatebert-mono-english](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english)), together with a custom keyword list with toxicity weights.

## Features
- Grammar, spelling, and readability scoring
- Toxicity detection using both transformer models and keyword filtering
- Easy-to-use Streamlit web interface
- Fast scoring and export for dataset curation or content moderation

## Quick Start
1. Clone the repo:
    ```
    git clone https://github.com/kullanici_adin/llm-text-quality-toxicity.git
    ```
2. Install requirements:
    ```
    pip install -r requirements.txt
    ```
3. Run the app:
    ```
    streamlit run app.py
    ```

## Models Used
- [unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert)
- [Hate-speech-CNERG/dehatebert-mono-english](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english)

---

*This project helps you filter out harmful or low-quality English text quickly and easily.*

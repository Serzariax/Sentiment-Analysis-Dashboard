# Sentiment Analysis Dashboard

This project is a Streamlit-based dashboard for performing sentiment analysis on text data such as customer reviews, social media posts, or any batch of textual input. It leverages Hugging Face's transformer models for robust sentiment classification and provides interactive visualizations and export options.

## Features

- **Input Methods:**  
  - Direct text entry (batch supported, one per line)
  - File upload (CSV or TXT, one text per row/line)

- **Sentiment Analysis:**  
  - Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` for classification (Positive, Negative, Neutral)
  - Confidence scores for each prediction
  - Keyword extraction via TF-IDF for explanation

- **Visualizations:**  
  - Sentiment distribution pie chart
  - Comparative bar chart of average confidence scores

- **Export Options:**  
  - Download results as CSV, JSON, or PDF

- **Custom Styling:**  
  - Modern color palette and UI enhancements

## Usage

1. **Install dependencies:**
    ```sh
    pip install streamlit pandas numpy transformers scikit-learn matplotlib torch PyPDF2 reportlab
    ```

2. **Run the app:**
    ```sh
    streamlit run app.py
    ```

3. **Interact:**
    - Choose input method
    - Enter or upload text
    - Click "Analyze Sentiment"
    - View results, charts, and download options

## Limitations

- Best performance on English text
- May misclassify sarcasm or slang
- Confidence scores below 0.6 may indicate uncertainty

## File Structure

- [`app.py`](app.py): Main Streamlit application
- [`ReadMe.md`](ReadMe.md): Project documentation

## Credits

- Built with [Streamlit](https://streamlit.io/) and [Hugging Face Transformers](https://huggingface.co/transformers/)
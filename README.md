# TextRank News Summarizer

This project implements a **TextRank-based** text summarization algorithm to generate concise summaries from news articles. It processes input text, cleans and tokenizes sentences, computes similarity scores, and selects the most important sentences using the PageRank algorithm.

## Features
- Reads multiple news articles from a text file
- Cleans and tokenizes text using **NLTK**
- Computes **TF-IDF similarity** between sentences
- Uses **PageRank algorithm** to rank and extract key sentences
- Allows users to specify the number of sentences in the summary

## Installation

### Prerequisites
Make sure you have Python installed. Then, install the required dependencies:

```bash
pip install nltk networkx scikit-learn
```

Download necessary NLTK data:

```python
import nltk
nltk.download('punkt')
```

## Usage

1. **Prepare Input:** Ensure `news_articles.txt` contains a **list of articles** stored as a Python list (e.g., `['article 1 text', 'article 2 text', ...]`).
2. **Run the script:** Execute the Python script:

```bash
python app.py
```

3. **Enter the desired summary length:**
   The program will prompt you to enter the number of sentences per summary.

## Code Overview

### 1. **Text Preprocessing**
- **`split_into_sentences(text)`**: Splits text into sentences using **NLTK**.
- **`preprocess_text(text)`**: Cleans text by removing extra spaces and special characters.

### 2. **Creating Similarity Matrix**
- **`create_similarity_matrix(sentences)`**: Converts sentences into **TF-IDF vectors** and computes cosine similarity.

### 3. **TextRank Summarization**
- **`text_rank_summarize(text, num_sentences)`**:
  - Preprocesses and tokenizes text
  - Creates a similarity matrix
  - Constructs a **graph representation** and applies **PageRank**
  - Extracts top-ranked sentences

### 4. **Reading Input and Generating Summaries**
- Reads `news_articles.txt`, processes articles, and prints summaries.

## Example Output
```
Enter the number of sentences you want in the summary: 2

🔹 **News Summaries** 🔹

📰 **Article 1:** Summary sentence 1. Summary sentence 2.
📰 **Article 2:** Summary sentence 1. Summary sentence 2.
...
```

## Limitations
- Works best for **well-structured text** (e.g., news articles, formal writing).
- May produce **redundant or incomplete summaries** for very short texts.


---



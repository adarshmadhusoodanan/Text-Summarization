

import re
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
import ast


# def split_into_sentences(text):
#     """Split text into sentences"""
#     # Split text based on common sentence terminators
#     text = text.replace('!', '.')
#     text = text.replace('?', '.')
#     # Handle common abbreviations
#     text = text.replace('Dr.', 'Dr')
#     text = text.replace('Mr.', 'Mr')
#     text = text.replace('Mrs.', 'Mrs')
#     text = text.replace('Ms.', 'Ms')
#     text = text.replace('Prof.', 'Prof')
#     text = text.replace('e.g.', 'eg')
#     text = text.replace('i.e.', 'ie')
    
#     # Split into sentences
#     sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
#     return sentences

import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt')

def split_into_sentences(text):
    """Split text into sentences using NLTK."""
    return sent_tokenize(text)



def preprocess_text(text):
    """Clean the text"""
    # Remove special characters and extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()




def create_similarity_matrix(sentences):
    """Create a similarity matrix using TF-IDF"""
    # Create a count matrix
    count_vectorizer = CountVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform(sentences)
    
    # Convert to TF-IDF matrix
    tfidf = TfidfTransformer()
    tfidf_matrix = tfidf.fit_transform(count_matrix)
    
    # Calculate cosine similarity
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    
    return similarity_matrix


def text_rank_summarize(text, num_sentences):
    """Summarize text using TextRank algorithm"""
    try:
        # Preprocess text
        text = preprocess_text(text)
        
        # Split into sentences
        sentences = split_into_sentences(text)
        
        # If there are fewer sentences than requested, return the whole text
        if len(sentences) <= num_sentences:
            return text
        
        # Create similarity matrix
        similarity_matrix = create_similarity_matrix(sentences)
        
        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Sort sentences by score
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        
        # Select top sentences
        selected_sentences = [s for _, s in ranked_sentences[:num_sentences]]
        
        # Reorder sentences based on their original position
        original_order = []
        for sentence in sentences:
            if sentence in selected_sentences:
                original_order.append(sentence)
        
        # Join sentences
        summary = ' '.join(original_order)
        return summary
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"




with open("news_articles.txt", "r", encoding="utf-8") as file:
    content = file.read()

# Convert the file content (string) back into a Python list
news_articles = ast.literal_eval(content)

print("enter the number of sentences you want in the summary")
num_sentences = int(input())

print("\n🔹 **News Summaries** 🔹\n")
for i, article in enumerate(news_articles, 1):
    summary = text_rank_summarize(article, num_sentences)  # Generate a summary with 3 sentences
    print(f"📰 **Article {i}:** {summary}\n")
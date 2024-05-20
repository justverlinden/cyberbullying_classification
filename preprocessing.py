import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from afinn import Afinn
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def lexicon_sentiment_analysis(text):
    afinn = Afinn()
    sentiment_score = afinn.score(text)
    return sentiment_score

data_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\aggression_parsed_dataset.csv"
data = pd.read_csv(data_path)

data['cleaned_text'] = data['Text'].apply(clean_text)
data['filtered_text'] = data['cleaned_text'].apply(remove_stopwords)

data['Lexicon_Sentiment_Score'] = data['cleaned_text'].apply(lexicon_sentiment_analysis)

cleaned_data_file = "cleaned_data.xlsx"
data[['cleaned_text', 'filtered_text', 'Lexicon_Sentiment_Score', 'oh_label']].to_excel(cleaned_data_file, index=False)
print(f"Cleaned data with sentiment analysis scores saved to {cleaned_data_file}")

def tfidf_vectorization(text_data):
    print("Performing TF-IDF vectorization...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    
    word_counts = Counter(word for text in text_data for word in text.split())
    less_than_4 = {word: count for word, count in word_counts.items() if count < 3}
    print("Number of words that appear less than 4 times:", len(less_than_4))
    
    return tfidf_matrix, tfidf_vectorizer

tfidf_matrix, tfidf_vectorizer = tfidf_vectorization(data['filtered_text'])

svd = TruncatedSVD(n_components=100)
tfidf_matrix_svd = svd.fit_transform(tfidf_matrix)

svd_columns = [f'SVD_Component_{i+1}' for i in range(100)]
svd_df = pd.DataFrame(tfidf_matrix_svd, columns=svd_columns)

data = pd.concat([data, svd_df], axis=1)

X = tfidf_matrix_svd
y = data['oh_label']  

output_file = "preprocessed_data.xlsx"
with pd.ExcelWriter(output_file) as writer:
    data.to_excel(writer, sheet_name='Preprocessed Data', index=False)


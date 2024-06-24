import pandas as pd
import string
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import BertTokenizer, BertModel
import torch
import nltk
from afinn import Afinn

nltk.download('stopwords')

data_path = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\aggression_parsed_dataset.csv"
data = pd.read_csv(data_path)

X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['oh_label'], test_size=0.1, random_state=42)

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

train_data = pd.DataFrame({'Text': X_train, 'oh_label': y_train})
train_data['cleaned_text'] = train_data['Text'].apply(clean_text)
train_data['filtered_text'] = train_data['cleaned_text'].apply(remove_stopwords)
train_data['Lexicon_Sentiment_Score'] = train_data['cleaned_text'].apply(lexicon_sentiment_analysis)

test_data = pd.DataFrame({'Text': X_test, 'oh_label': y_test})
test_data['cleaned_text'] = test_data['Text'].apply(clean_text)
test_data['filtered_text'] = test_data['cleaned_text'].apply(remove_stopwords)
test_data['Lexicon_Sentiment_Score'] = test_data['cleaned_text'].apply(lexicon_sentiment_analysis)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['filtered_text'])
X_train_tokenized = tokenizer.texts_to_sequences(train_data['filtered_text'])
X_test_tokenized = tokenizer.texts_to_sequences(test_data['filtered_text'])

max_sequence_length = max([len(seq) for seq in X_train_tokenized])

X_train_padded = pad_sequences(X_train_tokenized, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_tokenized, maxlen=max_sequence_length)

w2v_model = Word2Vec(sentences=train_data['filtered_text'].apply(str.split), vector_size=100, window=5, min_count=1, workers=4)

def get_word2vec_embeddings(text_data, w2v_model, vector_size=100):
    embeddings = []
    for text in text_data:
        tokens = text.split()
        vec = np.mean([w2v_model.wv[token] for token in tokens if token in w2v_model.wv] or [np.zeros(vector_size)], axis=0)
        embeddings.append(vec)
    return np.array(embeddings)

word2vec_train_embeddings = get_word2vec_embeddings(train_data['filtered_text'], w2v_model)
word2vec_test_embeddings = get_word2vec_embeddings(test_data['filtered_text'], w2v_model)

def tfidf_vectorization(text_data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    return tfidf_matrix, tfidf_vectorizer

tfidf_matrix_train, tfidf_vectorizer = tfidf_vectorization(train_data['filtered_text'])
tfidf_matrix_test = tfidf_vectorizer.transform(test_data['filtered_text'])

svd = TruncatedSVD(n_components=100)
tfidf_matrix_svd_train = svd.fit_transform(tfidf_matrix_train)
tfidf_matrix_svd_test = svd.transform(tfidf_matrix_test)

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text_data, tokenizer, model):
    embeddings = []
    for text in text_data:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
    return np.array(embeddings)

bert_train_embeddings = get_bert_embeddings(train_data['filtered_text'], tokenizer_bert, bert_model)
bert_test_embeddings = get_bert_embeddings(test_data['filtered_text'], tokenizer_bert, bert_model)

word2vec_train_data = pd.DataFrame(word2vec_train_embeddings, columns=[f'W2V_{i+1}' for i in range(100)])
word2vec_train_data['oh_label'] = train_data['oh_label']
word2vec_train_data['Lexicon_Sentiment_Score'] = train_data['Lexicon_Sentiment_Score']

word2vec_test_data = pd.DataFrame(word2vec_test_embeddings, columns=[f'W2V_{i+1}' for i in range(100)])
word2vec_test_data['oh_label'] = test_data['oh_label']
word2vec_test_data['Lexicon_Sentiment_Score'] = test_data['Lexicon_Sentiment_Score']

tokenized_train_data = pd.DataFrame(X_train_padded)
tokenized_train_data['oh_label'] = train_data['oh_label'].values

tokenized_test_data = pd.DataFrame(X_test_padded)
tokenized_test_data['oh_label'] = test_data['oh_label'].values

tfidf_train_data = pd.DataFrame(tfidf_matrix_svd_train, columns=[f'SVD_Component_{i+1}' for i in range(100)])
tfidf_train_data['oh_label'] = train_data['oh_label']
tfidf_train_data['Lexicon_Sentiment_Score'] = train_data['Lexicon_Sentiment_Score']

tfidf_test_data = pd.DataFrame(tfidf_matrix_svd_test, columns=[f'SVD_Component_{i+1}' for i in range(100)])
tfidf_test_data['oh_label'] = test_data['oh_label']
tfidf_test_data['Lexicon_Sentiment_Score'] = test_data['Lexicon_Sentiment_Score']

bert_train_data = pd.DataFrame(bert_train_embeddings, columns=[f'BERT_{i+1}' for i in range(768)])
bert_train_data['oh_label'] = train_data['oh_label']
bert_train_data['Lexicon_Sentiment_Score'] = train_data['Lexicon_Sentiment_Score']

bert_test_data = pd.DataFrame(bert_test_embeddings, columns=[f'BERT_{i+1}' for i in range(768)])
bert_test_data['oh_label'] = test_data['oh_label']
bert_test_data['Lexicon_Sentiment_Score'] = test_data['Lexicon_Sentiment_Score']

output_train_file_word2vec = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_word2vec.csv"
output_test_file_word2vec = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_word2vec.csv"
output_train_file_tokenized = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_tokenized.csv"
output_test_file_tokenized = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_tokenized.csv"
output_train_file_tfidf = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_tfidf.csv"
output_test_file_tfidf = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_tfidf.csv"
output_train_file_bert = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_train_data_bert.csv"
output_test_file_bert = r"C:\Users\just-\Documents\Tilburg University\Master thesis\Code2\preprocessed_test_data_bert.csv"

word2vec_train_data.to_csv(output_train_file_word2vec, index=False)
word2vec_test_data.to_csv(output_test_file_word2vec, index=False)

tokenized_train_data.to_csv(output_train_file_tokenized, index=False)
tokenized_test_data.to_csv(output_test_file_tokenized, index=False)

tfidf_train_data.to_csv(output_train_file_tfidf, index=False)
tfidf_test_data.to_csv(output_test_file_tfidf, index=False)

bert_train_data.to_csv(output_train_file_bert, index=False)
bert_test_data.to_csv(output_test_file_bert, index=False)

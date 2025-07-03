🎬 IMDB Sentiment Analysis with Classical ML, Topic Modeling & Deep Learning

This project performs sentiment analysis on the IMDB 50K Movie Reviews dataset using preprocessing, classical ML (Logistic Regression, Random Forest), topic modeling (LDA), and deep learning (LSTM with GloVe).

📂 Dataset Overview

Source: Kaggle IMDB Dataset of 50K Movie Reviews

Features:
  review: Movie review text
  sentiment: Target label (positive/negative)
  Class distribution is balanced.

🔍 Project Objectives

  Clean and preprocess review text
  Perform sentiment classification using:
  Logistic Regression
  Random Forest
  LSTM with GloVe embeddings
  Visualize important words using word clouds and bar plots
  Perform topic modeling with LDA

🧹 Text Preprocessing

  Removed HTML tags and punctuation
  Removed stopwords
  Tokenized and converted to lowercase
  Applied stemming and lemmatization (stemmed version kept)

📈 Classical Machine Learning

  TF-IDF Vectorization
  Extracted top 1000 features using TfidfVectorizer
  Logistic Regression
  Achieved solid performance:
  Accuracy: High
  Precision, recall, F1-score balanced
  Random Forest
  Comparable performance to Logistic Regression
  Used feature_importances_ to extract key terms

📊 Feature Visualization

  Top 10 positive and negative terms visualized using:
  WordClouds
  Bar Plots

📚 Topic Modeling (LDA)

  Used Gensim's LDA model with 2 topics
  Created dictionary + corpus from tokenized reviews
  Identified dominant topic for each review
  Visualized topic distribution across dataset

🧠 Deep Learning with Word Embeddings

Word Embeddings

  Loaded pre-trained GloVe 50d vectors
  Averaged embeddings for each review

Model Architecture

  Embedding layer with GloVe weights
  Bidirectional LSTM
  Dropout and Dense layers
  Output layer with sigmoid activation
  Performance
  Binary cross-entropy loss
  Validation accuracy competitive with classical models

✅ Summary

  Cleaned and processed large-scale text data
  Built and compared several models:
  Logistic Regression, Random Forest, LSTM
  Performed feature interpretation and topic modeling
  Achieved strong sentiment classification with interpretable results

💾 Deliverables

Final Jupyter Notebook

  Trained models
  Preprocessed data
  Visualizations: WordClouds, Confusion Matrices, Topic Distributions

👩‍💻 Author

Fatma BozovaEmail: turannfatma@gmail.comLinkedIn: linkedin.com/in/fatma-bozovaGitHub: github.com/FatmaBozova


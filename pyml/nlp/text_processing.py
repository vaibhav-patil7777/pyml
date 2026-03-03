"""
PyML NLP Text Processing Module
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from ..core.base import BaseModule
from ..logger import logger


class NLPProcessor(BaseModule):

    def __init__(self):
        super().__init__()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        logger.info("NLP Processor Initialized")

    # ---------------- TEXT CLEANING ----------------

    def clean_text(self, text):

        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ---------------- STOPWORDS REMOVAL ----------------

    def remove_stopwords(self, text):

        stop_words = set(stopwords.words("english"))
        words = text.split()

        filtered = [word for word in words if word not in stop_words]

        return " ".join(filtered)

    # ---------------- STEMMING ----------------

    def apply_stemming(self, text):

        words = text.split()
        stemmed = [self.stemmer.stem(word) for word in words]
        return " ".join(stemmed)

    # ---------------- LEMMATIZATION ----------------

    def apply_lemmatization(self, text):

        words = text.split()
        lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)

    # ---------------- VECTORIZATION ----------------

    def bag_of_words(self, corpus):

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)

        logger.info("Bag of Words Applied")
        return X, vectorizer

    def tfidf(self, corpus):

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)

        logger.info("TF-IDF Applied")
        return X, vectorizer
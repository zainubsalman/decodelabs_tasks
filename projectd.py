"""
DecodeLabs Project 4: NLP & Sentiment Analysis
======================================================================

Goal: Program a machine to read and mathematically categorize
unstructured human text (e.g., product reviews) as Positive or
Negative.

This script follows the exact 5-step Production Pipeline from the
brief:
    1. Stop-Words:    Exclude negations ('not') from default NLTK lists
    2. Lemmatization:  Map POS tags to WordNet for accurate word roots
    3. Vectorization:  TF-IDF with Unigrams + Bigrams, max_features capped
    4. Memory:         Store vectors in SciPy CSR sparse format
    5. Inference:      Multinomial/Complement Naive Bayes + Laplace smoothing

Dataset note: no specific dataset file was named in the slides you
shared. Replace the synthetic reviews below with your real dataset
(a CSV with a text column and a label column) once you have it.
"""

import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# STEP 0: LOAD THE DATA
# ============================================================

USE_REAL_DATA = False  # set True once you have your real dataset

def load_data():
    if USE_REAL_DATA:
        df = pd.read_csv("reviews.csv")  # expects columns: "review", "sentiment"
        return df

    # ---- Synthetic fallback (demo only) ----
    reviews = [
        ("This product is TERRIBLE!!! <br> I am not happy with it at all.", "Negative"),
        ("Absolutely wonderful, I love this so much.", "Positive"),
        ("Not good. Broke after one day.", "Negative"),
        ("Great quality and fast shipping, highly recommend.", "Positive"),
        ("I am not satisfied, this was a waste of money.", "Negative"),
        ("Works perfectly, exactly as described.", "Positive"),
        ("Terrible customer service and a bad product.", "Negative"),
        ("Amazing value for the price, very happy.", "Positive"),
        ("Not what I expected, quite disappointing.", "Negative"),
        ("Excellent build quality, I am very pleased.", "Positive"),
        ("This is not good at all, would not buy again.", "Negative"),
        ("Fantastic purchase, works great every time.", "Positive"),
        ("The item arrived broken and support was unhelpful.", "Negative"),
        ("Really good product, does exactly what it says.", "Positive"),
        ("Not impressed, feels cheap and flimsy.", "Negative"),
        ("Superb experience from start to finish.", "Positive"),
        ("I do not recommend this, it stopped working fast.", "Negative"),
        ("Very satisfied, will buy again for sure.", "Positive"),
        ("Poor quality and not worth the price.", "Negative"),
        ("Loved it, exceeded my expectations completely.", "Positive"),
    ] * 5  # repeated to give the vectorizer/classifier enough rows to work with

    df = pd.DataFrame(reviews, columns=["review", "sentiment"])
    return df


# ============================================================
# STEP 1: NEGATION-SAFE STOP-WORD LIST
# ============================================================

def build_negation_safe_stopwords():
    """
    Default NLTK stop words include negations like 'not', 'no', 'nor',
    which destroys sentiment meaning if removed ("I am not happy"
    becomes "I am happy"). We explicitly exclude negation words from
    the stop-word set using set-based difference.
    """
    default_stopwords = set(stopwords.words("english"))

    negation_words = {
        "not", "no", "nor", "never", "none", "nobody", "nothing",
        "neither", "nowhere", "cannot", "can't", "won't", "don't",
        "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
        "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't", "couldn't",
    }

    safe_stopwords = default_stopwords - negation_words
    return safe_stopwords


# ============================================================
# STEP 2: POS-GUIDED LEMMATIZATION
# ============================================================

def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Map NLTK's Treebank POS tags (e.g. 'VBD', 'JJ') to WordNet's
    simpler tag set. Without this mapping, the lemmatizer assumes
    every word is a noun and will not reduce verbs/adjectives
    correctly (e.g. "went" would stay "went" instead of becoming "go").
    """
    if treebank_tag.startswith("J"):
        return "a"  # adjective
    elif treebank_tag.startswith("V"):
        return "v"  # verb
    elif treebank_tag.startswith("N"):
        return "n"  # noun
    elif treebank_tag.startswith("R"):
        return "r"  # adverb
    else:
        return "n"  # default to noun


def preprocess_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """
    Full preprocessing pipeline for one piece of text:
      1. Lowercase + strip HTML tags/punctuation (character normalization)
      2. Tokenize into individual words
      3. Remove stop words (but negations are preserved, see step 1)
      4. POS-tag each remaining word, then lemmatize using that tag
    """
    # 1. Character normalization
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)          # strip HTML tags like <br>
    text = re.sub(r"[^a-z\s']", " ", text)      # keep letters, spaces, apostrophes

    # 2. Tokenization
    tokens = word_tokenize(text)

    # 3. Stop-word removal (negation-safe)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # 4. POS-guided lemmatization
    pos_tags = nltk.pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]

    return " ".join(lemmatized)


def preprocess_corpus(texts: pd.Series) -> pd.Series:
    stop_words = build_negation_safe_stopwords()
    lemmatizer = WordNetLemmatizer()
    return texts.apply(lambda t: preprocess_text(t, stop_words, lemmatizer))


# ============================================================
# STEP 3 & 4: TF-IDF VECTORIZATION (SPARSE CSR BY DEFAULT)
# ============================================================

def vectorize_text(train_texts, test_texts, max_features: int = 10000, min_df: int = 1):
    """
    Convert cleaned text into TF-IDF vectors.
      - ngram_range=(1, 2): captures both single words (unigrams) and
        two-word phrases (bigrams) like "not good", which preserves
        negated sentiment that unigrams alone would lose.
      - max_features: caps vocabulary size to control dimensionality.
      - min_df: ignores extremely rare terms/typos.

    scikit-learn's TfidfVectorizer already returns a SciPy CSR sparse
    matrix by default, so no extra conversion step is needed, this
    IS the "Memory Optimization" step from the brief.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
    )

    X_train = vectorizer.fit_transform(train_texts)  # sparse CSR matrix
    X_test = vectorizer.transform(test_texts)         # sparse CSR matrix

    print(f"TF-IDF matrix type: {type(X_train)}")
    print(f"Training matrix shape: {X_train.shape}")

    return X_train, X_test, vectorizer


# ============================================================
# STEP 5: NAIVE BAYES INFERENCE WITH LAPLACE SMOOTHING
# ============================================================

def train_naive_bayes(X_train, y_train, use_complement: bool = False):
    """
    alpha=1.0 applies Laplace Smoothing by default: this ensures that
    a single word never seen during training doesn't force the whole
    prediction probability to zero.

    Use ComplementNB instead of MultinomialNB when your sentiment
    classes are imbalanced (e.g. 95% positive, 5% negative), since it
    corrects for that bias.
    """
    if use_complement:
        model = ComplementNB(alpha=1.0)
    else:
        model = MultinomialNB(alpha=1.0)

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ============================================================
# FULL PIPELINE
# ============================================================

def run_sentiment_analysis():
    print("Loading data...")
    df = load_data()
    print(f"Total reviews: {len(df)}")
    print(f"Class balance:\n{df['sentiment'].value_counts()}")

    # Check balance to decide MultinomialNB vs ComplementNB
    class_counts = df["sentiment"].value_counts(normalize=True)
    is_imbalanced = class_counts.max() > 0.7
    use_complement = is_imbalanced
    print(f"\nDataset imbalanced: {is_imbalanced} -> using {'ComplementNB' if use_complement else 'MultinomialNB'}")

    print("\nStep 1 & 2: Preprocessing text (stop words + POS-guided lemmatization)...")
    df["cleaned_review"] = preprocess_corpus(df["review"])
    print("Example before/after:")
    print(f"  Before: {df['review'].iloc[0]}")
    print(f"  After:  {df['cleaned_review'].iloc[0]}")

    print("\nSplitting into train/test...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["cleaned_review"], df["sentiment"], test_size=0.25, random_state=42, stratify=df["sentiment"]
    )

    print("\nStep 3 & 4: TF-IDF vectorization (sparse CSR output)...")
    X_train, X_test, vectorizer = vectorize_text(X_train_text, X_test_text)

    print("\nStep 5: Training Naive Bayes classifier...")
    model = train_naive_bayes(X_train, y_train, use_complement=use_complement)

    evaluate_model(model, X_test, y_test)

    return model, vectorizer


def predict_new_review(model, vectorizer, review_text: str) -> str:
    """Helper to classify a brand-new, unseen review."""
    stop_words = build_negation_safe_stopwords()
    lemmatizer = WordNetLemmatizer()
    cleaned = preprocess_text(review_text, stop_words, lemmatizer)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return prediction


if __name__ == "__main__":
    model, vectorizer = run_sentiment_analysis()

    print("\n" + "=" * 50)
    print("TESTING ON NEW, UNSEEN REVIEWS")
    print("=" * 50)
    test_reviews = [
        "This is not good, very disappointing purchase.",
        "I absolutely love this, works perfectly!",
    ]
    for review in test_reviews:
        result = predict_new_review(model, vectorizer, review)
        print(f"Review: '{review}'  ->  Predicted: {result}")
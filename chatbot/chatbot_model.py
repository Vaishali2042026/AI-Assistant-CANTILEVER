import os
import re
import pickle
import difflib

import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHATBOT_DIR = os.path.join(BASE_DIR, "chatbot")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(CHATBOT_DIR, "chatbot_model.keras")
TOKENIZER_PATH = os.path.join(CHATBOT_DIR, "tokenizer.pkl")
WORD2VEC_PATH = os.path.join(CHATBOT_DIR, "word2vec.pkl")
MODEL_INFO_PATH = os.path.join(CHATBOT_DIR, "model_info.pkl")

DATASET_PATH = os.path.join(DATA_DIR, "conversation_dataset.csv")


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading conversation dataset...")

data = pd.read_csv(DATASET_PATH)

data = data.dropna(subset=["user_message", "bot_response"])

data["user_message"] = (
    data["user_message"]
    .astype(str)
    .str.strip()
)

data["bot_response"] = (
    data["bot_response"]
    .astype(str)
    .str.strip()
)

print(f"Loaded {len(data)} conversations.")


# ============================================================
# LOAD LSTM MODEL
# ============================================================

print("Loading LSTM model...")

model = load_model(MODEL_PATH)

print("LSTM model loaded successfully!")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

print("Tokenizer loaded successfully!")


# ============================================================
# LOAD WORD2VEC
# ============================================================

print("Loading Word2Vec model...")

with open(WORD2VEC_PATH, "rb") as file:
    word2vec_model = pickle.load(file)

print("Word2Vec model loaded successfully!")


# ============================================================
# LOAD MODEL INFORMATION
# ============================================================

if os.path.exists(MODEL_INFO_PATH):

    with open(MODEL_INFO_PATH, "rb") as file:
        model_info = pickle.load(file)

    max_length = model_info.get("max_length", 20)

else:

    max_length = 20


# ============================================================
# COMMON WORD CORRECTIONS
# ============================================================

WORD_CORRECTIONS = {

    # abbreviations
    "abt": "about",
    "u": "you",
    "ur": "your",
    "pls": "please",
    "plz": "please",
    "btw": "between",
    "wht": "what",
    "whats": "what",
    "whos": "who",
    "hows": "how",
    "wheres": "where",
    "cant": "cannot",
    "dont": "do not",
    "doesnt": "does not",
    "isnt": "is not",
    "im": "i am",
    "ive": "i have",
    "id": "i would",

    # common typing mistakes
    "helo": "hello",
    "helllo": "hello",
    "hii": "hi",
    "hiii": "hi",
    "heyy": "hey",
    "gm": "good morning",
    "gud": "good",
    "mornng": "morning",
    "mornin": "morning",
    "afternon": "afternoon",
    "evenng": "evening",

    # NLP / programming terms
    "pyhton": "python",
    "pythn": "python",
    "javscript": "javascript",
    "javasript": "javascript",
    "machne": "machine",
    "lernning": "learning",
    "learining": "learning",
    "artifical": "artificial",
    "inteligence": "intelligence",
    "intelligencee": "intelligence",
    "databse": "database",
    "databse": "database",
    "progrmming": "programming",
    "programing": "programming",
    "codng": "coding",
    "embeding": "embedding",
    "embeddingss": "embeddings",
    "algorthm": "algorithm",

    # chatbot terms
    "chatbt": "chatbot",
    "chatbott": "chatbot",
    "lstmm": "lstm",
    "rn": "rnn",
    "rnns": "rnn",
    "wordvec": "word2vec",
    "word2vecs": "word2vec",
}


# ============================================================
# STOP WORDS
#
# These words are not very useful for identifying the topic.
# For example:
#
# "tell me about BCA"
# "what is BCA"
#
# both should focus on BCA.
# ============================================================

STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "am",
    "was",
    "were",
    "be",
    "been",
    "being",
    "what",
    "who",
    "where",
    "when",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "tell",
    "me",
    "about",
    "please",
    "give",
    "explain",
    "describe",
    "you",
    "your",
    "i",
    "we",
    "it",
    "of",
    "to",
    "for",
    "in",
    "on",
    "and",
    "or",
    "my",
}


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower().strip()

    # Remove punctuation.
    text = re.sub(r"[^\w\s]", " ", text)

    # Fix joined common greetings.
    joined_words = {
        "goodmorning": "good morning",
        "goodafternoon": "good afternoon",
        "goodevening": "good evening",
        "goodbye": "good bye",
        "seeyoulater": "see you later",
    }

    for wrong, correct in joined_words.items():
        text = text.replace(wrong, correct)

    # Normalize multiple spaces.
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    corrected_words = []

    for word in words:

        if word in WORD_CORRECTIONS:
            corrected_words.extend(
                WORD_CORRECTIONS[word].split()
            )
        else:
            corrected_words.append(word)

    text = " ".join(corrected_words)

    return text


# ============================================================
# TOKENIZE NORMALIZED TEXT
# ============================================================

def get_words(text):

    normalized = normalize_text(text)

    if not normalized:
        return []

    return normalized.split()


# ============================================================
# IMPORTANT WORDS
#
# This is VERY important for your BCA/RNN problem.
#
# "tell me abt rnn"
#
# becomes:
#
# tell me about rnn
#
# Important word:
#
# rnn
#
# Therefore it will not incorrectly choose BCA just because
# both questions contain "tell me about".
# ============================================================

def get_important_words(text):

    words = get_words(text)

    important = []

    for word in words:

        if word not in STOP_WORDS:

            # Keep technical terms even if they are short.
            important.append(word)

    return important


# ============================================================
# HANDLE SIMPLE CHARACTER / TYPING ERRORS
# ============================================================

def fuzzy_word_correction(word):

    if word in WORD_CORRECTIONS:
        return WORD_CORRECTIONS[word]

    # Do not attempt fuzzy correction on extremely short words.
    if len(word) <= 3:
        return word

    known_words = set(tokenizer.word_index.keys())

    if not known_words:
        return word

    matches = difflib.get_close_matches(
        word,
        known_words,
        n=1,
        cutoff=0.88
    )

    if matches:
        return matches[0]

    return word


# ============================================================
# IMPROVE USER INPUT
# ============================================================

def smart_normalize(text):

    text = normalize_text(text)

    words = text.split()

    improved_words = []

    for word in words:

        corrected = fuzzy_word_correction(word)

        improved_words.append(corrected)

    return " ".join(improved_words)


# ============================================================
# WORD2VEC SENTENCE VECTOR
# ============================================================

def sentence_vector(text):

    words = get_words(text)

    vectors = []

    for word in words:

        if word in word2vec_model.wv:

            vectors.append(
                word2vec_model.wv[word]
            )

    if not vectors:

        return None

    return np.mean(vectors, axis=0)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):

    if vector_a is None or vector_b is None:
        return 0.0

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        / (norm_a * norm_b)
    )


# ============================================================
# WORD2VEC SIMILARITY
# ============================================================

def word2vec_similarity(user_text, question):

    user_vector = sentence_vector(user_text)
    question_vector = sentence_vector(question)

    similarity = cosine_similarity(
        user_vector,
        question_vector
    )

    # Convert possible negative similarity into 0-1 range.
    similarity = (similarity + 1) / 2

    return similarity


# ============================================================
# IMPORTANT WORD MATCHING
# ============================================================

def keyword_similarity(user_text, question):

    user_words = set(
        get_important_words(user_text)
    )

    question_words = set(
        get_important_words(question)
    )

    if not user_words or not question_words:
        return 0.0

    common_words = user_words.intersection(
        question_words
    )

    if not common_words:
        return 0.0

    # Calculate overlap relative to user's important words.
    user_overlap = (
        len(common_words) /
        len(user_words)
    )

    question_overlap = (
        len(common_words) /
        len(question_words)
    )

    # Favor matching important concepts.
    score = (
        0.7 * user_overlap +
        0.3 * question_overlap
    )

    return min(score, 1.0)


# ============================================================
# FUZZY SENTENCE MATCHING
# ============================================================

def fuzzy_similarity(user_text, question):

    user_text = smart_normalize(user_text)
    question = smart_normalize(question)

    if not user_text or not question:
        return 0.0

    sequence_score = difflib.SequenceMatcher(
        None,
        user_text,
        question
    ).ratio()

    user_words = set(get_words(user_text))
    question_words = set(get_words(question))

    if user_words and question_words:

        common = user_words.intersection(
            question_words
        )

        word_score = len(common) / max(
            len(user_words),
            len(question_words)
        )

    else:

        word_score = 0.0

    return (
        0.55 * sequence_score +
        0.45 * word_score
    )


# ============================================================
# EXACT / VERY CLOSE MATCH
# ============================================================

def exact_similarity(user_text, question):

    user_normalized = smart_normalize(user_text)
    question_normalized = smart_normalize(question)

    if user_normalized == question_normalized:
        return 1.0

    return 0.0


# ============================================================
# SPECIAL TECHNICAL KEYWORD BOOST
#
# This prevents:
#
# "tell me abt rnn"
#
# from selecting:
#
# "Tell me about BCA"
#
# because "rnn" is a highly meaningful technical word.
# ============================================================

TECHNICAL_TERMS = {
    "python",
    "java",
    "javascript",
    "html",
    "css",
    "flask",
    "github",
    "bca",
    "dbms",
    "database",
    "dataset",
    "chatbot",
    "nlp",
    "lstm",
    "rnn",
    "word2vec",
    "embedding",
    "embeddings",
    "machine",
    "learning",
    "programming",
    "coding",
    "artificial",
    "intelligence",
    "deep",
    "html",
    "css",
}


def technical_keyword_score(user_text, question):

    user_words = set(get_words(user_text))
    question_words = set(get_words(question))

    user_terms = user_words.intersection(
        TECHNICAL_TERMS
    )

    question_terms = question_words.intersection(
        TECHNICAL_TERMS
    )

    if not user_terms:
        return 0.0

    common_terms = user_terms.intersection(
        question_terms
    )

    if not common_terms:
        return 0.0

    # Very strong signal.
    return len(common_terms) / len(user_terms)


# ============================================================
# CALCULATE FINAL MATCH SCORE
# ============================================================

def calculate_match_score(user_text, question):

    exact_score = exact_similarity(
        user_text,
        question
    )

    # Exact normalized match should always win.
    if exact_score == 1.0:
        return 1.0

    keyword_score = keyword_similarity(
        user_text,
        question
    )

    technical_score = technical_keyword_score(
        user_text,
        question
    )

    w2v_score = word2vec_similarity(
        user_text,
        question
    )

    fuzzy_score = fuzzy_similarity(
        user_text,
        question
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Keyword and technical concept matching have higher
    # importance than Word2Vec.
    #
    # This is what fixes:
    #
    # tell me abt rnn
    #
    # incorrectly becoming BCA.
    # --------------------------------------------------------

    score = (
        0.40 * keyword_score +
        0.30 * technical_score +
        0.20 * w2v_score +
        0.10 * fuzzy_score
    )

    # Additional strong boost when a technical term matches.
    if technical_score > 0:

        score += 0.15 * technical_score

    return min(score, 1.0)


# ============================================================
# FIND BEST DATASET QUESTION
# ============================================================

def find_best_match(user_message):

    user_message = smart_normalize(
        user_message
    )

    best_question = None
    best_answer = None
    best_score = -1.0

    all_results = []

    for _, row in data.iterrows():

        question = row["user_message"]
        answer = row["bot_response"]

        score = calculate_match_score(
            user_message,
            question
        )

        all_results.append(
            (
                score,
                question,
                answer
            )
        )

        if score > best_score:

            best_score = score
            best_question = question
            best_answer = answer

    # Sort highest score first.
    all_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return (
        best_question,
        best_answer,
        best_score,
        all_results
    )


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(user_message):

    if not user_message:
        return "Please type a message."


    # --------------------------------------------------------
    # Normalize user's input
    # --------------------------------------------------------

    cleaned_message = smart_normalize(
        user_message
    )


    # --------------------------------------------------------
    # Find best matching CSV question
    # --------------------------------------------------------

    (
        matched_question,
        matched_answer,
        score,
        all_results
    ) = find_best_match(
        cleaned_message
    )


    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    print(
        f"[Normalized input: {cleaned_message}]"
    )

    print(
        f"[Matching score: {score:.3f}]"
    )

    print(
        f"[Matched question: {matched_question}]"
    )


    # --------------------------------------------------------
    # Confidence thresholds
    # --------------------------------------------------------
    #
    # High score:
    #   definitely answer.
    #
    # Medium score:
    #   answer if there is a meaningful keyword match.
    #
    # Low score:
    #   do not randomly choose an unrelated answer.
    # --------------------------------------------------------

    important_words = set(
        get_important_words(
            cleaned_message
        )
    )

    matched_important_words = set(
        get_important_words(
            matched_question
        )
    )

    common_important_words = (
        important_words.intersection(
            matched_important_words
        )
    )


    # --------------------------------------------------------
    # Strong match
    # --------------------------------------------------------

    if score >= 0.55:

        return matched_answer


    # --------------------------------------------------------
    # Medium match with meaningful shared words
    # --------------------------------------------------------

    if (
        score >= 0.40
        and len(common_important_words) >= 1
    ):

        return matched_answer


    # --------------------------------------------------------
    # Weak match
    #
    # DO NOT return a random dataset answer.
    # --------------------------------------------------------

    return (
        "I'm sorry, I don't understand that yet. "
        "Please try asking in another way."
    )


# ============================================================
# SHOW TOP MATCHES
#
# Useful while testing the chatbot.
# ============================================================

def show_top_matches(user_message, number=3):

    (
        matched_question,
        matched_answer,
        best_score,
        all_results
    ) = find_best_match(
        user_message
    )

    print()
    print("Top matches:")

    for i, result in enumerate(
        all_results[:number],
        start=1
    ):

        score, question, answer = result

        print(
            f"{i}. {score:.3f} -> {question}"
        )

    print()


# ============================================================
# TEST CHATBOT
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print("AI CHATBOT IS READY!")
    print("AI Assistant")
    print("==========================================")
    print("Type 'exit' to stop.")
    print()

    while True:

        user_message = input("You: ")

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if user_message.lower().strip() == "exit":

            print("Chatbot: Goodbye!")
            break

        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not user_message.strip():

            print("Chatbot: Please type something.")
            continue

        # ----------------------------------------------------
        # GENERATE RESPONSE
        # ----------------------------------------------------

        response = generate_response(
            user_message
        )

        print(
            "Chatbot:",
            response
        )

        print()
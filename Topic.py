# ============================================================
# INSTALL DEPENDENCIES
# ============================================================

!pip install bertopic sentence-transformers pdfplumber python-docx nltk tiktoken hdbscan scikit-learn


# ============================================================
# IMPORTS
# ============================================================

import re
import unicodedata
import pdfplumber
import docx
import nltk
import numpy as np
import tiktoken

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

nltk.download("punkt")


# ============================================================
# EMBEDDING MODEL (STRONG — ALL MINILM)
# ============================================================

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)


# ============================================================
# TOKEN COUNTER
# ============================================================

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(tokenizer.encode(text))


# ============================================================
# DOCUMENT LOADERS
# ============================================================

def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def load_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_document(path):
    if path.lower().endswith(".pdf"):
        return load_pdf(path)
    elif path.lower().endswith(".docx"):
        return load_docx(path)
    else:
        raise ValueError("Unsupported file format")


# ============================================================
# SAFE NORMALIZATION (NO OVER CLEANING)
# ============================================================

def normalize_text(text):

    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)

    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


# ============================================================
# SMART UNIT CREATION (PARAGRAPH BASED — NOT SENTENCE)
# ============================================================

def create_units(text, min_chars=200):

    paragraphs = text.split("\n\n")

    units = []
    buffer = ""

    for para in paragraphs:

        if len(buffer) + len(para) < min_chars:
            buffer += " " + para
        else:
            if buffer:
                units.append(buffer.strip())
            buffer = para

    if buffer:
        units.append(buffer.strip())

    return units


# ============================================================
# HDBSCAN CLUSTER MODEL (NO UMAP)
# ============================================================

def build_topic_model():

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=2,
        metric="euclidean",
        prediction_data=True
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=False
    )

    return topic_model


# ============================================================
# TOPIC GROUPING
# ============================================================

def topic_chunking(units):

    topic_model = build_topic_model()

    topics, probs = topic_model.fit_transform(units)

    topic_groups = {}

    for text, topic in zip(units, topics):

        if topic == -1:
            continue

        topic_groups.setdefault(topic, []).append(text)

    return topic_groups, topic_model


# ============================================================
# TOKEN AWARE CHUNK BUILDER (ROBUST)
# ============================================================

def build_chunks(topic_groups, max_tokens=300, overlap_tokens=50):

    final_chunks = []

    for topic, texts in topic_groups.items():

        buffer = ""

        for txt in texts:

            if count_tokens(buffer + txt) <= max_tokens:

                buffer += " " + txt

            else:

                final_chunks.append(buffer.strip())

                # overlap
                words = buffer.split()
                overlap = " ".join(words[-overlap_tokens:])

                buffer = overlap + " " + txt

        if buffer:
            final_chunks.append(buffer.strip())

    return final_chunks


# ============================================================
# EVALUATION METRICS
# ============================================================

def mean_pairwise_similarity(chunks):

    scores = []

    for chunk in chunks:

        sentences = sent_tokenize(chunk)

        if len(sentences) < 2:
            continue

        emb = embedding_model.encode(
            sentences,
            normalize_embeddings=True
        )

        sim = cosine_similarity(emb)

        upper = sim[np.triu_indices(len(sentences), k=1)]

        if len(upper) > 0:
            scores.append(np.mean(upper))

    return float(np.mean(scores)) if scores else 0.0


def chunk_stats(chunks):

    lengths = [count_tokens(c) for c in chunks]

    return {
        "num_chunks": len(chunks),
        "avg_tokens": int(np.mean(lengths)) if lengths else 0,
        "max_tokens": int(np.max(lengths)) if lengths else 0,
        "min_tokens": int(np.min(lengths)) if lengths else 0,
    }


def topic_diversity(topic_model):

    topics = topic_model.get_topics()

    unique_words = set()

    for topic_id in topics:

        words = [w for w, _ in topics[topic_id]]
        unique_words.update(words)

    return len(unique_words)


# ============================================================
# FULL PIPELINE
# ============================================================

def process_document(path):

    print("Loading document...")
    raw = load_document(path)

    print("Normalizing...")
    text = normalize_text(raw)

    print("Creating semantic units...")
    units = create_units(text)

    print("Total units:", len(units))

    print("Topic modeling...")
    topic_groups, topic_model = topic_chunking(units)

    print("Building chunks...")
    chunks = build_chunks(topic_groups)

    print("Evaluating...")
    sim_score = mean_pairwise_similarity(chunks)
    stats = chunk_stats(chunks)
    diversity = topic_diversity(topic_model)

    print("\n========== RESULTS ==========")
    print("Total Chunks:", stats["num_chunks"])
    print("Average Tokens:", stats["avg_tokens"])
    print("Max Tokens:", stats["max_tokens"])
    print("Min Tokens:", stats["min_tokens"])
    print("Mean Pairwise Similarity:", round(sim_score, 4))
    print("Topic Diversity:", diversity)

    return chunks, topic_model


# ============================================================
# RUN EXAMPLE
# ============================================================

file_path = "your_document.pdf"   # change path here

chunks, topic_model = process_document(file_path)

for i, ch in enumerate(chunks[:5]):
    print(f"\n------ Chunk {i+1} ------\n")
    print(ch[:500])

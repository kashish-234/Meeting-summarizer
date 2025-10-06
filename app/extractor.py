import spacy
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch

# Load spaCy NLP model for NER and dependency parsing
nlp = spacy.load("en_core_web_sm")

# Sentence transformer for semantic topic grouping
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

ACTION_KEYWORDS = ["please", "will", "assign", "action", "todo", "follow up", "task", "responsible", "deadline"]

def extract_actions_and_topics(transcript: str):
    """
    Extract actions (with owner/assignee if possible) and cluster topics semantically.
    Returns a dict: {"actions": [...], "topics": [...]}
    """
    doc = nlp(transcript)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    # --- Extract Actions ---
    actions = []
    for sent in sentences:
        if any(k in sent.lower() for k in ACTION_KEYWORDS):
            action = {"text": sent, "assignees": [], "due": None}
            sent_doc = nlp(sent)
            
            # Extract person names (potential assignees)
            action["assignees"] = [ent.text for ent in sent_doc.ents if ent.label_ == "PERSON"]
            
            # Extract dates (potential deadlines)
            action["due"] = [ent.text for ent in sent_doc.ents if ent.label_ in ["DATE", "TIME"]]
            
            actions.append(action)
    
    # --- Extract Topics (semantic clustering) ---
    embeddings = embed_model.encode(sentences, convert_to_tensor=True)
    # Use cosine similarity to cluster semantically similar sentences
    # Simple approach: group sentences with similarity > 0.6
    clusters = defaultdict(list)
    used = set()
    for i, emb_i in enumerate(embeddings):
        if i in used:
            continue
        clusters[i].append(sentences[i])
        used.add(i)
        for j, emb_j in enumerate(embeddings):
            if j in used:
                continue
            sim = util.cos_sim(emb_i, emb_j)
            if sim > 0.6:
                clusters[i].append(sentences[j])
                used.add(j)
    
    topics = ["; ".join(cluster) for cluster in clusters.values()]
    
    return {"actions": actions, "topics": topics}

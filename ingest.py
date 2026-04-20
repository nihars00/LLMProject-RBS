import os
import re
import json
import pickle
from glob import glob
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

DATA_DIR = "data"
OUTPUT_DIR = "output"

def setup_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def parse_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by the separator used in the text files
    blocks = content.split("========================================================================")
    
    documents = []
    # Try to extract category from the top block
    top_header = blocks[0]
    category_match = re.search(r'# Category:\s*(.*)', top_header)
    category = category_match.group(1).strip() if category_match else "general"

    for block in blocks[1:]:
        block = block.strip()
        if not block:
            continue
        
        # Parse basic metadata: ## section, URL:, Description:, Title:
        section_match = re.search(r'^##\s*(.*?)\n', block)
        url_match = re.search(r'URL:\s*(.*?)\n', block)
        desc_match = re.search(r'Description:\s*(.*?)\n', block)
        title_match = re.search(r'Title:\s*(.*?)\n', block)

        section = section_match.group(1).strip() if section_match else ""
        url = url_match.group(1).strip() if url_match else ""
        desc = desc_match.group(1).strip() if desc_match else ""
        title = title_match.group(1).strip() if title_match else ""
        
        doc = {
            "category": category,
            "section": section,
            "url": url,
            "description": desc,
            "title": title,
            "raw_text": block
        }
        documents.append(doc)
    return documents

def extract_structured_data(docs):
    contacts = []
    events = []
    requirements = []

    for doc in docs:
        text = doc["raw_text"]
        cat = doc["category"]

        # Extract Emails for Contacts
        if cat in ["contacts", "general"]:
            emails = re.findall(r'[\w\.-]+@business\.rutgers\.edu', text)
            if emails:
                contacts.append({
                    "section": doc["section"],
                    "title": doc["title"],
                    "emails": list(set(emails)),
                    "url": doc["url"]
                })
        
        # Extract Dates for Events (naive regex for dates like Mar 28, 2026)
        if cat == "events":
            dates = re.findall(r'[A-Z][a-z]{2}\s\d{1,2},\s\d{4}', text)
            if dates:
                events.append({
                    "title": doc["title"],
                    "dates_found": list(set(dates)),
                    "url": doc["url"]
                })
        
        # Extract Requirements for requirements.txt (search for 'credits')
        if cat == "requirements":
            credits_mention = re.findall(r'(\d+)\s*credits', text, re.IGNORECASE)
            if credits_mention:
                requirements.append({
                    "section": doc["section"],
                    "credits_mentioned": credits_mention,
                    "url": doc["url"]
                })
    
    with open(os.path.join(OUTPUT_DIR, "contacts.json"), 'w') as f:
        json.dump(contacts, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "events.json"), 'w') as f:
        json.dump(events, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "requirements.json"), 'w') as f:
        json.dump(requirements, f, indent=2)
    
    print(f"Structured data saved! Extracted {len(contacts)} contact blocks, {len(events)} event blocks, {len(requirements)} req blocks.")

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def build_index(all_docs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    chunked_data = []
    # Build text chunks
    for doc in all_docs:
        chunks = chunk_text(doc["raw_text"])
        for c in chunks:
            chunked_data.append({
                "category": doc["category"],
                "url": doc["url"],
                "title": doc["title"],
                "text": c,
                "metadata_prefix": f"[{doc['category'].upper()}] {doc['title']} ({doc['url']}):\n"
            })
    
    print(f"Total chunks generated: {len(chunked_data)}")
    
    # 1. FAISS Index (Dense)
    print("Computing embeddings...")
    texts_to_embed = [item["metadata_prefix"] + item["text"] for item in chunked_data]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    
    faiss.write_index(faiss_index, os.path.join(OUTPUT_DIR, "vector_index.faiss"))
    
    # 2. BM25 Index (Sparse)
    tokenized_corpus = [text.lower().split() for text in texts_to_embed]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save artifacts
    with open(os.path.join(OUTPUT_DIR, "chunked_data.pkl"), "wb") as f:
        pickle.dump(chunked_data, f)
        
    with open(os.path.join(OUTPUT_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(bm25, f)
        
    print("Indexes built entirely and saved to /output folder.")

if __name__ == "__main__":
    setup_output_dir()
    
    all_docs = []
    files = glob(os.path.join(DATA_DIR, "*.txt"))
    for f in files:
        docs = parse_file(f)
        all_docs.extend(docs)
        
    print(f"Loaded {len(all_docs)} sections across {len(files)} files.")
    
    extract_structured_data(all_docs)
    build_index(all_docs)

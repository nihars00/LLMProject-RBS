# RUStudentAssist - Student Life Assistant for Rutgers Business School

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![RAG](https://img.shields.io/badge/Architecture-RAG-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![BM25](https://img.shields.io/badge/Retrieval-BM25%20%2B%20Dense-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🚀 Overview

**Student Life Assistant for Rutgers Business School (RBS)** is a **Retrieval-Augmented Generation (RAG)** system that answers student questions about:

* 📞 Program contacts
* 📅 Events
* 🎓 Academic requirements
* 🏢 Student organizations
* 🌐 General student resources

The system uses **hybrid retrieval (BM25 + dense embeddings)** to provide **accurate, grounded, citation-based responses** using real Rutgers data.

---

## 🎯 Example Questions

* Who is the contact for the MITA program?
* What supply chain events are happening this week?
* What are the requirements for the Business Administration minor?
* What student organizations are available for supply chain students?

---

## 🧠 Key Features

* ✅ Retrieval-Augmented Generation (RAG)
* ✅ Hybrid search (Dense + BM25)
* ✅ Query routing (intent-based)
* ✅ Structured data extraction (JSON for events & contacts)
* ✅ Citation-based answers (no hallucinations)
* ✅ Lightweight & low-cost (< $30)

---
## 🏗️ System Architecture

```mermaid
graph TD
    User[User Query] --> Router{Query Router}
    
    Router -->|Contacts| BM25_Contacts[BM25 Contacts]
    Router -->|Events| Event_Filter[Event JSON Filter]
    Router -->|Requirements| BM25_Reqs[BM25 Requirements]
    Router -->|Other| Hybrid[Hybrid Search]

    Hybrid --> Dense[FAISS Dense Retrieval]
    Hybrid --> Sparse[BM25 Sparse Retrieval]

    Dense --> Fusion
    Sparse --> Fusion
    BM25_Contacts --> Fusion
    BM25_Reqs --> Fusion
    Event_Filter --> Fusion

    Fusion[Rank Fusion RRF] --> TopK[Top K Chunks]
    TopK --> Prompt[Prompt Builder]
    Prompt --> LLM[LLM Generation]
    LLM --> Output[Answer with Citations]
```

---

## 🛠️ Tech Stack

| Component      | Tool                                      |
| -------------- | ----------------------------------------- |
| Embeddings     | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector DB      | FAISS                                     |
| Keyword Search | BM25 (`rank_bm25`)                        |
| LLM            | GPT-4o-mini / Gemini Flash / Local LLM    |
| UI             | Streamlit                                 |
| Language       | Python                                    |

---

## 📂 Project Structure

```
rbs-student-life-assistant/
│
├── data/
│   ├── raw/                # scraped text files
│   ├── processed/          # cleaned + chunked data
│   └── structured/         # JSON (contacts, events, etc.)
│
├── src/
│   ├── ingest.py           # data processing pipeline
│   ├── retrieval.py        # FAISS + BM25 + hybrid logic
│   ├── generator.py        # LLM prompting + response
│   ├── router.py           # query routing logic
│   └── utils.py
│
├── app/
│   └── app.py              # Streamlit UI
│
├── eval/
│   ├── benchmark.json
│   └── evaluate.py
│
├── notebooks/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/rbs-student-life-assistant.git
cd rbs-student-life-assistant

pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 1. Build Index

```bash
python src/ingest.py
```

### 2. Run App

```bash
streamlit run app/app.py
```

---

## 🔍 Retrieval Strategy

### Baseline

* Dense retrieval (FAISS)

### Final System (Improved)

* Hybrid Retrieval:

  * Dense embeddings (semantic)
  * BM25 (keyword)
  * Combined via **Reciprocal Rank Fusion (RRF)**

---

## 🧪 Evaluation

We evaluate the system using:

* 📊 Retrieval Accuracy (Hits@K)
* 🧠 Answer Correctness
* 🚫 Hallucination Rate
* 📎 Citation Accuracy

### Comparison

| Method            | Accuracy | Latency | Notes                  |
| ----------------- | -------- | ------- | ---------------------- |
| Dense Only        | Medium   | Fast    | Misses keyword matches |
| Hybrid (Final)    | High     | Medium  | Best balance           |
| Hybrid + Reranker | Highest  | Slow    | Too expensive          |

---

## ⚖️ Tradeoffs

| Aspect     | Choice               | Reason                      |
| ---------- | -------------------- | --------------------------- |
| Retrieval  | Hybrid               | Better accuracy             |
| Model      | Cheap API            | Reliable + low cost         |
| Chunk Size | Medium (~400 tokens) | Balance context + precision |
| Structure  | JSON + Text          | Improves precision          |

---

## ⚠️ Limitations

* ❗ Missing data (e.g., Finance minor ambiguity)
* ⏳ Events may become outdated
* 🏫 Campus differences (Newark vs NB)
* 🔒 No personalization (no student-specific data)

---


## 📚 Course Project Context


* RAG system design
* Evaluation + benchmarking
* Tradeoff analysis
* Low-cost implementation
* Real-world dataset usage

---

## ⭐ Acknowledgements

* Rutgers Business School
* Open-source ML community
* SentenceTransformers, FAISS, BM25

---

## 📜 License

MIT License
# LLMProject-RBS

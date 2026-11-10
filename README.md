# ShopWise — E-Commerce Product Finder with RAG

A beginner-to-intermediate Retrieval-Augmented Generation (RAG) project that helps shoppers find and compare products using natural-language queries. It uses sentence embeddings for semantic retrieval, ChromaDB as a persistent vector store, and Google Gemini for optional grounded explanations.

> **Dataset note:** `data/products.json` is a small, illustrative sample catalog created for this project. Prices, ratings, reviews, and product descriptions are demo data, not live listings or verified market information.

## Features

- Natural-language product search (for example, “lightweight headphones for travel with long battery life”).
- Semantic retrieval using `all-MiniLM-L6-v2` embeddings.
- Persistent vector storage and similarity search with ChromaDB.
- Category and price filters.
- Gemini-generated explanations grounded in retrieved product details (optional API key).
- Product comparison table.
- Retrieval similarity scores to make search results more transparent.
- Small, editable JSON product catalog.

## RAG workflow

1. **Load:** Read product details from `data/products.json`.
2. **Prepare:** Combine each product's name, category, description, specifications, and reviews into a text document.
3. **Embed:** Convert product documents into vectors using Sentence Transformers.
4. **Index:** Store vectors, source documents, and product metadata in ChromaDB.
5. **Retrieve:** Embed the user's query and retrieve semantically similar products, applying category and price filters.
6. **Generate:** Send only the retrieved product context to Gemini to create a grounded explanation. If no API key is configured, semantic search still works and the app displays the retrieved products.

## Tech stack

- Python
- Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- Google Gemini API (optional)

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/shopwise-rag.git
cd shopwise-rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini (optional)

Copy `.env.example` to `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_api_key
```

Do not commit `.env` or share your API key. The app works without a key; Gemini is only used for natural-language recommendation explanations.

### 5. Start the app

```bash
streamlit run app.py
```

The first run downloads the embedding model and creates the local ChromaDB index. This can take a few minutes. The generated `chroma_db/` folder is local runtime data and is excluded from Git.

## Project structure

```text
shopwise-rag/
├── app.py
├── data/
│   └── products.json
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Example queries

- “I need headphones for long flights with noise cancellation.”
- “Find a laptop for programming under ₹70,000.”
- “Show comfortable shoes for running.”
- “I want a book to learn SQL and data analysis.”

## Limitations and next steps

- The included catalog is intentionally small and uses synthetic demo information.
- Semantic similarity is not the same as a proven recommendation-quality metric. Build a labeled query-to-product test set before reporting Precision@K or Recall@K.
- Product availability, live prices, and verified reviews are not fetched from a retailer.
- Possible extensions: add a larger licensed catalog, collect relevance feedback, evaluate retrieval with a labeled test set, and add automated tests.

## Attribution and originality

This repository is an independently written implementation of a common e-commerce RAG project idea. It is not a copy of another repository's source code. If you use code, assets, or datasets from third parties, check their licenses and provide attribution as required.

## Responsible use

This is an educational demo, not a live shopping service. Confirm product details and prices independently before making a purchase.

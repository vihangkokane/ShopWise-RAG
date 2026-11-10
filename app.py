"""Context-aware e-commerce product recommender powered by RAG."""

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "products.json"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "ecommerce_products"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

load_dotenv(BASE_DIR / ".env")


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_products() -> list[dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        products = json.load(file)
    required = {"id", "name", "category", "price", "description", "specifications"}
    for product in products:
        missing = required - product.keys()
        if missing:
            raise ValueError(f"Product {product.get('name', '<unknown>')} is missing: {missing}")
    return products


def product_document(product: dict[str, Any]) -> str:
    """Create the text that will be embedded and retrieved."""
    reviews = " ".join(product.get("reviews", []))
    return (
        f"Product: {product['name']}. Category: {product['category']}. "
        f"Description: {product['description']}. "
        f"Specifications: {product['specifications']}. "
        f"Customer reviews: {reviews}"
    )


def index_products(products: list[dict[str, Any]], embedder, collection) -> None:
    """Refresh the small demo index safely; upsert avoids duplicate IDs."""
    ids = [p["id"] for p in products]
    documents = [product_document(p) for p in products]
    embeddings = embedder.encode(documents, normalize_embeddings=True).tolist()
    metadatas = [
        {
            "name": p["name"],
            "category": p["category"],
            "price": float(p["price"]),
            "description": p["description"],
            "specifications": p["specifications"],
            "rating": float(p.get("rating", 0)),
            "image_url": p.get("image_url", ""),
        }
        for p in products
    ]
    # Remove stale IDs if the sample catalog has been edited.
    existing = collection.get()
    existing_ids = existing.get("ids", [])
    stale_ids = [item_id for item_id in existing_ids if item_id not in ids]
    if stale_ids:
        collection.delete(ids=stale_ids)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def retrieve_products(query: str, collection, embedder, category: str, min_price: float,
                      max_price: float, top_k: int = 5) -> list[dict[str, Any]]:
    query_vector = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    filters = []
    if category != "All":
        filters.append({"category": {"$eq": category}})
    filters.extend([
        {"price": {"$gte": float(min_price)}},
        {"price": {"$lte": float(max_price)}},
    ])
    where = filters[0] if len(filters) == 1 else {"$and": filters}
    count = collection.count()
    if count == 0:
        return []
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(max(top_k, 1), count),
        where=where,
        include=["metadatas", "documents", "distances"],
    )
    recommendations = []
    for item_id, metadata, document, distance in zip(
        result["ids"][0], result["metadatas"][0], result["documents"][0], result["distances"][0]
    ):
        recommendations.append({
            "id": item_id,
            **metadata,
            "document": document,
            "similarity": max(0.0, min(1.0, 1.0 - float(distance))),
        })
    return recommendations


def generate_recommendation_answer(query: str, products: list[dict[str, Any]]) -> str:
    """Generate a grounded answer with Gemini; never invent unavailable products."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "Add your GEMINI_API_KEY to a local .env file to enable generated explanations. "
            "The retrieved products below are still available from semantic search."
        )
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        context = "\n\n".join(
            f"{i}. {p['name']} | Category: {p['category']} | Price: ₹{p['price']:.0f} | "
            f"Rating: {p['rating']}/5 | Description: {p['description']} | "
            f"Specifications: {p['specifications']}"
            for i, p in enumerate(products, start=1)
        )
        prompt = f"""You are an e-commerce shopping assistant. Answer using only the product facts below.
If none of the products fits the request, say so. Do not invent prices, features, or products.
Briefly explain why each suggested product matches the user's needs.

User request: {query}

Retrieved product context:
{context}
"""
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text or "I couldn't generate an explanation. Please review the retrieved products."
    except Exception as exc:
        return f"AI explanation is currently unavailable ({type(exc).__name__}). You can still review the retrieved products below."


def main() -> None:
    st.set_page_config(page_title="ShopWise | AI Product Finder", page_icon="🛍️", layout="wide")
    st.title("🛍️ ShopWise: AI Product Finder")
    st.caption("Find products using natural-language search, semantic retrieval, and grounded AI explanations.")

    try:
        products = load_products()
        embedder = load_embedder()
        collection = load_collection()
        index_products(products, embedder, collection)
    except Exception as exc:
        st.error(f"Could not initialize the product catalog: {exc}")
        st.stop()

    categories = ["All"] + sorted({p["category"] for p in products})
    with st.sidebar:
        st.header("Search filters")
        category = st.selectbox("Category", categories)
        catalog_min = min(float(p["price"]) for p in products)
        catalog_max = max(float(p["price"]) for p in products)
        min_price, max_price = st.slider(
            "Price range (₹)", min_value=int(catalog_min), max_value=int(catalog_max),
            value=(int(catalog_min), int(catalog_max)),
        )
        st.divider()
        st.caption(f"Catalog: {len(products)} sample products")
        st.caption("Embeddings: all-MiniLM-L6-v2")
        st.caption("Vector store: ChromaDB")

    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. Lightweight headphones for travel with good battery life",
    )
    search_clicked = st.button("Find products", type="primary", disabled=not query.strip())

    if search_clicked:
        with st.spinner("Searching the product catalog..."):
            results = retrieve_products(query, collection, embedder, category, min_price, max_price)
        st.session_state["last_query"] = query
        st.session_state["last_results"] = results

    results = st.session_state.get("last_results", [])
    previous_query = st.session_state.get("last_query", "")
    if results:
        st.subheader("AI shopping guidance")
        st.write(generate_recommendation_answer(previous_query, results))
        st.subheader(f"Matching products ({len(results)})")
        for product in results:
            with st.container(border=True):
                col1, col2 = st.columns([1, 3])
                with col1:
                    if product.get("image_url"):
                        st.image(product["image_url"], use_container_width=True)
                with col2:
                    st.markdown(f"### {product['name']}")
                    st.markdown(f"**₹{product['price']:,.0f}** · {product['category']} · ⭐ {product['rating']}/5")
                    st.write(product["description"])
                    st.caption(f"Specifications: {product['specifications']}")
                    st.caption(f"Semantic similarity: {product['similarity']:.3f}")

        if len(results) >= 2:
            st.subheader("Compare results")
            compare_names = [p["name"] for p in results]
            selected_names = st.multiselect(
                "Choose products to compare", compare_names,
                default=compare_names[: min(3, len(compare_names))],
            )
            selected = [p for p in results if p["name"] in selected_names]
            if selected:
                st.dataframe(
                    [{"Product": p["name"], "Price (₹)": p["price"], "Category": p["category"],
                      "Rating": p["rating"], "Specifications": p["specifications"]} for p in selected],
                    use_container_width=True, hide_index=True,
                )
    elif search_clicked:
        st.warning("No products matched those filters. Try a different query or widen the price range.")
    else:
        st.info("Enter a natural-language request above to search the sample product catalog.")

    with st.expander("How this application works"):
        st.markdown(
            "1. Product descriptions, specifications, and reviews are combined into documents.\n"
            "2. Sentence Transformers converts each document into an embedding.\n"
            "3. ChromaDB stores the vectors and retrieves semantically similar products.\n"
            "4. Retrieved product details are supplied to Gemini to generate a grounded explanation.\n"
            "5. Category and price filters narrow the search results."
        )


if __name__ == "__main__":
    main()

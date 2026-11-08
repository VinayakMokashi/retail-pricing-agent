"""
Pricing agent grounded in a pricing knowledge base (LangChain + Groq + Chroma).

Stage 1 priced a product from its five inputs alone. Stage 2 answered
questions from a price list. This stage combines the two: the agent recommends
a selling price and backs it with evidence retrieved from a knowledge base of
the company's pricing history:

- elasticity benchmarks per category
- historical margins per category
- past pricing decisions with their outcomes, failures as well as successes
- competitor trends per category
- pricing guidelines

The knowledge base (knowledge_base.json) is fictional Indian retail data in
rupees. new_category.json adds a sixth category, Sports & Fitness, to show how
an answer changes when new knowledge arrives.

Each record becomes one document with metadata, so no chunking is needed.
Retrieval is a set of small filtered searches, one per kind of evidence,
instead of one similarity search, which mixed in other categories and confused
High with Low elasticity. The answer cites the numbered documents, like [3],
so every claim can be traced back to its source.

How to run
----------
1. Copy `.env.example` at the repository root to `.env` and add your Groq API key.
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python 03_pricing_knowledge_rag/pricing_knowledge_rag.py

With no arguments it prices the reference product (Puma sneakers: cost
₹1,800, 30% target margin, High elasticity). Run with --help to price your own
product, compare with a prompt-only baseline, add the new category, or inspect
retrieval (no API key needed).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load GROQ_API_KEY from the .env file at the repository root.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KNOWLEDGE_BASE_PATH = Path(__file__).with_name("knowledge_base.json")
NEW_CATEGORY_PATH = Path(__file__).with_name("new_category.json")
ELASTICITY_LEVELS = ("Low", "Medium", "High")

# Reference product used by the CLI defaults and the walkthrough notebook.
EXAMPLE_PRODUCT = {
    "product_name": "Puma Sneakers",
    "category": "Footwear",
    "cost_price": 1800,
    "target_margin": 30,
    "price_elasticity": "High",
}

# How many documents of each kind to retrieve for a product's category. Four past
# decisions, not three: for big-ticket fitness equipment, the most relevant decision
# (a treadmill) ranks only fourth by similarity.
CATEGORY_EVIDENCE = [("elasticity_benchmark", 1), ("historical_margin", 1),
                     ("competitor_trend", 1), ("past_decision", 4)]
GENERAL_GUIDELINES = 2

# Rules shared by the RAG agent and the prompt-only baseline, so a comparison between
# them shows the effect of retrieval and nothing else.
PRICING_RULES = """- Gross margin is measured on selling price: margin = (price - cost) / price, so the minimum price for a target margin = cost / (1 - target margin). Never recommend a price below it: when rounding to a price point, round up.
- If you lack information you need, say so plainly. Never invent market data.
- Show your calculations in plain text, with no tables or LaTeX, and keep the whole answer under 250 words."""

ANSWER_SECTIONS = """1. Margin floor: the lowest price that meets the target margin (show the calculation).
2. Evidence: the facts your price is based on, and what each one suggests.
3. Recommendation: one price, its margin, and where it sits against the competitor or market price.
4. Risks: what to watch after launch, in one or two lines."""

RAG_SYSTEM_PROMPT = f"""You are PriceWise, a retail pricing strategist with access to the company's pricing knowledge base: category elasticity benchmarks, historical margins, past pricing decisions with their outcomes, competitor trends and pricing guidelines.

Rules:
- Ground every market fact (elasticity, typical margins, competitor prices, past results) in the numbered knowledge-base context you are given, and cite it like [2]. If the context does not cover something, say so.
- Learn from past decisions: repeat what worked in similar situations and avoid what failed.
{PRICING_RULES}"""

BASELINE_SYSTEM_PROMPT = f"""You are PriceWise, a retail pricing strategist.

Rules:
{PRICING_RULES}"""


# --- Knowledge base ---------------------------------------------------------------

def load_knowledge(path=KNOWLEDGE_BASE_PATH):
    """Load a knowledge-base file and add each past decision's gross margin."""
    knowledge = json.loads(Path(path).read_text(encoding="utf-8"))
    # Calculated, not stored, so a margin always matches its prices.
    for decision in knowledge["pricing_decisions"]:
        price = decision["recommended_price"]
        decision["margin"] = f"{(price - decision['cost']) / price:.1%}"
    return knowledge


def create_documents(knowledge):
    """Turn every record into one small, self-contained document.

    The metadata lets the agent filter by category, document type and the
    elasticity a guideline applies to. The doc_id is built from the content, so
    adding the same record again replaces it instead of duplicating it.
    """
    documents = []

    for b in knowledge["elasticity_benchmarks"]:
        content = (f"Elasticity benchmark for {b['category']}: {b['elasticity']} price elasticity "
                   f"(coefficient {b['coefficient']}). {b['notes']}")
        documents.append(Document(page_content=content, metadata={
            "doc_type": "elasticity_benchmark", "category": b["category"],
            "elasticity": b["elasticity"], "doc_id": f"benchmark:{b['category']}"}))

    for m in knowledge["historical_margins"]:
        content = (f"Historical margins for {m['category']}: average gross margin {m['avg_margin']}, "
                   f"typical range {m['typical_range']}. {m['notes']}")
        documents.append(Document(page_content=content, metadata={
            "doc_type": "historical_margin", "category": m["category"], "doc_id": f"margins:{m['category']}"}))

    for d in knowledge["pricing_decisions"]:
        content = (f"Past pricing decision: {d['product']} ({d['category']}, {d['elasticity']} elasticity). "
                   f"Cost ₹{d['cost']:,}, price ₹{d['recommended_price']:,} ({d['margin']} margin), "
                   f"competitor price ₹{d['competitor_price']:,}. Outcome: {d['outcome']}. "
                   f"Reasoning: {d['reasoning']}.")
        documents.append(Document(page_content=content, metadata={
            "doc_type": "past_decision", "category": d["category"], "doc_id": f"decision:{d['product']}"}))

    for t in knowledge["competitor_trends"]:
        content = f"Competitor trends in {t['category']}: {t['trend']}"
        documents.append(Document(page_content=content, metadata={
            "doc_type": "competitor_trend", "category": t["category"], "doc_id": f"trend:{t['category']}"}))

    for g in knowledge["pricing_guidelines"]:
        content = (f"Pricing guideline - {g['title']} (applies to {g['applies_to']}): {g['rule']} "
                   f"Why: {g['rationale']}")
        documents.append(Document(page_content=content, metadata={
            "doc_type": "guideline", "category": "All", "applies_to": g["applies_to"],
            "doc_id": f"guideline:{g['title']}"}))

    return documents


def make_embeddings():
    """Create the local embedding model (384 dimensions; the first run downloads it)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})


def add_documents(store, documents):
    """Add documents under their doc_id, so a record that is already there is replaced."""
    store.add_documents(documents, ids=[doc.metadata["doc_id"] for doc in documents])


def build_store(embeddings, knowledge=None, name="pricing_knowledge"):
    """Embed the knowledge base into an in-memory Chroma collection (nothing is written to disk)."""
    store = Chroma(collection_name=name, embedding_function=embeddings)
    add_documents(store, create_documents(knowledge or load_knowledge()))
    return store


# --- Model and prompts --------------------------------------------------------------

def make_llm():
    """Create the Groq chat model shared by the agent and the baseline."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env at the repository root "
            "and add your Groq API key."
        )
    # Same model and temperature as stage 1. gpt-oss reasons before it answers and those
    # tokens count towards max_tokens; max_retries rides out free-tier rate limits.
    return ChatGroq(model=MODEL, temperature=0.2, max_tokens=2048, max_retries=6)


def money(value):
    """Format a rupee amount, or say that it is missing."""
    if value is None:
        return "not provided"
    return f"₹{value:,.0f}" if float(value).is_integer() else f"₹{value:,.2f}"


def describe_product(product_name, category, cost_price, current_price=None, target_margin=None,
                     competitor_price=None, price_elasticity=None):
    """The product details as prompt lines; inputs that were not given say so."""
    return "\n".join([
        f"- Product: {product_name}",
        f"- Category: {category}",
        f"- Cost price: {money(cost_price)}",
        f"- Current price: {money(current_price)}",
        f"- Target margin: {f'{target_margin:g}%' if target_margin is not None else 'not provided'}",
        f"- Competitor price: {money(competitor_price)}",
        f"- Price elasticity: {price_elasticity or 'not provided'}",
    ])


def answer_text(response):
    """The model's answer, flagged if it hit max_tokens (gpt-oss's hidden reasoning counts
    toward the limit, and a cut-off answer can come back short or empty)."""
    if response.response_metadata.get("finish_reason") == "length":
        return "[Cut off at max_tokens: run again or raise max_tokens]\n" + response.content
    return response.content


def format_context(docs):
    """Number the retrieved documents so the answer can cite them."""
    return "\n\n".join(f"[{i}] ({doc.metadata['doc_type'].replace('_', ' ')}) {doc.page_content}"
                       for i, doc in enumerate(docs, 1))


def cited_numbers(answer, n_docs):
    """The document numbers an answer cites. gpt-oss often writes 【2】 instead of [2]."""
    return sorted({int(n) for n in re.findall(r"[\[【](\d+)[\]】]", answer) if 0 < int(n) <= n_docs})


# --- The agent ----------------------------------------------------------------------

class RAGPricingAgent:
    def __init__(self, store, llm=None):
        self.store = store
        self.llm = llm or make_llm()
        self.last_docs = []  # the documents behind the latest answer, numbered as in its citations

    def retrieve(self, query, k=5, filter=None):
        """Similarity search, optionally limited to documents whose metadata matches filter."""
        return self.store.similarity_search(query, k=k, filter=filter)

    def retrieve_product_knowledge(self, query, category, price_elasticity=None):
        """Run one filtered search per kind of evidence.

        This guarantees the context has the category's benchmark, margins and
        competitor trend, its most similar past decisions, and the guidelines
        that apply. If the category is unknown, it fetches the guideline for
        categories with no history instead.
        """
        docs = []
        for doc_type, k in CATEGORY_EVIDENCE:
            docs += self.retrieve(query, k=k, filter={"$and": [{"category": category}, {"doc_type": doc_type}]})
        if not docs:
            docs += self.retrieve(query, k=1, filter={"applies_to": "Categories with no history"})

        # The guideline for this elasticity: the one given, or else the category benchmark's
        elasticity = price_elasticity or next(
            (doc.metadata["elasticity"] for doc in docs if doc.metadata["doc_type"] == "elasticity_benchmark"), None)
        if elasticity:
            docs += self.retrieve(query, k=1, filter={"applies_to": f"{elasticity} elasticity"})
        docs += self.retrieve(query, k=GENERAL_GUIDELINES, filter={"applies_to": "All categories"})
        return docs

    def sources(self):
        """The documents behind the latest answer, numbered as in its citations."""
        return "\n".join(f"  [{i}] {doc.metadata['doc_id']}" for i, doc in enumerate(self.last_docs, 1))

    def recommend(self, product_name, category, cost_price, current_price=None, target_margin=None,
                  competitor_price=None, price_elasticity=None):
        """Retrieve the evidence for a product and ask the model for a cited price recommendation."""
        query = f"{product_name} ({category})" + (f", {price_elasticity} price elasticity" if price_elasticity else "")
        docs = self.retrieve_product_knowledge(query, category, price_elasticity)
        self.last_docs = docs

        context = format_context(docs)
        if not any(doc.metadata["category"] == category for doc in docs):
            context = (f"(The knowledge base has no benchmarks, margins, competitor trends or past decisions "
                       f"for {category}. Only general guidelines were found.)\n\n" + context)

        prompt = f"""Recommend a selling price for this product.

PRODUCT
{describe_product(product_name, category, cost_price, current_price, target_margin, competitor_price, price_elasticity)}

KNOWLEDGE BASE CONTEXT (retrieved for this product)
{context}

Answer in these sections, citing the context like [3] for every fact you take from it:
{ANSWER_SECTIONS}"""
        return answer_text(self.llm.invoke([SystemMessage(content=RAG_SYSTEM_PROMPT), HumanMessage(content=prompt)]))


def prompt_only_recommendation(llm, product_name, category, cost_price, current_price=None, target_margin=None,
                               competitor_price=None, price_elasticity=None):
    """The baseline: same model, rules and answer sections, but no knowledge base, so any
    market fact in its answer comes from the model's own memory."""
    prompt = f"""Recommend a selling price for this product.

PRODUCT
{describe_product(product_name, category, cost_price, current_price, target_margin, competitor_price, price_elasticity)}

Answer in these sections:
{ANSWER_SECTIONS}"""
    return answer_text(llm.invoke([SystemMessage(content=BASELINE_SYSTEM_PROMPT), HumanMessage(content=prompt)]))


def compare(agent, **product):
    """Price one product with and without retrieval, and show which retrieved documents the RAG answer cites."""
    rag_answer = agent.recommend(**product)
    docs = agent.last_docs
    cited = cited_numbers(rag_answer, len(docs))
    sources = "\n".join(f"  [{i}]{'*' if i in cited else ' '} {doc.metadata['doc_id']}"
                        for i, doc in enumerate(docs, 1))
    baseline_answer = prompt_only_recommendation(agent.llm, **product)
    line = "-" * 70
    return f"""RAG AGENT ({len(docs)} documents retrieved, {len(cited)} cited)
{line}
{rag_answer}

Retrieved documents (* = cited in the answer):
{sources}

PROMPT-ONLY BASELINE (no knowledge base)
{line}
{baseline_answer}"""


def explore(store, query, k=3, filter=None):
    """Print what a similarity search returns, with distances (lower = closer). No API key needed."""
    print(f"Query: '{query}'" + (f"   filter: {filter}" if filter else ""))
    print("=" * 50)
    results = store.similarity_search_with_score(query, k=k, filter=filter)
    for i, (doc, distance) in enumerate(results, 1):
        print(f"{i}. [{doc.metadata['doc_type']} | {doc.metadata['category']}]  distance {distance:.2f}")
        print(f"   {doc.page_content}\n")
    return results


# --- Command line -------------------------------------------------------------------

def parse_args(argv=None):
    p = EXAMPLE_PRODUCT
    parser = argparse.ArgumentParser(
        description="Recommend a selling price with evidence from a pricing knowledge base (prices in ₹).")
    parser.add_argument("--product", default=p["product_name"], help="product name")
    parser.add_argument("--category", default=p["category"],
                        help="category, e.g. Electronics, Footwear, Apparel, Luxury Goods, Groceries, "
                             "Home & Kitchen (or Sports & Fitness with --add-new-category)")
    parser.add_argument("--cost", type=float, default=p["cost_price"], help="cost price in ₹")
    parser.add_argument("--current", type=float, help="current selling price in ₹")
    parser.add_argument("--margin", type=float, default=p["target_margin"],
                        help="target gross margin in percent, on selling price")
    parser.add_argument("--competitor", type=float, help="competitor price in ₹")
    parser.add_argument("--elasticity", type=str.capitalize, choices=ELASTICITY_LEVELS,
                        default=p["price_elasticity"], help="low, medium or high (case-insensitive)")
    parser.add_argument("--no-elasticity", action="store_true",
                        help="leave elasticity out, so the agent takes it from the category benchmark")
    parser.add_argument("--compare", action="store_true", help="also price the product with no knowledge base")
    parser.add_argument("--add-new-category", action="store_true",
                        help="add the Sports & Fitness data from new_category.json first")
    parser.add_argument("--explore", metavar="QUERY",
                        help="show what a similarity search returns for QUERY (no API key needed)")
    parser.add_argument("--filter-category", help="with --explore: only search this category")
    parser.add_argument("--k", type=int, default=5, help="with --explore: number of results")
    args = parser.parse_args(argv)

    if args.cost <= 0:
        parser.error("--cost must be positive")
    if not 0 <= args.margin < 100:
        parser.error("--margin must be at least 0 and below 100")
    return args


def main(argv=None):
    # Force UTF-8 stdout so ₹ and model output do not crash on Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)

    store = build_store(make_embeddings())
    if args.add_new_category:
        add_documents(store, create_documents(load_knowledge(NEW_CATEGORY_PATH)))

    if args.explore:
        explore(store, args.explore, args.k, {"category": args.filter_category} if args.filter_category else None)
        return

    product = {
        "product_name": args.product, "category": args.category, "cost_price": args.cost,
        "current_price": args.current, "target_margin": args.margin, "competitor_price": args.competitor,
        "price_elasticity": None if args.no_elasticity else args.elasticity,
    }
    agent = RAGPricingAgent(store)
    print(describe_product(**product) + "\n")
    if args.compare:
        print(compare(agent, **product))
        return
    answer = agent.recommend(**product)
    print("Knowledge retrieved:")
    print(agent.sources() + "\n")
    print(answer)


if __name__ == "__main__":
    main()

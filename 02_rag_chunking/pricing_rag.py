"""
Retrieval-augmented pricing agent (LangChain + Groq + Chroma).

Answers questions about a price list by retrieving the relevant passages and
letting the model answer from those passages alone. The price list is
pricing_guide.md, a fictional software catalogue: three products with several
plans each, plus bundles and billing policies.

The main decision in this stage is how to cut the document into chunks, so the
strategies are defined side by side and scored on the same questions:

- fixed-300           - fixed-size windows cut at line breaks.
- recursive-<size>    - recursive splitting at 100, 300 or 600 characters.
- markdown-default    - one chunk per section, headers moved into metadata.
- markdown-contextual - the same sections with the product and plan names
                        written back into the text. The agent's default.
- semantic            - boundaries chosen where embedding similarity drops.

Keyword search with HyDE (searching with a generated answer as well as the
question) is included for vague questions that share few words with the document.

How to run
----------
1. Copy `.env.example` at the repository root to `.env` and add your Groq API key.
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python 02_rag_chunking/pricing_rag.py

With no arguments it answers a set of sample questions. Run with --help to ask
your own question, change the strategy, score every strategy (no API key needed)
or try HyDE.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from rank_bm25 import BM25Okapi

# Load GROQ_API_KEY from the .env file at the repository root.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GUIDE_PATH = Path(__file__).with_name("pricing_guide.md")

DEFAULT_STRATEGY = "markdown-contextual"
# At k=4 the second Enterprise plan ranks just outside the results; see the walkthrough notebook.
DEFAULT_K = 5

PRICING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a pricing assistant for the DataFlow Platform.

Answer the customer's question using ONLY the context below.
- Be precise: quote exact prices and full plan names.
- Never estimate, round or invent a price, limit or feature.
- A rule stated for a group of plans, such as "Enterprise plans", applies to every plan in that group.
- If the context does not cover the question, reply exactly: "I don't have that information."

Context:
{context}"""),
    ("human", "{question}"),
])

HYDE_PROMPT = ChatPromptTemplate.from_template(
    """Write a short, realistic passage from a software pricing page that answers this customer need:

"{question}"

Describe one entry-level plan: its name, monthly price, user limit, storage, support level and a few basic features.
Plain text only, 4 to 6 lines, no tables or markdown."""
)

SAMPLE_QUESTIONS = [
    "What is included in the Analytics Pro plan?",
    "Which plan should I choose for a team of 10 that needs CRM?",
    "What is the annual discount and how does it apply?",
    "Which plans include HIPAA compliance?",
    "How much does the Enterprise plan cost?",  # ambiguous: Analytics and CRM each have one
    "Is there a student discount?",  # not in the guide, so the agent should refuse
    "How much do I save with the DataFlow Suite Bundle?",  # the guide's own claim is checkable
]

# (question, exact strings the context must contain to answer it, product the context must name)
FACT_CHECKS = [
    ("What is included in the Analytics Pro plan?",
     ["$99/month", "10,000 calls/month", "Predictive modeling", "Up to 50 per month"], "Analytics"),
    ("How much is the Storage Premium plan?", ["$79/month"], "Storage"),
    ("How many contacts does the CRM Growth plan include?", ["50,000 contacts"], "CRM"),
    ("Which plan should I choose for a team of 10 that needs CRM?", ["$49/user/month"], "CRM"),
    ("Does the Analytics Starter plan include API access?", ["API access: Not included"], "Analytics"),
    ("What uptime does the Storage Premium plan guarantee?", ["99.99% uptime SLA"], "Storage"),
    ("How much does the Security Add-on cost?", ["$19/month"], None),
    ("What does the DataFlow Suite Bundle cost and what does it contain?",
     ["$149/month", "Analytics Pro, CRM Growth, and Storage Standard"], None),
    ("How much is the training and onboarding package?", ["$499 one-time fee"], None),
    ("What is the annual discount?", ["20% discount"], None),
    ("What happens if I cancel an annual plan?", ["30-day notice", "14-day grace period"], None),
    ("How long is the free trial?", ["14-day free trial"], None),
    ("Which plans include HIPAA compliance?", ["HIPAA compliance is available on Enterprise plans only"], None),
]


def load_guide():
    """Return the price list as one string."""
    return GUIDE_PATH.read_text(encoding="utf-8")


def make_embeddings():
    """Create the local embedding model (384 dimensions; the first run downloads it)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})


def make_llm(temperature=0, max_tokens=1024):
    """Create the Groq chat model that answers from the retrieved context."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env at the repository root "
            "and add your Groq API key."
        )
    # gpt-oss reasons before it answers and those tokens count towards max_tokens.
    # Reading prices out of retrieved text needs little reasoning, so low effort
    # leaves headroom; max_retries rides out free-tier rate limits.
    return ChatGroq(
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort="low",
        max_retries=10,
    )


def split_fixed(text, size=300):
    """Fixed-size windows, cut at line breaks with no regard for the plans."""
    splitter = CharacterTextSplitter(chunk_size=size, chunk_overlap=size // 10, separator="\n")
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]


def split_recursive(text, size):
    """Recursive splitting: tries paragraphs first, then lines, then words."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size // 10)
    return splitter.create_documents([text])


def split_markdown(text):
    """One chunk per section. The headers, and so the prices, move into metadata."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Platform"), ("##", "Product"), ("###", "Plan")]
    )
    return splitter.split_text(text)


def add_context(doc):
    """Write the product and plan headers back into the text.

    The text is what gets embedded and what the model reads, so without this a
    chunk has no price and no product name. strip_headers=False is not enough:
    the ## product header appears only once per product.
    """
    return Document(
        page_content=f"{doc.metadata['Product']} - {doc.metadata['Plan']}\n{doc.page_content}",
        metadata=doc.metadata,
    )


def split_markdown_contextual(text):
    """One chunk per section, each one naming its product, plan and price."""
    return [add_context(doc) for doc in split_markdown(text)]


def split_semantic(text, embeddings):
    """Start a new chunk where the embedding similarity between sentences drops."""
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,
    )
    # strip() first: a trailing newline otherwise comes out as an empty final chunk.
    return chunker.create_documents([text.strip()])


STRATEGIES = {
    "fixed-300": lambda text, embeddings: split_fixed(text),
    "recursive-100": lambda text, embeddings: split_recursive(text, 100),
    "recursive-300": lambda text, embeddings: split_recursive(text, 300),
    "recursive-600": lambda text, embeddings: split_recursive(text, 600),
    "markdown-default": lambda text, embeddings: split_markdown(text),
    "markdown-contextual": lambda text, embeddings: split_markdown_contextual(text),
    "semantic": split_semantic,
}


def build_store(chunks, embeddings, name):
    """Embed chunks into an in-memory Chroma collection."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=name,
        # Fixed ids, so building the same collection twice replaces it instead of doubling it.
        ids=[str(i) for i in range(len(chunks))],
    )


def format_docs(docs):
    """Join retrieved chunks into the context block of the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(store, llm, k=DEFAULT_K, prompt=PRICING_PROMPT):
    """Retrieve the top k chunks, fill the prompt and call the model."""
    retriever = store.as_retriever(search_kwargs={"k": k})
    return {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm


def ask(chain, question):
    """Return the agent's answer, failing loudly if the model ran out of tokens."""
    response = chain.invoke(question)
    if response.response_metadata.get("finish_reason") == "length":
        raise RuntimeError(f"Answer cut off by max_tokens for question: {question}")
    return response.content


def score_strategy(chunks, embeddings, name, k=DEFAULT_K):
    """Run retrieval only for every fact check and count what comes back.

    Returns (facts found, facts found in a chunk that also names the right
    product, average characters of context per question).
    """
    store = build_store(chunks, embeddings, f"score-{name}")
    found = attributed = context_chars = 0
    for question, facts, product in FACT_CHECKS:
        texts = [doc.page_content for doc in store.similarity_search(question, k=k)]
        context_chars += sum(len(text) for text in texts)
        for fact in facts:
            found += any(fact in text for text in texts)
            attributed += any(fact in text and (product is None or product in text) for text in texts)
    store.delete_collection()
    return found, attributed, context_chars / len(FACT_CHECKS)


def compare_strategies(text, embeddings, k=DEFAULT_K):
    """Print one line per strategy, so they can be compared on the same questions."""
    total_facts = sum(len(facts) for _, facts, _ in FACT_CHECKS)
    print(f"{len(FACT_CHECKS)} questions, {total_facts} facts, top {k} chunks\n")
    print(f"{'Strategy':<22}{'Chunks':>7}{'Facts found':>13}{'Attributed':>12}{'Chars sent to LLM':>19}")
    for name, split in STRATEGIES.items():
        chunks = split(text, embeddings)
        found, attributed, context_chars = score_strategy(chunks, embeddings, name, k)
        print(f"{name:<22}{len(chunks):>7}{found:>13}{attributed:>12}{context_chars:>19.0f}")


def build_keyword_index(chunks):
    """Build a BM25 index and return it with the chunk texts it covers."""
    texts = [chunk.page_content for chunk in chunks]
    return BM25Okapi([text.lower().split() for text in texts]), texts


def keyword_search(index, query, top_n=3):
    """Return the best chunks for query as (score, text, matched words), best first."""
    bm25, texts = index
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    return [
        (float(scores[i]), texts[i], sorted(set(tokens) & set(texts[i].lower().split())))
        for i in np.argsort(scores)[::-1][:top_n]
    ]


def hypothetical_answer(question, llm):
    """Generate the passage HyDE searches with. It may invent details, so it is never shown as an answer."""
    return (HYDE_PROMPT | llm).invoke({"question": question}).content


def print_hits(hits):
    for rank, (score, text, matched) in enumerate(hits, 1):
        print(f"[{rank}] score {score:.2f}, matched {matched}")
        print(text + "\n")


def run_hyde(question, text, llm):
    """Show keyword search for a vague question without and with a hypothetical answer."""
    index = build_keyword_index(split_recursive(text, 250))
    print("--- Keyword search alone ---\n")
    print_hits(keyword_search(index, question))
    passage = hypothetical_answer(question, llm)
    print(f"--- Hypothetical answer ---\n\n{passage}\n")
    print("--- Keyword search with the hypothetical answer ---\n")
    print_hits(keyword_search(index, f"{question} {passage}"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Answer questions about a price list with a RAG pricing agent.")
    parser.add_argument("--question", help="question to ask; leave out to run the sample questions")
    parser.add_argument("--strategy", choices=tuple(STRATEGIES), default=DEFAULT_STRATEGY, help="chunking strategy")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="number of chunks to retrieve")
    parser.add_argument("--compare", action="store_true",
                        help="score every strategy on retrieval only (no API key needed)")
    parser.add_argument("--hyde", metavar="QUESTION", help="compare keyword search with and without HyDE")
    args = parser.parse_args(argv)

    if args.k < 1:
        parser.error("--k must be at least 1")
    return args


def main(argv=None):
    # Force UTF-8 stdout so model output with special characters does not crash on Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    text = load_guide()

    if args.hyde:
        run_hyde(args.hyde, text, make_llm())
        return

    embeddings = make_embeddings()
    if args.compare:
        compare_strategies(text, embeddings, args.k)
        return

    llm = make_llm()
    store = build_store(STRATEGIES[args.strategy](text, embeddings), embeddings, args.strategy)
    chain = build_rag_chain(store, llm, args.k)
    print(f"Strategy: {args.strategy}, top {args.k} chunks\n")
    for question in [args.question] if args.question else SAMPLE_QUESTIONS:
        print(f"Q: {question}")
        print(f"A: {ask(chain, question)}\n")


if __name__ == "__main__":
    main()

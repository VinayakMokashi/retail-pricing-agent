"""
Validated pricing agent: the model chooses, the code calculates and checks (LangGraph + LangChain).

Stage 3 recommends a price from a pricing knowledge base, but the arithmetic and the pricing rules
are left to the model, and one answer priced a luxury handbag below the competitor while citing the
rule against it. Stage 4 showed that calculation tools make the arithmetic exact. Stage 5 showed that
a LangGraph loop can check an answer and send it back. This stage puts the three together:

1. Evidence comes from stage 3's knowledge base, through stage 3's own filtered retrieval. A model
   grades only the documents that were picked by similarity (past decisions, general guidelines);
   documents picked by their metadata (the category's benchmark, margins and trend, the guideline
   for the product's elasticity) are kept by code.
2. Code calculates everything numeric, with stage 4's formulas and the knowledge base's own rules:
   the margin floor, the price band the elasticity guideline allows around the competitor, the price
   points inside it (prices end in 49 or 99; luxury prices are round), and the margin of the chosen
   price.
3. The model chooses one price from that list and explains it, as structured output (a Pydantic
   object, not free text). It never states a margin: the code adds the numbers to the answer.
4. Code checks every hard rule, and a second model call checks that the explanation is supported by
   the evidence. A failed check sends the decision back with the reason, at most MAX_ATTEMPTS times.
   If the rules still fail, the agent falls back to the rule-based price and says so: it never
   returns a price that breaks a rule.
5. Every answer is scored as it is produced: the code checks are re-run on the final answer and an
   LLM judge rates the decision. Scores can be appended to a JSONL log to watch quality over time.

How to run
----------
1. Copy `.env.example` at the repository root to `.env` and add a Cerebras or Groq API key.
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python 06_validated_pricing_agent/validated_pricing_agent.py

With no arguments it prices the reference product (Puma sneakers: cost ₹1,800, 30% target margin,
competitor ₹2,699). Run with --help to price your own product, describe it in plain words (--text),
add the Sports & Fitness data, log the scores, or print the calculations alone (--tools-only, no API
key needed).
"""

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing_extensions import TypedDict

# Stage 3 provides the knowledge base, its documents and the filtered retrieval. Importing it also
# loads the .env file at the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "03_pricing_knowledge_rag"))
import pricing_knowledge_rag as kb  # noqa: E402

MAX_ATTEMPTS = 3

# Reference product used by the CLI defaults and the walkthrough notebook (stage 3's sneakers, with a
# competitor price so the elasticity guideline has something to work with).
EXAMPLE_PRODUCT = {
    "product_name": "Puma Sneakers",
    "category": "Footwear",
    "cost_price": 1800,
    "target_margin": 30,
    "competitor_price": 2699,
}

# The price band each elasticity guideline in the knowledge base allows, in % of the competitor's price.
GUIDELINE_BANDS = {
    "High": (-3.0, 3.0),   # "stay within 3% of the main competitor"
    "Medium": (-5.0, 5.0),  # "price within 5% of the competitor"
    "Low": (0.0, 10.0),    # "price at or up to 10% above the competitor"
}
NO_HISTORY_BAND = (-5.0, 5.0)  # categories with no history: "stay within 5% of the competitor"
BAND_WORDS = {
    "High": "within 3% of the competitor (High elasticity guideline)",
    "Medium": "within 5% of the competitor (Medium elasticity guideline)",
    "Low": "at or up to 10% above the competitor (Low elasticity guideline)",
    None: "within 5% of the competitor (guideline for categories with no history)",
}
LUXURY_CATEGORIES = {"Luxury Goods"}  # round prices; every other category ends in 49 or 99

# Guidelines the code enforces, so the explanation can always cite them.
RULE_GUIDELINES = ["guideline:Never price below the margin floor", "guideline:Psychological price points"]


# --- Model ---------------------------------------------------------------------------------

def provider():
    """Cerebras when its key is set (much higher rate limits), otherwise Groq's free tier."""
    if os.environ.get("CEREBRAS_API_KEY"):
        return "Cerebras", "gpt-oss-120b"
    return "Groq", "openai/gpt-oss-120b"


class UsageCounter(BaseCallbackHandler):
    """Counts every model call and the tokens it used."""

    def __init__(self):
        self.calls = 0
        self.tokens = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.calls += 1

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                self.tokens += usage.get("total_tokens", 0)


USAGE = UsageCounter()


def make_llm(max_tokens=2048, reasoning_effort="low", temperature=0.0):
    """gpt-oss-120b on Cerebras or Groq.

    gpt-oss reasons before it answers, and those hidden tokens count towards max_tokens, so the
    limit leaves room for both. max_retries rides out rate limits instead of failing the run.
    """
    name, model = provider()
    if name == "Groq" and not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("No API key found. Copy .env.example to .env at the repository root and add "
                           "CEREBRAS_API_KEY or GROQ_API_KEY.")
    settings = dict(temperature=temperature, max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                    max_retries=10, callbacks=[USAGE])
    if name == "Cerebras":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, base_url="https://api.cerebras.ai/v1",
                          api_key=os.environ["CEREBRAS_API_KEY"], **settings)
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, **settings)


# --- The request ---------------------------------------------------------------------------

def _as_percent(value):
    """Read 30 or 0.30 as 30% (stage 4's rule: a margin under 1% is unrealistic)."""
    value = float(value)
    return value * 100 if 0 < value < 1 else value


class PricingRequest(BaseModel):
    """A product to price. Prices are in rupees; the margin is a percent of the selling price."""
    product_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    cost_price: float = Field(gt=0)
    target_margin: float = Field(ge=0, lt=100)
    competitor_price: Optional[float] = Field(default=None, gt=0)
    current_price: Optional[float] = Field(default=None, gt=0)
    price_elasticity: Optional[Literal["Low", "Medium", "High"]] = None

    @field_validator("target_margin", mode="before")
    @classmethod
    def _margin(cls, value):
        return _as_percent(value) if value is not None else value

    @field_validator("price_elasticity", mode="before")
    @classmethod
    def _elasticity(cls, value):
        return value.strip().title() if isinstance(value, str) and value.strip() else None

    @field_validator("category", mode="before")
    @classmethod
    def _category(cls, value):
        # Match a known category case-insensitively ("footwear" -> "Footwear").
        known = {c.lower(): c for c in kb_categories(include_new_category=True)}
        return known.get(str(value).strip().lower(), str(value).strip())


class ExtractedRequest(BaseModel):
    """What the merchant's message states. Every field is optional: missing stays empty."""
    product_name: Optional[str] = Field(None, description="The product, as the merchant names it")
    category: Optional[str] = Field(None, description="The category from the list in the instructions")
    cost_price: Optional[float] = Field(None, description="What one unit costs the merchant, in rupees")
    target_margin: Optional[float] = Field(None, description="Target gross margin in percent, e.g. 30")
    competitor_price: Optional[float] = Field(None, description="The main competitor's price, in rupees")
    current_price: Optional[float] = Field(None, description="The merchant's current selling price, in rupees")
    price_elasticity: Optional[str] = Field(None, description="Low, Medium or High, only if the merchant says so")


REQUIRED_FIELDS = {"product_name": "the product", "category": "the category", "cost_price": "the cost price",
                   "target_margin": "the target margin"}
NUMBER_FIELDS = ["cost_price", "target_margin", "competitor_price", "current_price"]


def _stated(number, text):
    """True if the number appears in the text (ignoring thousands separators and a trailing .0)."""
    digits = re.sub(r"(?<=\d),(?=\d)", "", text)
    found = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", digits)}
    return any(math.isclose(number, n) or math.isclose(number, n * 100) for n in found)


def parse_request(text, llm=None, categories=None):
    """Turn a merchant's message into a PricingRequest, or list what is missing.

    The model copies values into a schema; code then checks that every number it returns really
    appears in the message, so a guessed number is treated as missing rather than used.
    Returns (request or None, list of problems).
    """
    categories = categories or kb_categories()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Copy the pricing details a merchant gives into the fields. Copy only values the "
                   "message states; leave a field empty when the message does not state it, and never "
                   "guess or calculate a value. Write the category as one of: {categories}; if none "
                   "fits, use the merchant's own word for it."),
        ("human", "{text}"),
    ])
    chain = prompt | (llm or make_llm()).with_structured_output(ExtractedRequest)
    extracted = chain.invoke({"text": text, "categories": ", ".join(categories)})
    values = extracted.model_dump()
    problems = []
    for name in NUMBER_FIELDS:
        if values[name] is not None and not _stated(values[name], text):
            problems.append(f"{name.replace('_', ' ')} {values[name]:g} is not in the message, so it was ignored")
            values[name] = None
    missing = [label for name, label in REQUIRED_FIELDS.items() if values[name] in (None, "")]
    if missing:
        return None, problems + [f"Please give {', '.join(missing)}."]
    try:
        return PricingRequest(**values), problems
    except ValidationError as error:
        return None, problems + [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()]


# --- Knowledge base (stage 3) ----------------------------------------------------------------

def kb_categories(include_new_category=False):
    knowledge = kb.load_knowledge()
    names = [b["category"] for b in knowledge["elasticity_benchmarks"]]
    if include_new_category:
        names += [b["category"] for b in kb.load_knowledge(kb.NEW_CATEGORY_PATH)["elasticity_benchmarks"]]
    return names


class KnowledgeBase:
    """Stage 3's knowledge base and filtered retrieval, plus the two guidelines the code enforces."""

    def __init__(self, embeddings=None, include_new_category=False, name="validated_pricing"):
        self.store = kb.build_store(embeddings or kb.make_embeddings(), name=name)
        if include_new_category:
            kb.add_documents(self.store, kb.create_documents(kb.load_knowledge(kb.NEW_CATEGORY_PATH)))
        # Stage 3's agent only for its retrieval method (its model is never called here).
        self._stage3 = kb.RAGPricingAgent(self.store, llm=object())

    def add_new_category(self):
        kb.add_documents(self.store, kb.create_documents(kb.load_knowledge(kb.NEW_CATEGORY_PATH)))

    def retrieve(self, request):
        query = f"{request.product_name} ({request.category})" + (
            f", {request.price_elasticity} price elasticity" if request.price_elasticity else "")
        docs = self._stage3.retrieve_product_knowledge(query, request.category, request.price_elasticity)
        seen = {d.metadata["doc_id"] for d in docs}
        docs += [d for d in self.store.get_by_ids(RULE_GUIDELINES) if d.metadata["doc_id"] not in seen]
        return docs


def needs_grading(doc, request):
    """Similarity picked these documents, so a model checks them. Metadata picked the rest."""
    meta = doc.metadata
    if meta["doc_type"] == "past_decision":
        return True
    return meta["doc_type"] == "guideline" and meta.get("applies_to") == "All categories" \
        and meta["doc_id"] not in RULE_GUIDELINES


class GradeDocument(BaseModel):
    """Binary score for whether a retrieved document helps price this product."""
    binary_score: str = Field(description="'yes' if the document helps price this product, otherwise 'no'")


def _yes_no(result):
    """Accept only a clear yes or no; anything else raises, so the call is retried."""
    score = (getattr(result, "binary_score", None) or "").strip().strip(".'\"").lower()
    if score not in ("yes", "no"):
        raise ValueError(f"expected yes or no, got {result!r}")
    result.binary_score = score
    return result


GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You check whether a document from a retail pricing knowledge base helps set the regular
selling price of a product. Answer 'yes' or 'no'.
- A past pricing decision helps if its product is comparable: a similar kind of product, price level or
  buyer. A decision for a very different product in the same category (a smartphone for a cable, a
  saree for socks) does not help.
- A general guideline helps if it applies to setting this product's regular price now. A guideline
  about a situation the merchant did not mention (a festive sale, passing on a cost increase) does not."""),
    ("human", "Product: {product}\n\nDocument: {document}"),
])


# --- Calculations (stage 4's tools, with the knowledge base's rules) ----------------------------

def margin_floor(cost_price, target_margin):
    """Stage 4's margin_calculator: the lowest price that gives the target margin on selling price."""
    return cost_price / (1 - target_margin / 100)


def margin_percent(price, cost_price):
    return (price - cost_price) / price * 100


def is_price_point(price, luxury=False):
    """Mass-market prices end in 49 or 99; luxury prices are round (multiples of ₹500, or ₹100 below ₹10,000)."""
    if abs(price - round(price)) > 1e-9:
        return False
    whole = int(round(price))
    if luxury:
        return whole % (500 if whole >= 10_000 else 100) == 0
    return whole % 100 in (49, 99)


def price_points(low, high, luxury=False):
    """Every price point from low to high, inclusive."""
    points = []
    step = 100
    start = int(math.floor(low / step)) * step
    for base in range(start, int(math.floor(high)) + step, step):
        candidates = [base] if luxury else [base + 49, base + 99]
        if luxury and base >= 10_000:
            candidates = [base] if base % 500 == 0 else []
        points += [p for p in candidates if low - 1e-9 <= p <= high + 1e-9]
    return points


def next_price_point(value, luxury=False):
    """The smallest price point at or above value (stage 3's rule: round up, never down)."""
    return price_points(value, value + 1_000, luxury)[0]


def demand_effect(current_price, new_price, coefficient_text):
    """Stage 4's elasticity_adjustment, with the category benchmark's coefficient range.

    Returns the price change and the expected change in units sold, as a (low, high) range.
    """
    numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", coefficient_text or "")]
    if not numbers or not current_price:
        return None
    coefficients = [-abs(n) for n in numbers]
    change = (new_price - current_price) / current_price * 100
    effects = sorted(c * change for c in coefficients)
    return {"price_change_percent": change, "units_change_percent": (effects[0], effects[-1]),
            "coefficient": coefficient_text}


@dataclass
class PriceOptions:
    """Everything the code works out before the model is asked anything."""
    floor: float
    luxury: bool
    elasticity: Optional[str]
    elasticity_source: str
    has_history: bool
    band: Optional[tuple] = None        # (low, high) in rupees, or None with no competitor price
    band_rule: str = ""
    candidates: List[int] = field(default_factory=list)
    conflict: bool = False              # no price point meets both the floor and the band
    default_price: int = 0              # the rule-based price: the lowest allowed point, or the floor's
    coefficient: Optional[str] = None   # the category benchmark's elasticity coefficient

    def describe(self, request):
        """The calculated facts, as lines for the prompts and the notebook."""
        money = kb.money
        lines = [f"- Margin floor: {money(request.cost_price)} / (1 - {request.target_margin:g}%) = "
                 f"{money(round(self.floor, 2))}; nothing below it."]
        lines.append(f"- Elasticity: {self.elasticity or 'unknown'} ({self.elasticity_source}).")
        if self.band:
            lines.append(f"- Competitor price {money(request.competitor_price)}; allowed band "
                         f"{money(round(self.band[0], 2))} to {money(round(self.band[1], 2))}: {self.band_rule}.")
        else:
            lines.append("- No competitor price was given, so there is no competitor band.")
        lines.append("- Price points: " + ("round prices (luxury goods)." if self.luxury else "prices end in 49 or 99."))
        if self.conflict:
            lines.append(f"- CONFLICT: no price point meets both the floor and the band. The only allowed price "
                         f"is {money(self.default_price)}, the first price point above the floor.")
        elif not self.band:
            lines.append(f"- Allowed prices: any price point from {money(self.candidates[0])} up; the first ten are "
                         + ", ".join(money(p) for p in self.candidates[:10]) + ".")
        else:
            shown = self.candidates if len(self.candidates) <= 12 else (
                self.candidates[:6] + ["..."] + self.candidates[-3:])
            lines.append("- Allowed prices: " + ", ".join(money(p) if p != "..." else p for p in shown)
                         + (f" ({len(self.candidates)} in all)" if len(self.candidates) > 12 else "") + ".")
        return "\n".join(lines)


def compute_options(request, docs):
    """Floor, band, allowed price points and the rule-based default, all in code."""
    has_history = any(d.metadata["category"] == request.category for d in docs)
    benchmark = next((d for d in docs if d.metadata["doc_type"] == "elasticity_benchmark"), None)
    if request.price_elasticity:
        elasticity, source = request.price_elasticity, "given by the merchant"
    elif benchmark is not None:
        elasticity, source = benchmark.metadata["elasticity"], "from the category benchmark"
    else:
        elasticity, source = None, "no benchmark for this category"
    luxury = request.category in LUXURY_CATEGORIES
    floor = margin_floor(request.cost_price, request.target_margin)
    coefficient = None
    if benchmark is not None:
        match = re.search(r"coefficient ([^)]+)\)", benchmark.page_content)
        coefficient = match.group(1) if match else None
    options = PriceOptions(floor=floor, luxury=luxury, elasticity=elasticity, elasticity_source=source,
                           has_history=has_history, coefficient=coefficient)

    if request.competitor_price:
        bands = []
        if elasticity:
            bands.append(GUIDELINE_BANDS[elasticity])
        if not has_history:
            bands.append(NO_HISTORY_BAND)
        low_pct, high_pct = max(b[0] for b in bands), min(b[1] for b in bands)
        comp = request.competitor_price
        options.band = (comp * (1 + low_pct / 100), comp * (1 + high_pct / 100))
        options.band_rule = " and ".join(BAND_WORDS[e] for e in ([elasticity] if elasticity else []) +
                                         ([None] if not has_history else []))
        options.candidates = price_points(max(floor, options.band[0]), options.band[1], luxury)
    else:
        start = next_price_point(floor, luxury)
        options.candidates = price_points(start, start * 1.25, luxury)  # no upper limit; these are examples

    if options.candidates:
        options.default_price = options.candidates[0]
    else:
        options.conflict = True
        options.default_price = next_price_point(floor, luxury)
        options.candidates = [options.default_price]
    return options


def check_price(request, options, price):
    """The hard rules, as code. Returns the list of rules the price breaks (empty if none)."""
    money = kb.money
    problems = []
    if price is None:
        return ["no price was given"]
    if price < options.floor - 0.005:
        problems.append(f"{money(price)} is below the margin floor of {money(round(options.floor, 2))}")
    if options.conflict:
        if price != options.default_price:
            problems.append(f"the floor and the competitor band conflict, so the price must be "
                            f"{money(options.default_price)}")
    elif options.band and not options.band[0] - 0.005 <= price <= options.band[1] + 0.005:
        problems.append(f"{money(price)} is outside the allowed band {money(round(options.band[0], 2))} to "
                        f"{money(round(options.band[1], 2))} ({options.band_rule})")
    if not is_price_point(price, options.luxury):
        problems.append(f"{money(price)} is not a price point: " +
                        ("luxury prices are round" if options.luxury else "prices end in 49 or 99"))
    return problems


# --- The decision (structured output) ----------------------------------------------------------

class EvidenceUse(BaseModel):
    doc: int = Field(description="The number of a document in the evidence list, e.g. 3 for [3]")
    point: str = Field(description="What that document says that supports the price, in a few words")


class PricingDecision(BaseModel):
    """One price chosen from the allowed list, and why."""
    recommended_price: float = Field(description="One price copied exactly from the list of allowed prices")
    evidence: List[EvidenceUse] = Field(description="The one to four documents the price is based on")
    rationale: str = Field(description="Two or three sentences on why this price, citing documents like [3]")
    trade_off: str = Field(description="When the floor and the competitor band conflict: the trade-off accepted "
                                       "and what to do about it. Otherwise an empty string.")
    risks: str = Field(description="What to watch after the change, in one sentence")


DECIDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are PriceWise, a retail pricing strategist. The code has already calculated the margin
floor, the price band the knowledge base's guidelines allow, and the list of allowed prices. Your job is
to choose ONE price from that list and explain the choice from the evidence.
- Choose where in the allowed list to price from the evidence: what worked and what failed in past
  decisions, the category benchmark and trend, and the guidelines.
- Do not calculate margins, percentages or differences: the code adds them to the answer.
- Every fact in the rationale must come from the numbered evidence and cite it like [3]. If the
  evidence has no data for the category, say so instead of borrowing figures from elsewhere.
- If the floor and the band conflict, use the one allowed price and explain the trade-off (the
  knowledge base says to keep the floor price and compete on value, or escalate to the category
  manager)."""),
    ("human", """PRODUCT
{product}

CALCULATED BY CODE
{options}

EVIDENCE
{evidence}{feedback}"""),
])

FEEDBACK_NOTE = """

YOUR PREVIOUS ANSWER WAS REJECTED
{reason}
Previous answer: {previous}
Write a new answer that fixes this."""


class GroundingVerdict(BaseModel):
    """Whether the explanation is supported by the evidence."""
    binary_score: str = Field(description="'yes' if every claim is supported, otherwise 'no'")
    unsupported: str = Field(description="If 'no': the first unsupported claim, quoted. Otherwise an empty string.")


GROUNDING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You check a pricing explanation against its evidence. Answer 'yes' if every factual claim in it
is supported, otherwise 'no'.
- Facts about the market, the category, past decisions and guidelines must appear in the numbered
  evidence, and a citation like [3] must point to a document that says what is claimed.
- The merchant's own figures and the facts calculated by code may also be used.
- Saying that the evidence has no data for something is supported.
- Opinions and recommendations ("pricing just below the competitor should help volume") are not claims
  of fact: judge only whether the facts behind them are in the evidence.
- Small wording differences are fine. Do not fail an explanation for leaving something out."""),
    ("human", """PRODUCT
{product}

CALCULATED BY CODE
{options}

EVIDENCE
{evidence}

EXPLANATION TO CHECK
{explanation}"""),
])


def explanation_text(decision):
    parts = [f"Price chosen: {kb.money(decision.recommended_price)}",
             "Evidence used: " + "; ".join(f"[{e.doc}] {e.point}" for e in decision.evidence),
             f"Rationale: {decision.rationale}"]
    if decision.trade_off.strip():
        parts.append(f"Trade-off: {decision.trade_off}")
    parts.append(f"Risks: {decision.risks}")
    return "\n".join(parts)


def check_decision(decision, request, options, n_docs):
    """Code checks on a decision: the price rules plus the structure of the explanation."""
    problems = check_price(request, options, decision.recommended_price)
    cited = cited_documents(decision)
    if not decision.evidence:
        problems.append("the answer cites no evidence")
    bad = sorted(n for n in cited if not 1 <= n <= n_docs)
    if bad:
        problems.append(f"it cites document(s) {bad}, but the evidence list has {n_docs}")
    if options.conflict and not decision.trade_off.strip():
        problems.append("the floor and the band conflict, but no trade-off is stated")
    return problems


# --- The graph ------------------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    rejections: list      # why each rejected decision was sent back
    request: PricingRequest
    docs: list            # everything retrieved
    evidence: list        # what the answer may cite (graded or picked by metadata)
    dropped: list         # doc_ids the grader rejected
    options: PriceOptions
    decision: Optional[PricingDecision]
    attempts: int
    problems: list        # rule problems found in the last decision
    grounded: Optional[bool]
    unsupported: str
    feedback: str
    status: str           # "ok", "low_confidence" or "rule_fallback"
    answer: str
    final_price: int


def _product_lines(request):
    return kb.describe_product(request.product_name, request.category, request.cost_price,
                               request.current_price, request.target_margin, request.competitor_price,
                               request.price_elasticity)


class ValidatedPricingAgent:
    """Stage 3's knowledge base + code calculations + structured decisions + checks + live scoring."""

    def __init__(self, knowledge=None, decide_llm=None, grader_llm=None, grounding_llm=None, judge_llm=None,
                 judge=True, log_path=None, verbose=True):
        self.knowledge = knowledge or KnowledgeBase()
        grader = grader_llm or make_llm(max_tokens=1024)
        self.grader = (GRADER_PROMPT | grader.with_structured_output(GradeDocument)
                       | RunnableLambda(_yes_no)).with_retry(stop_after_attempt=3)
        # Choosing and explaining needs more thought than a yes/no: medium reasoning effort, as in stage 5.
        self.decider = DECIDE_PROMPT | (decide_llm or make_llm(max_tokens=4096, reasoning_effort="medium")
                                        ).with_structured_output(PricingDecision)
        self.grounding = (GROUNDING_PROMPT | (grounding_llm or make_llm(max_tokens=4096, reasoning_effort="medium")
                                              ).with_structured_output(GroundingVerdict)
                          | RunnableLambda(_yes_no)).with_retry(stop_after_attempt=3)
        self.judge = Judge(judge_llm) if judge else None
        self.log_path = Path(log_path) if log_path else None
        self.verbose = verbose
        self.graph = self._build_graph()

    def _say(self, text):
        if self.verbose:
            print(text)

    # Nodes
    def _retrieve(self, state):
        self._say("---RETRIEVE (stage 3's filtered searches)---")
        return {"docs": self.knowledge.retrieve(state["request"])}

    def _grade(self, state):
        request, docs = state["request"], state["docs"]
        to_grade = [d for d in docs if needs_grading(d, request)]
        product = f"{request.product_name} ({request.category})"
        self._say(f"---GRADE EVIDENCE ({len(to_grade)} picked by similarity; "
                  f"{len(docs) - len(to_grade)} picked by metadata, kept)---")
        verdicts = self.grader.batch([{"product": product, "document": d.page_content} for d in to_grade],
                                     config={"max_concurrency": 4})
        rejected = {d.metadata["doc_id"] for d, v in zip(to_grade, verdicts) if v.binary_score == "no"}
        for d in to_grade:
            self._say(f"   {'drop' if d.metadata['doc_id'] in rejected else 'keep'}: {d.metadata['doc_id']}")
        return {"evidence": [d for d in docs if d.metadata["doc_id"] not in rejected], "dropped": sorted(rejected)}

    def _compute(self, state):
        options = compute_options(state["request"], state["evidence"])
        self._say("---CALCULATE (code)---\n" + options.describe(state["request"]))
        return {"options": options, "attempts": 0, "feedback": "", "rejections": []}

    def _decide(self, state):
        attempts = state.get("attempts", 0) + 1
        self._say(f"---DECIDE (model, structured output) attempt {attempts} of {MAX_ATTEMPTS}---")
        inputs = {"product": _product_lines(state["request"]),
                  "options": state["options"].describe(state["request"]),
                  "evidence": kb.format_context(state["evidence"]), "feedback": state.get("feedback", "")}
        try:
            decision = self.decider.invoke(inputs)
        except Exception as error:  # a reply that does not fit the schema counts as a failed attempt
            self._say(f"   reply did not fit the schema: {type(error).__name__}")
            return {"decision": None, "attempts": attempts,
                    "problems": ["the reply did not fit the required structure"]}
        self._say(f"   chose {kb.money(decision.recommended_price)}")
        return {"decision": decision, "attempts": attempts}

    def _validate(self, state):
        decision = state.get("decision")
        if decision is None:
            problems = state.get("problems") or ["no decision"]
        else:
            problems = check_decision(decision, state["request"], state["options"], len(state["evidence"]))
        self._say("---CHECK RULES (code)---\n" + ("   all rules pass" if not problems else
                                                  "\n".join(f"   FAIL: {p}" for p in problems)))
        feedback, rejections = "", state.get("rejections", [])
        if problems:
            previous = decision.model_dump_json() if decision else "(none)"
            feedback = FEEDBACK_NOTE.format(reason="It broke these rules: " + "; ".join(problems) + ".",
                                            previous=previous)
            rejections = rejections + ["rules: " + "; ".join(problems)]
        return {"problems": problems, "feedback": feedback, "rejections": rejections}

    def _check_grounding(self, state):
        self._say("---CHECK THE EXPLANATION AGAINST THE EVIDENCE (model)---")
        verdict = self.grounding.invoke({
            "product": _product_lines(state["request"]), "options": state["options"].describe(state["request"]),
            "evidence": kb.format_context(state["evidence"]), "explanation": explanation_text(state["decision"])})
        grounded = verdict.binary_score == "yes"
        self._say("   supported" if grounded else f"   NOT supported: {verdict.unsupported}")
        feedback = "" if grounded else FEEDBACK_NOTE.format(
            reason=f"This claim is not supported by the evidence: {verdict.unsupported}",
            previous=state["decision"].model_dump_json())
        rejections = state.get("rejections", []) + ([] if grounded else [f"grounding: {verdict.unsupported}"])
        return {"grounded": grounded, "unsupported": "" if grounded else verdict.unsupported, "feedback": feedback,
                "rejections": rejections}

    def _finalize(self, state):
        request, options, decision = state["request"], state["options"], state.get("decision")
        if state.get("problems"):
            status, price = "rule_fallback", options.default_price
        elif not state.get("grounded"):
            status, price = "low_confidence", int(decision.recommended_price)
        else:
            status, price = "ok", int(decision.recommended_price)
        self._say(f"---FINALIZE: {status}---")
        return {"status": status, "final_price": price,
                "answer": render_answer(request, options, state["evidence"], decision, price, status,
                                        state.get("attempts", 0), state.get("unsupported", ""))}

    # Routing
    @staticmethod
    def _after_rules(state):
        if not state["problems"]:
            return "check_grounding"
        return "decide" if state["attempts"] < MAX_ATTEMPTS else "finalize"

    @staticmethod
    def _after_grounding(state):
        if state["grounded"]:
            return "finalize"
        return "decide" if state["attempts"] < MAX_ATTEMPTS else "finalize"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_evidence", self._grade)
        graph.add_node("calculate", self._compute)
        graph.add_node("decide", self._decide)
        graph.add_node("check_rules", self._validate)
        graph.add_node("check_grounding", self._check_grounding)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade_evidence")
        graph.add_edge("grade_evidence", "calculate")
        graph.add_edge("calculate", "decide")
        graph.add_edge("decide", "check_rules")
        graph.add_conditional_edges("check_rules", self._after_rules,
                                    {"check_grounding": "check_grounding", "decide": "decide", "finalize": "finalize"})
        graph.add_conditional_edges("check_grounding", self._after_grounding,
                                    {"finalize": "finalize", "decide": "decide"})
        graph.add_edge("finalize", END)
        return graph.compile()

    # Public API
    def price(self, request, run_label=None):
        """Price one product. Returns a PricingResult with the answer, its checks and its scores."""
        request = request if isinstance(request, PricingRequest) else PricingRequest(**request)
        calls, tokens, started = USAGE.calls, USAGE.tokens, time.time()
        path, state = [], {"request": request}
        for update in self.graph.stream({"request": request}, config={"recursion_limit": 30}):
            for node, value in update.items():
                path.append(node)
                state.update(value or {})
        result = PricingResult(request=request, state=state, path=path, calls=USAGE.calls - calls,
                               tokens=USAGE.tokens - tokens, latency_s=time.time() - started, label=run_label)
        result.code_checks = score_with_code(result)
        if self.judge:
            j_calls, j_tokens = USAGE.calls, USAGE.tokens
            result.judge = self.judge.rate(result)
            result.judge_calls, result.judge_tokens = USAGE.calls - j_calls, USAGE.tokens - j_tokens
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result.log_record(), ensure_ascii=False) + "\n")
        return result

    def price_from_text(self, text, llm=None):
        """Parse a merchant's message, then price it, or return what is missing."""
        request, problems = parse_request(text, llm=llm)
        for p in problems:
            self._say(f"   note: {p}")
        if request is None:
            return None, problems
        return self.price(request), problems


def render_answer(request, options, evidence, decision, price, status, attempts, unsupported=""):
    """The merchant-facing answer. Every number in it is calculated by code."""
    money = kb.money
    lines = []
    if status == "rule_fallback":
        lines.append(f"RULE-BASED PRICE: the model's choice broke a pricing rule {attempts} times, so this is "
                     "the lowest price the rules allow. Review it before use.")
    elif status == "low_confidence":
        lines.append(f"LOW CONFIDENCE: the explanation was not fully supported by the evidence after {attempts} "
                     f"attempts ({unsupported}). Check it before use.")
    margin = margin_percent(price, request.cost_price)
    lines.append(f"Recommended price: {money(price)}")
    lines.append(f"- Margin: {margin:.1f}% (target {request.target_margin:g}%, floor {money(round(options.floor, 2))})")
    if request.competitor_price:
        diff = (price - request.competitor_price) / request.competitor_price * 100
        where = "at" if abs(diff) < 0.05 else (f"{abs(diff):.1f}% below" if diff < 0 else f"{diff:.1f}% above")
        lines.append(f"- Competitor: {money(request.competitor_price)}; this price is {where} it")
        if options.band:
            lines.append(f"- Allowed: {money(round(options.band[0], 2))} to {money(round(options.band[1], 2))}, "
                         f"{options.band_rule}" + (" (no price point fits both this and the floor)"
                                                   if options.conflict else ""))
    effect = demand_effect(request.current_price, price, options.coefficient) if request.current_price else None
    if effect:
        low, high = effect["units_change_percent"]
        lines.append(f"- Against the current price {money(request.current_price)} ({effect['price_change_percent']:+.1f}%): "
                     f"units sold {low:+.1f}% to {high:+.1f}% (benchmark coefficient {effect['coefficient']})")
    if decision is not None and status != "rule_fallback":
        lines.append(f"Why: {decision.rationale}")
        if decision.trade_off.strip():
            lines.append(f"Trade-off: {decision.trade_off}")
        lines.append(f"Risks: {decision.risks}")
        cited = sorted(n for n in cited_documents(decision) if 1 <= n <= len(evidence))
        lines.append("Sources: " + ", ".join(f"[{n}] {evidence[n - 1].metadata['doc_id']}" for n in cited))
    return "\n".join(lines)


def cited_documents(decision):
    """Every document number a decision cites, in its evidence list or in its text ([3] or 【3】)."""
    text = " ".join([decision.rationale, decision.trade_off, decision.risks])
    return {e.doc for e in decision.evidence} | {int(n) for n in re.findall(r"[\[【](\d+)[\]】]", text)}


# --- Live evaluation ------------------------------------------------------------------------------

def score_with_code(result):
    """Re-run the hard checks on the final answer, independently of the loop that produced it."""
    request, options, price = result.request, result.state["options"], result.final_price
    checks = {
        "above_floor": price >= options.floor - 0.005,
        "in_band": (options.band is None or options.conflict
                    or options.band[0] - 0.005 <= price <= options.band[1] + 0.005),
        "price_point": is_price_point(price, options.luxury),
        "margin_in_answer_correct": f"{margin_percent(price, request.cost_price):.1f}%" in result.answer,
    }
    decision = result.state.get("decision")
    n_docs = len(result.state["evidence"])
    if decision is not None and result.status != "rule_fallback":
        cited = {e.doc for e in decision.evidence}
        checks["citations_valid"] = bool(cited) and all(1 <= n <= n_docs for n in cited)
    return checks


class JudgeVerdict(BaseModel):
    follows_evidence: str = Field(description="'pass' or 'fail': the price follows what the evidence suggests")
    follows_evidence_reason: str = Field(description="One sentence")
    complete: str = Field(description="'pass' or 'fail': it says why this price and names a risk")
    complete_reason: str = Field(description="One sentence")
    clear: str = Field(description="'pass' or 'fail': one price, plain language, no contradictions")
    clear_reason: str = Field(description="One sentence")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You review a retail pricing recommendation, as a senior pricing manager would. The numbers in it
were calculated by code and are correct; judge the decision and the explanation. For each criterion
answer 'pass' or 'fail' with one sentence.
- follows_evidence: given the evidence, is this a sensible place to price within the allowed range?
  Does it repeat what worked in similar past decisions and avoid what failed, and apply the guideline
  for the product's elasticity? Without category data, does it say so and stay cautious?
- complete: does it explain why this price was chosen and name at least one risk to watch?
- clear: is there exactly one recommended price, in plain language, with no contradictions?"""),
    ("human", """PRODUCT
{product}

CALCULATED BY CODE
{options}

EVIDENCE
{evidence}

RECOMMENDATION
{answer}"""),
])


class Judge:
    """Stage 4's LLM judge, adapted: it rates what code cannot check. Not a gate, a monitor."""

    def __init__(self, llm=None):
        self.chain = (JUDGE_PROMPT | (llm or make_llm(max_tokens=2048)).with_structured_output(JudgeVerdict)
                      ).with_retry(stop_after_attempt=3)

    def rate(self, result):
        verdict = self.chain.invoke({
            "product": _product_lines(result.request),
            "options": result.state["options"].describe(result.request),
            "evidence": kb.format_context(result.state["evidence"]), "answer": result.answer})
        out = {}
        for name in ("follows_evidence", "complete", "clear"):
            out[name] = getattr(verdict, name).strip().lower().startswith("pass")
            out[f"{name}_reason"] = getattr(verdict, f"{name}_reason")
        return out


@dataclass
class PricingResult:
    request: PricingRequest
    state: dict
    path: list
    calls: int
    tokens: int
    latency_s: float
    label: Optional[str] = None
    code_checks: dict = field(default_factory=dict)
    judge: Optional[dict] = None
    judge_calls: int = 0
    judge_tokens: int = 0

    @property
    def answer(self):
        return self.state["answer"]

    @property
    def final_price(self):
        return self.state["final_price"]

    @property
    def status(self):
        return self.state["status"]

    @property
    def attempts(self):
        return self.state.get("attempts", 0)

    @property
    def rejections(self):
        """Why each rejected decision was sent back ("rules: ..." or "grounding: ...")."""
        return self.state.get("rejections", [])

    def rejection_counts(self):
        counts = {}
        for reason in self.rejections:
            kind = reason.split(":")[0]
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    def summary(self):
        """One row for a results table."""
        request = self.request
        row = {
            "product": request.product_name, "price": self.final_price,
            "margin_%": round(margin_percent(self.final_price, request.cost_price), 1),
            "vs_competitor_%": (round((self.final_price - request.competitor_price) / request.competitor_price * 100, 1)
                                if request.competitor_price else None),
            "status": self.status, "attempts": self.attempts,
            "dropped": len(self.state.get("dropped", [])),
            "rejected_by": ", ".join(f"{kind} x{n}" for kind, n in self.rejection_counts().items()) or "-",
            "code_checks": f"{sum(self.code_checks.values())}/{len(self.code_checks)}",
        }
        if self.judge:
            row["judge"] = f"{sum(self.judge[k] for k in ('follows_evidence', 'complete', 'clear'))}/3"
        row.update({"calls": self.calls, "tokens": self.tokens, "seconds": round(self.latency_s, 1)})
        return row

    def log_record(self):
        return {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": self.label,
                "request": self.request.model_dump(), "price": self.final_price, "status": self.status,
                "attempts": self.attempts, "path": self.path, "dropped": self.state.get("dropped", []),
                "rejections": self.rejections,
                "code_checks": self.code_checks, "judge": self.judge, "calls": self.calls,
                "tokens": self.tokens, "latency_s": round(self.latency_s, 2), "answer": self.answer}


# --- Command line ---------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validated pricing agent: the model chooses, the code calculates and checks.")
    p.add_argument("--product", default=EXAMPLE_PRODUCT["product_name"])
    p.add_argument("--category", default=EXAMPLE_PRODUCT["category"])
    p.add_argument("--cost", type=float, default=EXAMPLE_PRODUCT["cost_price"], help="cost price in ₹")
    p.add_argument("--margin", type=float, default=EXAMPLE_PRODUCT["target_margin"], help="target margin in percent")
    p.add_argument("--competitor", type=float, default=EXAMPLE_PRODUCT["competitor_price"], help="competitor price in ₹")
    p.add_argument("--current", type=float, help="current selling price in ₹")
    p.add_argument("--elasticity", help="low, medium or high (default: from the category benchmark)")
    p.add_argument("--text", help="describe the product in plain words instead of the options above")
    p.add_argument("--add-new-category", action="store_true", help="add the Sports & Fitness data first")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge's scores")
    p.add_argument("--log", help="append the answer and its scores to this JSONL file")
    p.add_argument("--quiet", action="store_true", help="print only the answer and its scores")
    p.add_argument("--tools-only", action="store_true",
                   help="print the calculations for the product, without calling a model (no API key needed)")
    return p.parse_args(argv)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # ₹ on Windows consoles
    args = parse_args(argv)
    knowledge = KnowledgeBase(include_new_category=args.add_new_category)
    request = None
    if not args.text:
        request = PricingRequest(product_name=args.product, category=args.category, cost_price=args.cost,
                                 target_margin=args.margin, competitor_price=args.competitor,
                                 current_price=args.current, price_elasticity=args.elasticity)
    if args.tools_only:
        if request is None:
            sys.exit("--tools-only needs the product options, not --text")
        docs = knowledge.retrieve(request)
        print(f"{request.product_name} ({request.category})\n")
        print(compute_options(request, docs).describe(request))
        return
    name, model = provider()
    print(f"Model: {model} on {name}\n")
    agent = ValidatedPricingAgent(knowledge, judge=not args.no_judge, log_path=args.log, verbose=not args.quiet)
    if args.text:
        result, problems = agent.price_from_text(args.text)
        if result is None:
            print("\n".join(problems))
            return
    else:
        result = agent.price(request)
    print("\n" + result.answer)
    checks = ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in result.code_checks.items())
    print(f"\nCode checks: {checks}")
    if result.judge:
        print("Judge: " + ", ".join(f"{k} {'pass' if result.judge[k] else 'FAIL'}"
                                    for k in ("follows_evidence", "complete", "clear")))
    for reason in result.rejections:
        print(f"Sent back once ({reason})")
    print(f"Path: {' -> '.join(result.path)} | {result.calls} model calls (+{result.judge_calls} judge), "
          f"{result.tokens + result.judge_tokens:,} tokens, {result.latency_s:.1f} s")


if __name__ == "__main__":
    main()

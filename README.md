# Retail Pricing Agent

An LLM agent for pricing work, built with [LangChain](https://python.langchain.com/) and [Groq](https://groq.com/)-hosted `gpt-oss` models. It recommends a selling price for a retail product from five inputs — **cost price, current price, target margin, competitor price and price elasticity** — and answers questions from a **price list** using retrieval.

The project is built in stages, each in its own folder, so every stage can be run and compared with the ones before it. Stage 1 is a **prompt-only baseline**: everything the agent knows and does comes from its prompts, with no tools and no data lookup. Stage 2 adds **retrieval**: a RAG agent that reads a price list, with chunking strategies scored on the same questions.

| Stage | Folder | What it adds | Status |
| --- | --- | --- | --- |
| 1 | [`01_prompt_only/`](01_prompt_only/) | Prompt-only baseline: four prompting strategies compared on the same product | ✅ Done |
| 2 | [`02_rag_chunking/`](02_rag_chunking/) | Retrieval: a RAG agent that answers from a price list, with chunking strategies scored on the same questions | ✅ Done |
| 3+ | — | Structured, validated output; tool calls for exact arithmetic; retrieval of product and competitor data; evaluation across many products | Planned |

## Repository structure

```
retail-pricing-agent/
├── 01_prompt_only/
│   ├── pricing_agent.py            # prompts, PricingAgent class and command-line interface
│   └── prompt_only_pricing.ipynb   # walkthrough: runs and compares every strategy (with saved outputs)
├── 02_rag_chunking/
│   ├── pricing_guide.md            # the price list the agent reads (fictional)
│   ├── pricing_rag.py              # chunking strategies, RAG chain, scoring, HyDE and command-line interface
│   └── rag_chunking_comparison.ipynb  # walkthrough: every strategy, failure and fix (with saved outputs)
├── requirements.txt
├── .env.example                    # template for your Groq API key
└── LICENSE
```

## Stage 1 — prompt-only baseline

### The pricing problem

| Input | Example | Meaning |
| --- | --- | --- |
| Cost price | $400 | What the product costs us |
| Current price | $599 | Today's selling price |
| Target margin | 25% | Gross margin on selling price: `(price − cost) / price` |
| Competitor price | $579 | The market anchor |
| Price elasticity | Medium | How strongly demand reacts to price: Low / Medium / High |

The margin target sets a **price floor**: `minimum price = cost / (1 − target margin)`, which is $533.33 for the example. A good recommendation never goes below the floor and uses elasticity to decide how close to the competitor to price: at or above it when demand is insensitive (Low), close to it (Medium), at or just below it when customers are price-sensitive (High).

### Prompting strategies

All prompts live in [`pricing_agent.py`](01_prompt_only/pricing_agent.py), so the notebook and the command-line tool always run exactly the same text.

| Strategy | Prompt | What it tests |
| --- | --- | --- |
| `basic` | `BASIC_PROMPT` — the task, the five inputs and "give a price and a brief explanation" | What the model does with no guidance |
| `cot` | `COT_PROMPT` — a role, the margin formula, four ordered analysis steps (margin floor → competitive position → elasticity → recommendation) and an exact output format | Whether explicit reasoning steps make answers checkable |
| `few-shot` | `FEW_SHOT_PROMPT` — two worked pricing decisions, then the new case | Whether the model can infer a policy from examples alone |
| `agent` | `PricingAgent` — a system message with the persona and pricing rules, plus a step-by-step user prompt | Whether rules in a system message make behaviour consistent |
| `quick` | `PricingAgent.quick_price_check` — three inputs, **no** system message | How much of the agent's discipline comes from its system message |

### How the agent works

```
product inputs ─► AGENT_PROMPT (step-by-step user message) ─┐
                                                            ├─► ChatGroq (openai/gpt-oss-120b) ─► analysis + "Recommended Price: $…"
AGENT_SYSTEM_PROMPT (persona "PriceWise" + pricing rules) ──┘
```

- **`make_llm()`** creates the shared `ChatGroq` model: `openai/gpt-oss-120b`, temperature 0.2 for stable numeric reasoning, `max_tokens=4096` (gpt-oss spends hidden reasoning tokens before it answers, and Groq's default limit can cut answers off), and `max_retries` to ride out free-tier rate limits. It fails fast with a clear message if `GROQ_API_KEY` is missing.
- **`PricingAgent`** keeps the persona and rules in a `SystemMessage` — never price below the margin floor, margin is measured on selling price, the competitor price is the anchor, how each elasticity level should move the price, show the working, give exactly one price — and sends each product as a `HumanMessage` built from `AGENT_PROMPT`.
- **`recommend(strategy, product)`** runs any strategy on a product dictionary: the three prompt-only strategies are LangChain `prompt | llm` chains, and `agent` goes through `PricingAgent`.

### Results on the reference product

From the run saved in the [walkthrough notebook](01_prompt_only/prompt_only_pricing.ipynb), where each strategy was run twice on the example above and every margin was checked in Python:

| Strategy | Run 1 | Run 2 | Notes |
| --- | --- | --- | --- |
| Basic | $575 | $549 | Least stable: $26 apart on the same input |
| Chain-of-thought | $574 | $579 | Fully checkable working and the exact output line both times, yet it reported a 30.4% margin for $574 (really 30.3%) |
| Few-shot | $575 | $575 | Stable — copies the examples' "a few dollars under the competitor" pattern |
| Agent | $585 | $585 | Stable with correct arithmetic; read "stay close to the competitor" as a $6 premium |

On the hardest case — a competitor priced **below** the margin floor ($420 vs a $461.54 floor) — the agent held the price at $462 (35.1% margin) rather than matching the competitor. The notebook's observations cover the rest: why structure buys stability, why terms and rules need precise wording, and why prompt-only arithmetic needs checking.

## Stage 2 — retrieval from a price list

Stage 1 only knew the numbers typed into its prompt. Stage 2 gives the agent a document to read: it retrieves passages from a price list and answers from those passages alone. The agent itself is simple, so the stage focuses on the decision that matters most in RAG, **how to cut the document into chunks**.

### The price list

[`pricing_guide.md`](02_rag_chunking/pricing_guide.md) is a fictional software price list ("DataFlow Platform"): three products with 10 plans, plus bundles, add-ons and billing policies. It makes a good test because plans in different products share names and prices (there are two Enterprise plans, and $99 buys either Analytics Pro or a seat of CRM Scale), and every price appears only in a header. This stage answers questions about a price list; it is not yet connected to the selling-price recommender in stage 1.

### How the agent works

```
question ─► embed (all-MiniLM-L6-v2, local) ─► Chroma: top 5 chunks ─┐
                                                                     ├─► ChatGroq (openai/gpt-oss-20b) ─► answer
PRICING_PROMPT (answer ONLY from the context, or refuse) ────────────┘
```

- **`STRATEGIES`** holds every chunking strategy, so they all run through the same code.
- **`build_store()`** embeds chunks into an in-memory Chroma collection, so nothing is written to disk.
- **`build_rag_chain()`** connects retrieval, the prompt and the model, and **`ask()`** raises an error if an answer was cut off by `max_tokens` instead of returning it silently.
- **`score_strategy()`** measures retrieval with no model involved, so its results are identical on every run.
- **`keyword_search()`** and **`hypothetical_answer()`** provide BM25 search and HyDE for vague questions.

### Chunking strategies

| Strategy | How it cuts |
| --- | --- |
| `fixed-300` | Fixed-size windows, cut at line breaks |
| `recursive-100/300/600` | Recursive splitting: paragraphs first, then lines, then words |
| `markdown-default` | One chunk per section, headers moved into metadata |
| `markdown-contextual` | The same sections, with product and plan names written back into the text (the default) |
| `semantic` | New chunk where embedding similarity between sentences drops |

### Results

Every strategy was scored on 13 questions covering 18 facts, retrieving the top 4 chunks. **Facts found** counts exact strings that reached the context. **Attributed** counts only those in a chunk that also names the right product.

| Strategy | Chunks | Facts found | Attributed | Characters sent to the model |
| --- | ---: | ---: | ---: | ---: |
| `fixed-300` | 17 | 16 | 10 | 1,116 |
| `recursive-100` | 64 | 12 | 8 | 283 |
| `recursive-300` | 22 | 15 | 10 | 974 |
| `recursive-600` | 10 | 17 | 9 | 1,812 |
| `markdown-default` | 17 | 10 | 5 | 975 |
| **`markdown-contextual`** | **17** | **18** | **18** | **1,128** |
| `semantic` | 8 | 18 | 18 | 3,164 |

`semantic` matches the score only because one chunk holds 66% of the document: bullet lines have no sentence-ending punctuation, so most of the price list counts as one sentence. These numbers come from a small hand-written question set on one 4.5 KB document, and Attributed rewards exactly what `markdown-contextual` adds, so they show a direction and are not a benchmark.

The sample questions show what those differences mean in practice. Each answer below was checked against the guide:

| Question | Baseline (`recursive-300`, k=4) | Final agent (`markdown-contextual`, k=5) |
| --- | --- | --- |
| What is in the Analytics Pro plan? | ❌ Refused | ✅ All 10 features |
| Plan for a team of 10 needing CRM? | ✅ | ✅ |
| Annual discount? | ⚠️ 20%, but lists "Free Trials" as a plan | ✅ |
| Which plans include HIPAA? | ✅ But no product names | ✅ |
| Enterprise plan cost? | ⚠️ Both prices, no product names | ✅ Both, with products |
| Student discount? | ✅ Refused | ✅ Refused |
| Suite Bundle saving? | ⚠️ Repeats "35%" | ⚠️ Repeats "35%" |

### What the failures showed

- **The Pro plan refusal was a chunking failure.** The retrieved chunk says "Pro Plan" but never "Analytics", because that heading landed in a different chunk. It repeats on every run.
- **The markdown splitter's default loses every price.** It moves headers into metadata, and only the text is embedded and sent to the model, so none of its 17 chunks contains a price. `strip_headers=False` isn't enough either, because the product header appears only once per product.
- **Two answers lost a plan for different reasons.** The Enterprise question was a retrieval miss: the Analytics Enterprise plan ranked fifth, so `k` went from 4 to 5. HIPAA was the model's reading: both plans were in the context, but it named both only 3 of 4 times at `k=4` and at most 1 of 4 at `k=5`, until one prompt line said a rule for a group of plans applies to every plan in it (then 4 of 4 at both, on every run).
- **RAG repeats what the source says.** The guide claims the $149 bundle saves 35%, but its list prices come to $177 with one CRM seat, a saving of 15.8%. About 35% needs two seats. Catching that needs a calculation, not retrieval.

The [walkthrough notebook](02_rag_chunking/rag_chunking_comparison.ipynb) shows every step, plus semantic chunking and HyDE, with saved outputs.

## Tech stack

- Python 3.10+
- `langchain-core` + `langchain-groq` 1.x (prompt templates, messages and the Groq chat model)
- `python-dotenv` (loads your API key from a `.env` file)
- Groq-hosted models: `openai/gpt-oss-120b` (stage 1) and `openai/gpt-oss-20b` (stage 2, where the model reads prices out of retrieved text and needs far less reasoning)
- Stage 2 only: `langchain-text-splitters` and `langchain-experimental` (chunking), `langchain-chroma` (vector store), `langchain-huggingface` with `sentence-transformers` (local `all-MiniLM-L6-v2` embeddings), `rank-bm25` (keyword search)
- Jupyter (optional, to run the walkthrough notebooks)

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/VinayakMokashi/retail-pricing-agent.git
   cd retail-pricing-agent
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Stage 2 installs PyTorch through `sentence-transformers`, so this can take a few minutes. Its first run also downloads the embedding model (about 90 MB).

4. **Add your Groq API key.** Copy `.env.example` to `.env` at the repository root and paste in your key
   (get a free one at https://console.groq.com/keys):

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Both stages load `.env` from the repository root, so the key is found no matter which
   directory you launch from. `.env` is git-ignored and never committed.

## How to run

### Stage 1 command line

```bash
# Price the reference product with the agent
python 01_prompt_only/pricing_agent.py

# Run all four strategies on the same product, one after another
python 01_prompt_only/pricing_agent.py --strategy all

# Price your own product
python 01_prompt_only/pricing_agent.py --cost 300 --current 450 --margin 35 --competitor 420 --elasticity medium

# Quick check from three inputs (no system message)
python 01_prompt_only/pricing_agent.py --strategy quick --cost 250 --margin 20 --competitor 350
```

| Option | Default | Description |
| --- | --- | --- |
| `--cost` | 400 | Cost price |
| `--current` | 599 | Current selling price |
| `--margin` | 25 | Target gross margin in percent, measured on selling price (0 ≤ margin < 100) |
| `--competitor` | 579 | Competitor price |
| `--elasticity` | Medium | `low`, `medium` or `high` (case-insensitive) |
| `--strategy` | `agent` | `basic`, `cot`, `few-shot`, `agent`, `quick`, or `all` (runs the first four) |

Sample output for the reference product (`--strategy agent`), abridged:

```
Product: cost $400, current $599, target margin 25%, competitor $579, elasticity Medium

**1. Margin Floor**
...
**4. Recommendation**
- Set the selling price equal to the competitor’s price: **$579**.
...
**Recommended Price:** $579
**Expected Margin:** 30.9 %
**Reasoning:** The $579 price meets the 25 % target margin, aligns with the medium‑elasticity market anchor, and positions the product competitively without undercutting profitability.
```

Answers vary from run to run; the notebook shows how much.

### Stage 2 command line

```bash
# Answer the sample questions with the default strategy (markdown-contextual, top 5 chunks)
python 02_rag_chunking/pricing_rag.py

# Ask your own question
python 02_rag_chunking/pricing_rag.py --question "Which storage plan has the best uptime?"

# Try another chunking strategy
python 02_rag_chunking/pricing_rag.py --strategy recursive-300 --k 4

# Score every strategy on retrieval only (no API key needed)
python 02_rag_chunking/pricing_rag.py --compare

# Compare keyword search with and without HyDE
python 02_rag_chunking/pricing_rag.py --hyde "affordable option for a small startup with basic needs"
```

| Option | Default | Description |
| --- | --- | --- |
| `--question` | Sample questions | The question to ask |
| `--strategy` | `markdown-contextual` | Any strategy in the table above |
| `--k` | 5 | Number of chunks to retrieve |
| `--compare` | Off | Score every strategy on retrieval only, with no model calls |
| `--hyde` | Off | Takes a question and compares keyword search with and without HyDE |

Sample output for the ambiguous Enterprise question:

```
Strategy: markdown-contextual, top 5 chunks

Q: How much does the Enterprise plan cost?
A: The Enterprise plans are priced as follows:

- **DataFlow CRM – Enterprise Plan**: **$199 per user per month**
- **DataFlow Analytics – Enterprise Plan**: **Custom pricing, starting at $500 per month**
```

### Walkthrough notebooks

```bash
pip install jupyter
jupyter notebook 01_prompt_only/prompt_only_pricing.ipynb
jupyter notebook 02_rag_chunking/rag_chunking_comparison.ipynb
```

Both notebooks are saved with their outputs, so you can read the full comparisons on GitHub without running them. Running them makes 12 and 38 model calls.

### From Python

```python
# run from inside 01_prompt_only/
from pricing_agent import PricingAgent

agent = PricingAgent()
print(agent.get_price_recommendation(cost_price=400, current_price=599, target_margin=25,
                                     competitor_price=579, price_elasticity="Medium"))
```

## Limitations of the prompt-only baseline

- **No verification.** The margin floor and the one-price rule are enforced only by the prompt; nothing checks the model's arithmetic or its final price.
- **No tools.** Every calculation is done by the model itself, so small arithmetic slips are possible.
- **No data.** The competitor price and the elasticity are single numbers typed in; there is no sales history or demand model behind them.
- **Free-text output.** The recommended price has to be pattern-matched out of prose.

These limitations define the next stages.

## Limitations of the retrieval stage

- **Small evaluation.** 13 hand-written questions on one short document show a direction, not a benchmark.
- **It relies on clean headers.** `markdown-contextual` works because the document has them. A messy PDF would need a different approach.
- **Rank order is fragile.** The Enterprise failure was one rank away, and raising `k` is a blunt fix. Reranking, or using the `Plan` metadata to fetch every plan with a matching name, would be sturdier.
- **No arithmetic.** The agent repeats whatever the document claims, so checking a saving needs a calculation tool.
- **Not connected to stage 1.** The two stages run separately for now.

## License

Released under the [MIT License](LICENSE).

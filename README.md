# Retail Pricing Agent

An LLM agent that recommends a selling price for a retail product from five inputs — **cost price, current price, target margin, competitor price and price elasticity** — built with [LangChain](https://python.langchain.com/) and [Groq](https://groq.com/)-hosted `gpt-oss` models.

The project is built in stages, each in its own folder, so every stage can be run and compared with the ones before it. Stage 1 is a **prompt-only baseline**: everything the agent knows and does comes from its prompts, with no tools and no data lookup.

| Stage | Folder | What it adds | Status |
| --- | --- | --- | --- |
| 1 | [`01_prompt_only/`](01_prompt_only/) | Prompt-only baseline: four prompting strategies compared on the same product | ✅ Done |
| 2+ | — | Structured, validated output; tool calls for exact arithmetic; retrieval of product and competitor data; evaluation across many products | Planned |

## Repository structure

```
retail-pricing-agent/
├── 01_prompt_only/
│   ├── pricing_agent.py            # prompts, PricingAgent class and command-line interface
│   └── prompt_only_pricing.ipynb   # walkthrough: runs and compares every strategy (with saved outputs)
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

## Tech stack

- Python 3.10+
- `langchain-core` + `langchain-groq` 1.x (prompt templates, messages and the Groq chat model)
- `python-dotenv` (loads your API key from a `.env` file)
- Groq-hosted model: `openai/gpt-oss-120b`
- Jupyter (optional, to run the walkthrough notebook)

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

4. **Add your Groq API key.** Copy `.env.example` to `.env` at the repository root and paste in your key
   (get a free one at https://console.groq.com/keys):

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

   `pricing_agent.py` loads `.env` from the repository root, so the key is found no matter which
   directory you launch from. `.env` is git-ignored and never committed.

## How to run

### Command line

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

### Walkthrough notebook

```bash
pip install jupyter
jupyter notebook 01_prompt_only/prompt_only_pricing.ipynb
```

The notebook is saved with its outputs, so you can read the full comparison on GitHub without running it. Running it makes 12 model calls.

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

## License

Released under the [MIT License](LICENSE).

# Retail Pricing Agent

An LLM agent for pricing work, built with [LangChain](https://python.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/) and [Groq](https://groq.com/)-hosted `gpt-oss` models. It recommends a selling price for a retail product from five inputs — **cost price, current price, target margin, competitor price and price elasticity** — answers questions from a **price list** using retrieval, backs its recommendations with evidence from a **pricing knowledge base**, calls **pricing tools** for exact arithmetic, measured by an **evaluation framework**, **checks its own retrieval and answers** before replying, and finally **leaves every number and pricing rule to code**, so the model only chooses a price and explains it.

The project is built in stages, each in its own folder, so every stage can be run and compared with the ones before it. Stage 1 is a **prompt-only baseline**: everything the agent knows and does comes from its prompts, with no tools and no data lookup. Stage 2 adds **retrieval**: a RAG agent that reads a price list, with chunking strategies scored on the same questions. Stage 3 brings the two together: the recommender retrieves past pricing decisions, benchmarks and guidelines from a **knowledge base** and cites them. Stage 4 gives the agent **calculation tools** and measures three prompt versions with an **evaluation framework**: an LLM judge validated against hand-labelled answers, a price check in code, tool-use metrics, and cost. Stage 5 makes the agent **check its own work**: a LangGraph state machine grades every retrieved document, checks that the answer's numbers are grounded and that it answers the question, and rewrites the question or the answer when a check fails (Self-RAG). Stage 6 brings stages 3-5 together: code calculates the prices the knowledge base's rules allow, the **model chooses** one as **structured output**, code checks every hard rule, a model checks the explanation, and every answer is **scored as it is produced**.

| Stage | Folder | What it adds | Status |
| --- | --- | --- | --- |
| 1 | [`01_prompt_only/`](01_prompt_only/) | Prompt-only baseline: four prompting strategies compared on the same product | ✅ Done |
| 2 | [`02_rag_chunking/`](02_rag_chunking/) | Retrieval: a RAG agent that answers from a price list, with chunking strategies scored on the same questions | ✅ Done |
| 3 | [`03_pricing_knowledge_rag/`](03_pricing_knowledge_rag/) | Grounded recommendations: the recommender retrieves past decisions, benchmarks and guidelines from a knowledge base and cites them, compared with a prompt-only baseline | ✅ Done |
| 4 | [`04_tools_and_evaluation/`](04_tools_and_evaluation/) | Tools and evaluation: three pricing tools, three prompt versions, and a framework that scores answers, tool use and cost on 12 questions run 3 times each | ✅ Done |
| 5 | [`05_self_correcting_rag/`](05_self_correcting_rag/) | Self-correction: a LangGraph agent that grades its retrieval, checks its answers for unsupported numbers and usefulness, rewrites the question or the answer when a check fails, and is compared with plain RAG on cost and quality | ✅ Done |
| 6 | [`06_validated_pricing_agent/`](06_validated_pricing_agent/) | Validation: stage 3's knowledge base, stage 4's calculations and stage 5's loop in one agent; the model chooses a price from a list the code calculates, as structured output, code enforces every pricing rule, and every answer is scored live | ✅ Done |
| 7+ | — | Real sales data: elasticity measured per product rather than per category, and price changes checked against what happened | Planned |

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
├── 03_pricing_knowledge_rag/
│   ├── knowledge_base.json         # benchmarks, margins, past decisions, competitor trends, guidelines (fictional, ₹)
│   ├── new_category.json           # Sports & Fitness data, added in the new-category test
│   ├── pricing_knowledge_rag.py    # documents, filtered retrieval, RAG agent, prompt-only baseline and CLI
│   └── knowledge_rag_pricing.ipynb # walkthrough: retrieval design, comparisons and tests (with saved outputs)
├── 04_tools_and_evaluation/
│   └── pricing_agent_evaluation.ipynb  # tools, three agent versions, judge, metrics, charts, cost (with saved outputs)
├── 05_self_correcting_rag/
│   └── self_correcting_pricing_agent.ipynb  # graders, query rewriter, LangGraph loop, retry limit, comparison with plain RAG (with saved outputs)
├── 06_validated_pricing_agent/
│   ├── validated_pricing_agent.py      # calculations, structured decision, rule checks, LangGraph loop, live scoring and CLI
│   └── validated_pricing_agent.ipynb   # walkthrough: test set, failure tests, comparison with stage 3, live evaluation (with saved outputs)
├── requirements.txt
├── .env.example                    # template for your API key(s)
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

## Stage 3 — recommendations grounded in a pricing knowledge base

Stage 1 priced a product from its five inputs and the model's own judgement, so every market fact in its answers came from the model's memory. Stage 3 gives the recommender a knowledge base of the company's pricing history to retrieve from, and requires it to cite every fact it uses.

### The knowledge base

[`knowledge_base.json`](03_pricing_knowledge_rag/knowledge_base.json) is fictional Indian retail data in rupees (₹) for six categories: Electronics, Footwear, Apparel, Luxury Goods, Groceries, and Home & Kitchen.

| Kind | Records | Example |
| --- | ---: | --- |
| Elasticity benchmarks | 6 | Footwear: High; a 5% price rise cuts branded-sneaker sales by 7-10% |
| Historical margins | 6 | Footwear averages 36%; above a 40% margin, branded sneakers sold under 60% of stock at full price |
| Past pricing decisions | 26 | Puma Smash at ₹2,399, 4% below the competitor, sold 92% of its stock at full price |
| Competitor trends | 6 | Entry-level branded sneakers cluster in the ₹2,199-₹2,999 band |
| Pricing guidelines | 8 | High elasticity: move in steps of 3-5% and stay within 3% of the main competitor |

The past decisions include failures on purpose, such as running shoes priced 14% above a competitor that sold 28% below plan, so the agent can learn what to avoid. [`new_category.json`](03_pricing_knowledge_rag/new_category.json) adds a seventh category, Sports & Fitness, with 5 more decisions, to test what happens when new knowledge arrives.

### How the agent works

```
product inputs ─► one filtered search per kind of evidence (Chroma, local all-MiniLM-L6-v2 embeddings)
                    ├─ the category's benchmark, margins and competitor trend (1 each)
                    ├─ the category's 4 most similar past decisions
                    └─ the guideline for the product's elasticity + the 2 closest general guidelines
                                         │  10 numbered documents
                                         ▼
RAG_SYSTEM_PROMPT (persona, cite like [3], learn from outcomes, shared pricing rules)
                                         ▼
                    ChatGroq (openai/gpt-oss-120b) ─► margin floor, evidence, recommendation, risks
```

- **One record, one document.** Records are short and self-contained, so nothing is chunked. Each carries metadata: `doc_type`, `category`, `applies_to` (for guidelines) and a stable `doc_id`, so adding a record twice replaces it.
- **Filtered searches instead of one similarity search.** A single search mixed in other categories, and for a high-elasticity query it ranked the *Low* elasticity guideline almost level with the *High* one, because the two share most of their words. Filtering on metadata guarantees every kind of evidence and picks the elasticity guideline by exact match.
- **Unknown categories** get the guideline for categories with no history, and the model is told that no category data exists.
- **A fair baseline.** `prompt_only_recommendation()` uses the same model, rules and answer sections with no knowledge base, so a comparison isolates what retrieval adds.
- **`cited_numbers()`** reads the citations back out of an answer (gpt-oss often writes `【2】` instead of `[2]`), so you can see which retrieved records the answer used.

### Results

From the run saved in the [walkthrough notebook](03_pricing_knowledge_rag/knowledge_rag_pricing.ipynb):

| Product | Inputs | Recommendation | What it rested on |
| --- | --- | --- | --- |
| Puma sneakers, RAG | cost ₹1,800, 30% margin, High | ₹2,599 (30.7%) | The sneaker price band [3] and the Puma Smash outcome [4]; 5 of 10 documents cited |
| Puma sneakers, prompt only | same | ₹2,600 (30.8%) | An **invented** "usual" market range of ₹2,500-₹3,000, and a wrong claim that ₹2,600 is the lowest price meeting the margin |
| Headphones, Electronics | cost ₹22,000, 18%, competitor ₹26,990, High | ₹26,899, ₹91 below the competitor | Headphones that lost sales at ₹500 above a competitor, earbuds that sold well at ₹100 below |
| Handbag, Luxury Goods | cost ₹16,000, 55%, competitor ₹38,500, Low | ₹36,000, 6.5% below the competitor | ⚠️ Contradicts the Low elasticity guideline it cites, which says to price at or above the competitor |
| Exercise bike, no Sports & Fitness data | cost ₹9,800, 30%, competitor ₹14,499 | ₹14,499, equal to the competitor | The guideline for categories with no history; it said no category data exists |
| Exercise bike, with the new data | same | ₹14,099, 2.8% below the competitor | Big-ticket equipment behaves like High elasticity [1]; a treadmill priced 5% above a competitor sold 30% below plan [7] |

### What the results showed

- **Retrieval replaced invented market facts with cited ones,** and citations make the rest checkable. The RAG answer still contained one competitor range that no record supports, and a later answer cited the wrong record for a number; both are quick to find because every claim is supposed to point at a record.
- **Citing a rule is not following it.** The luxury answer cited the right guideline, then priced below the competitor anyway. An earlier run with the same prompt chose ₹40,000, above it. A rule that must always hold has to be checked in code.
- **New knowledge changes answers immediately.** Adding 8 documents moved the exercise bike from a cautious copy of the competitor's price to an evidence-based price just below it, with no retraining.
- **A small embedding model misses obvious links.** The treadmill, the most relevant decision for an exercise bike, ranks only fourth of the five Sports & Fitness decisions, so the agent fetches 4 decisions, not 3.

## Stage 4 — pricing tools and an evaluation framework

Stages 1-3 left the arithmetic to the model and judged each stage on a handful of products. Stage 4 adds three **calculation tools** and an **evaluation framework**, and uses the framework to compare three versions of the agent that differ only in their system prompt.

### The tools

| Tool | What it computes |
| --- | --- |
| `margin_calculator(cost_price, target_margin)` | The price that gives a target margin on the selling price (accepts 25 or 0.25) |
| `competitor_matching(our_cost, competitor_price, positioning)` | Match, undercut (5% below) or premium (10% above) a competitor, never below a 10% margin floor, with a warning when the floor applies |
| `elasticity_adjustment(base_price, elasticity, price_change_percent)` | The effect of a price change on units sold and revenue (elastic −2, unit −1, inelastic −0.5), and whether to make it |

Bad input returns an error message instead of raising, so the agent can read it and the evaluation can count it as a failed call.

### The three agents

The same model (`gpt-oss-120b`), tools and settings; only the system prompt differs:

- **basic:** one line, "You are a pricing assistant. Answer the user's pricing question."
- **structured:** a retail pricing analyst with three steps (find the numbers, use the tools, give one price with a short reason).
- **comprehensive:** a senior strategist with the tool list, a four-step analysis, "never calculate prices yourself", and a fixed last line `Recommended price: $X.XX`.

### How it is evaluated

- **12 questions:** seven that need one tool, three that need several (one needs all three), and two edge cases: undercutting below cost, and a margin written as 0.35. Every expected price range was worked out by hand from the formulas. Every question runs **3 times per agent**: 108 runs in all.
- **Answer quality:** an LLM judge gives pass/fail verdicts on accuracy (final price in range), completeness (every key factor addressed) and clarity (one unambiguous price). Plain code also extracts the recommended price and checks it against the range.
- **The judge is tested before it is trusted:** on 7 hand-labelled answers, including deliberately wrong ones (markup instead of margin, falling for the elastic-demand trap), and then against the code check on every real answer.
- **Tool use:** precision, recall, success rate, and too few / optimal / too many calls.
- **Cost and speed:** real token counts, at illustrative rates, and time per question.

### Results

From the run saved in the [notebook](04_tools_and_evaluation/pricing_agent_evaluation.ipynb):

| | basic | structured | comprehensive |
| --- | --- | --- | --- |
| Accuracy (judge) | 0.972 | 0.972 | 1.000 |
| Completeness | 1.000 | 0.972 | 0.944 |
| Clarity | 0.833 | 0.972 | 1.000 |
| Tool precision / recall | 1.000 / 0.958 | 1.000 / 0.991 | 1.000 / 1.000 |
| **Overall score** | **0.957** | **0.984** | **0.991** |
| Cost per question (illustrative rates) | $0.0022 | $0.0023 | $0.0029 |

### What the results showed

- **The judge earned its place.** It matched all 21 hand labels (0 false passes) and the code's price check on 104 of 108 real answers. In all 4 disagreements the judge was right. The code had picked up the cost ($60 instead of $90.25), a rejected option ($55, $10.45), or an earlier $47.50 when the agent's final, wrong answer was $30.77. Text rules break easily on free-form answers.
- **The prompt mattered most for clarity and multi-tool questions.** The basic agent sometimes gave two prices, and on the hardest question it skipped a tool and once recommended a price that loses revenue. The comprehensive agent gave the same price on every run of every question.
- **Tool metrics can't see wrong numbers.** One structured run applied a 5% cut twice ($45.12 instead of $47.50). It called the right tools, so it scored 100% on tool use and failed only on accuracy.
- **The detailed prompt costs more:** about 29% more per question than basic, mostly because its long system prompt nearly doubles the input tokens.

The recommendation is the comprehensive prompt: the best score, the same price on every run, and a final line that code can read. Its scores are near the ceiling, though, and differences of one or two runs out of 36 are within noise.

## Stage 5 — a self-correcting agent (Self-RAG)

Stage 3 trusts whatever the retriever returns and never checks its own answer. Stage 5 turns retrieval and generation into a [LangGraph](https://langchain-ai.github.io/langgraph/) state machine that grades its own work and loops back when a check fails, following the [Self-RAG](https://arxiv.org/abs/2310.11511) pattern and LangGraph's [Self-RAG example](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb). The goal is to catch an unsupported number before it reaches the merchant.

### The knowledge base

Twenty short entries written for this stage (fictional, in $): elasticity benchmarks, historical margins, past pricing decisions and general guidelines for Electronics, Footwear, Apparel and Luxury goods. Grocery and SaaS pricing are deliberately missing, to test whether the agent admits a gap instead of borrowing another category's numbers. Each entry is one document (no chunking), embedded locally with `all-MiniLM-L6-v2` in Chroma.

### How the agent works

```
START → retrieve (top 4) → grade_documents ─┬─ none relevant ─► transform_query ─► retrieve
                                             └─ some relevant ─► generate ─► check the answer
                                                                   ├─ numbers not grounded ─► generate again
                                                                   ├─ grounded, not useful ─► transform_query
                                                                   └─ grounded and useful  ─► END
```

| Step | What it does |
| --- | --- |
| Relevance grader | One small yes/no call per retrieved document; off-topic documents are dropped before the answer is written |
| Generator | Uses only the retrieved context, gives one price with its margin and cites its evidence; with no data for the product's category it says so instead of recommending a margin |
| Groundedness grader | Is every number in the answer traceable to the documents? The merchant's question is passed only as the source of their own figures (cost, competitor price, target margin) |
| Usefulness grader | Does the answer give a price, or say clearly what data is missing? |
| Query rewriter | Rewrites vague questions ("my sneakers aren't selling…") in the knowledge base's vocabulary, keeping the merchant's numbers |
| Retry limit | One counter for rewrites and answers; after 3 attempts a failed check stops the run and the answer carries a LOW CONFIDENCE warning. A rewrite after a failed groundedness check is shown the rejected answer |

Every grader accepts only a clear yes or no and retries the call otherwise, and every run also passes LangGraph's `recursion_limit` as a backstop.

### Results

From the run saved in the [notebook](05_self_correcting_rag/self_correcting_pricing_agent.ipynb):

| Scenario | Path | Answer |
| --- | --- | --- |
| A: Puma sneakers, cost $1,800, competitor $2,799, 30% target | retrieve → grade → generate (7 calls) | $2,570, about 30%, 8% below the competitor, citing the Puma decision |
| B: "my sneakers aren't selling, should I drop the price?" | Two Apparel documents dropped; no rewrite needed (7 calls) | Names the missing numbers and gives the footwear rule: match for medium elasticity, about 7% below for high |
| C: organic grocery produce, cost $3 (not covered) | One answer rejected as not grounded, then rewritten (9 calls) | "The knowledge base does not contain data for this category", plus the margin floor rule with the guideline's 25% as an example |
| D: a phone case costing $4 | All four documents rejected → question rewritten as an Electronics accessory → retrieved again (12 calls) | Applies the Electronics benchmark: price at or just below the leading competitor |

Against a plain RAG pipeline (same retriever, prompt and model, no checks), over 12 runs each:

| | Plain RAG | Self-RAG |
| --- | --- | --- |
| LLM calls per question | 1 | 9.3 |
| Tokens per question | 1.1k | 7.7k |
| Time per question | 0.7 s | 5.2 s |
| Answers the groundedness grader judged hallucinated | 4/12 | 1/12 |

### What the results showed

- **The clearest win was a failed retrieval.** For the phone case, plain RAG got smartphone, headphone, Luxury and Footwear entries, then invented "30-40% for high-elasticity items" or borrowed the smartphone's 11% margin to suggest $4.49 for a $4 case. Self-RAG rejected those documents, rewrote the question and answered from the Electronics benchmark in all 3 runs.
- **LLM graders are noisy on subtle cases.** An earlier version of the agent answered the grocery question with "$4.00, a 25% margin", taking the 25% from a guideline's worked example, and every grader passed it. Even with that mistake spelled out in its prompt, the groundedness grader still passed it 3 times out of 3 in the saved run. Identical answers got different verdicts, so the 4/12 vs 1/12 above is indicative at best (and circular for Self-RAG, whose answers had already passed the same grader).
- **The fix that held was in the generator:** check for data on the product's own category before giving any number, at medium reasoning effort. At low effort the model skipped the check.
- **A plain "regenerate" can spin.** It resends the same prompt, and at temperature 0 the model writes almost the same answer; an earlier run rejected its grocery answer 8 times in a row (31 calls). The retry limit, and feedback on what was rejected, end that.
- **The cost is about 7-9× the calls and tokens of plain RAG.** That is worth it for a merchant-facing price, but not for low-stakes text.

## Stage 6 — a validated agent: the model chooses, the code calculates and checks

Each earlier stage left one problem open. Stage 3 grounded its prices in a knowledge base but left the arithmetic and the rules to the model, and one answer priced a luxury handbag below the competitor while citing the guideline against it. Stage 4's tools made the arithmetic exact, but they were never connected to the knowledge base, and their rules (a fixed 5% undercut, a 10% premium) were not the knowledge base's own. Stage 5 checked its answers in a loop, but every check was an LLM's judgement. Stage 6 combines them under one principle: **the model chooses, the code calculates and checks.**

### How the agent works

```
product details (fields, or a message parsed into fields and checked by code)
   │
retrieve ── stage 3's filtered searches (imported from stage 3), plus the two guidelines the code enforces
   │
grade evidence ── a model checks only what similarity picked (past decisions, general guidelines)
   │
calculate (code) ── margin floor · the band the elasticity guideline allows · the allowed price points
   │
decide (model) ── structured output: one price from the allowed list, with cited reasons
   │
check rules (code) ──── fail ──► decide again, told why (3 attempts; then the rule-based price)
   │ pass
check the explanation against the evidence (model) ── fail ──► decide again (3 attempts; then LOW CONFIDENCE)
   │ pass
answer ── every number written by code ──► live scores: code checks + an LLM judge, logged as JSONL
```

**The knowledge base's guidelines become code.** Stage 3's answers cited these rules; stage 6 enforces them:

| Guideline in the knowledge base | Rule in code |
| --- | --- |
| Never price below the margin floor | `price ≥ cost / (1 − target margin)` (stage 4's `margin_calculator`) |
| High elasticity: stay within 3% of the main competitor | band: competitor ± 3% |
| Medium elasticity: price within 5% of the competitor | band: competitor ± 5% |
| Low elasticity: price at or up to 10% above the competitor | band: competitor to competitor + 10% |
| Categories with no history: stay within 5% of the competitor | band: competitor ± 5% |
| Psychological price points: end in 49 or 99; round prices for luxury | allowed prices end in 49 or 99; luxury prices are multiples of ₹500 (₹100 below ₹10,000) |
| If the market price is below the floor, keep the floor price and compete on value, or escalate | the only allowed price is the first price point above the floor, and the answer must state the trade-off |

- **The elasticity** comes from the merchant or, if not given, from the category's benchmark (metadata, not the model). A category with no data gets the no-history band.
- **Structured output in, structured output out.** The request is a validated `PricingRequest`; a merchant's message can be turned into one by `parse_request()`, which then checks in code that every number it extracted appears in the message (a guessed number is dropped, and the agent asks for what is missing). The decision is a `PricingDecision`: a price, the evidence used (document numbers), a rationale, a trade-off and a risk. The model never writes a margin: the answer's numbers are rendered by code.
- **Fail closed.** If the model's price still breaks a rule after 3 attempts, the agent answers with the rule-based price (the lowest allowed one) and says so; it never returns a price that breaks a rule. If the explanation is still not supported after 3 attempts, the price stands but the answer carries a LOW CONFIDENCE warning.
- **Live evaluation.** Every answer is scored as it is produced: the code checks are re-run on the final answer (floor, band, price point, the margin in the text, valid citations), and stage 4's LLM judge, adapted, rates what code can't: does the price follow the evidence, is the explanation complete and clear. The judge is a monitor, not a gate. With `log_path` (or `--log`), each answer, its path, the reasons it was sent back and its scores are appended to a JSONL file.

### Results

From the run saved in the [notebook](06_validated_pricing_agent/validated_pricing_agent.ipynb): seven products in ₹, each priced twice.

| Product | What it tests | Price | Margin | Against the competitor |
| --- | --- | --- | --- | --- |
| Puma sneakers: cost ₹1,800, 30% target, competitor ₹2,699 | High elasticity; only three allowed prices | ₹2,649 (both runs) | 32.0% | 1.9% below |
| Headphones: cost ₹22,000, 18%, competitor ₹26,990, current price ₹27,990 | A floor just under the competitor | ₹26,899 (both) | 18.2% | 0.3% below; units sold +5.8% to +9.7% against the current price |
| Coach handbag: cost ₹16,000, 55%, competitor ₹38,500 | Low elasticity, round luxury prices | ₹39,000 (both) | 59.0% | 1.3% above |
| JBL speaker: cost ₹2,400, 25%, competitor ₹2,999 | A competitor below the margin floor | ₹3,249 (both), the only allowed price, with the trade-off stated | 26.1% | 8.3% above |
| Pressure cooker: cost ₹1,500, 38%, no competitor price | No band | ₹2,499 / ₹2,449 | 40.0% / 38.8% | — |
| Exercise bike: cost ₹9,800, 30%, competitor ₹14,499 | A category with no data | ₹14,049 (both), the first price point above the floor | 30.2% | 3.1% below |
| The same bike, with the Sports & Fitness data | New knowledge | ₹14,499 / ₹14,049 | 32.4% / 30.2% | at / 3.1% below |

| Live evaluation of the 14 answers | |
| --- | --- |
| Code checks passed (floor, band, price point, margin in the answer, citations) | 100% |
| Judge: follows the evidence / complete / clear | 86% / 100% / 100% |
| Answered on the first attempt | 12 of 14; 2 sent back by the grounding check |
| Rule-based fallbacks / LOW CONFIDENCE answers | 0 / 0 |
| Per answer | 6.3 model calls, about 7,000 tokens and 2.6 s, plus 1 judge call |

Stage 3's agent priced the same products (its own prompt and settings, the same model), and the same code checks were applied to its answers:

| | Stage 3 | Stage 6 |
| --- | --- | --- |
| Products with a pricing rule broken | **4 of 7** | **0 of 7** |
| What broke | Sneakers at ₹2,599, 3.7% below the competitor where the High elasticity guideline allows 3%, stated as a 30.8% margin that is 30.74%; ₹3,200 and ₹2,420, which are not price points; the bike at ₹14,899, stated as a 32.4% margin that is 34.22% | — |

### What the results showed

- **Choosing from a list leaves little room to break a rule.** The rule checks never had to send a decision back in the saved run. The fallback was exercised by a stand-in decider that always answers stage 3's below-competitor handbag price: after 3 rejections the agent answered ₹38,500, the lowest allowed price, labelled RULE-BASED PRICE.
- **The grounding check caught the model's own arithmetic.** Told not to calculate, the model still wrote that ₹2,649 is "2.5% below" the competitor (it is 1.9%), every time it priced the sneakers. The check sent it back, and the second attempt left the number out.
- **The judge found a data problem, not a model problem.** With the Sports & Fitness data, the band comes from the category benchmark's label (Medium, ±5%), but the same benchmark's notes say big-ticket equipment behaves like High elasticity (±3%). One run priced the bike 3.1% below the competitor: allowed by the code, outside the High guideline. The judge failed "follows the evidence" on both runs.
- **Stage 3's prices were mostly sensible,** but each rule held only when the model remembered it, and the arithmetic was right only when the model got it right. In stage 6, code does both.

## Tech stack

- Python 3.10+
- `langchain-core` + `langchain-groq` 1.x (prompt templates, messages and the Groq chat model)
- `python-dotenv` (loads your API key from a `.env` file)
- Groq-hosted models: `openai/gpt-oss-120b` (stages 1 and 3) and `openai/gpt-oss-20b` (stage 2, where the model reads prices out of retrieved text and needs far less reasoning)
- Stages 2 and 3: `langchain-chroma` (vector store) and `langchain-huggingface` with `sentence-transformers` (local `all-MiniLM-L6-v2` embeddings)
- Stage 2 only: `langchain-text-splitters` and `langchain-experimental` (chunking), `rank-bm25` (keyword search)
- Stage 3 walkthrough notebook: `pandas` (tables)
- Stage 4: `langchain-groq` or, with a Cerebras key, `langchain-openai` (Cerebras serves `gpt-oss-120b` through an OpenAI-compatible API); `pandas`, `numpy` and `matplotlib` for the tables and charts
- Stage 5: `langgraph` (the state machine), with the same vector store and embeddings as stages 2 and 3 and the same Cerebras/Groq switch as stage 4
- Stage 6: the same stack as stage 5, with `pydantic` models (installed with LangChain) for the request and the decision; it imports stage 3's module for the knowledge base and its retrieval
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

   Stages 2 and 3 install PyTorch through `sentence-transformers`, so this can take a few minutes. The first run also downloads the embedding model (about 90 MB).

4. **Add your Groq API key.** Copy `.env.example` to `.env` at the repository root and paste in your key
   (get a free one at https://console.groq.com/keys):

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Every stage loads `.env` from the repository root, so the key is found no matter which
   directory you launch from. `.env` is git-ignored and never committed.

   Stages 4-6 also accept an optional `CEREBRAS_API_KEY`. When it is set, they run on Cerebras,
   which serves the same model with much higher rate limits. Stage 6 needs only one of the two keys.

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

### Stage 3 command line

```bash
# Price the reference product (Puma sneakers) and list the documents retrieved
python 03_pricing_knowledge_rag/pricing_knowledge_rag.py

# Price your own product (prices in ₹)
python 03_pricing_knowledge_rag/pricing_knowledge_rag.py --product "Wireless Noise-Cancelling Headphones" --category Electronics --cost 22000 --current 27990 --margin 18 --competitor 26990 --elasticity high

# Compare with the prompt-only baseline on the same product
python 03_pricing_knowledge_rag/pricing_knowledge_rag.py --compare

# Add the Sports & Fitness data and let the agent find the elasticity itself
python 03_pricing_knowledge_rag/pricing_knowledge_rag.py --add-new-category --product "Magnetic Exercise Bike" --category "Sports & Fitness" --cost 9800 --margin 30 --competitor 14499 --no-elasticity

# See what a similarity search returns, with distances (no API key needed)
python 03_pricing_knowledge_rag/pricing_knowledge_rag.py --explore "high elasticity pricing strategy"
python 03_pricing_knowledge_rag/pricing_knowledge_rag.py --explore "Coach Leather Handbag" --filter-category "Luxury Goods"
```

| Option | Default | Description |
| --- | --- | --- |
| `--product` | Puma Sneakers | Product name |
| `--category` | Footwear | Category; an unknown category gets the no-history guideline |
| `--cost` | 1800 | Cost price in ₹ |
| `--current` | none | Current selling price in ₹ |
| `--margin` | 30 | Target gross margin in percent, on selling price (0 ≤ margin < 100) |
| `--competitor` | none | Competitor price in ₹ |
| `--elasticity` | High | `low`, `medium` or `high` (case-insensitive) |
| `--no-elasticity` | Off | Leave elasticity out, so the agent takes it from the category benchmark |
| `--compare` | Off | Also price the product with the prompt-only baseline |
| `--add-new-category` | Off | Add the Sports & Fitness data from `new_category.json` first |
| `--explore` | Off | Takes a query and prints the matching documents with distances, without calling the model |
| `--filter-category` | none | With `--explore`: search one category only |
| `--k` | 5 | With `--explore`: number of results |

Sample output for the reference product, abridged:

```
Knowledge retrieved:
  [1] benchmark:Footwear
  [2] margins:Footwear
  [3] trend:Footwear
  [4] decision:Puma Smash Sneakers
  ...
  [10] guideline:Passing on cost increases

**1. Margin floor**
Target margin = 30 % → Minimum price = Cost ÷ (1 – target) = 1,800 ÷ 0.70 = **₹2,571.43**.
...
**3. Recommendation**
Set the selling price at **₹2,599**.
```

### Stage 6 command line

```bash
# Price the reference product (Puma sneakers: cost ₹1,800, 30% target margin, competitor ₹2,699)
python 06_validated_pricing_agent/validated_pricing_agent.py

# Price your own product; --quiet prints only the answer and its scores
python 06_validated_pricing_agent/validated_pricing_agent.py --product "Coach Leather Handbag" --category "Luxury Goods" --cost 16000 --margin 55 --competitor 38500 --quiet

# Describe it in plain words; the agent asks for anything required that is missing
python 06_validated_pricing_agent/validated_pricing_agent.py --text "Puma sneakers, they cost me ₹1,800. Competitor sells at ₹2,699, I need a 30% margin."

# Only the calculations: floor, band and allowed prices (no API key needed)
python 06_validated_pricing_agent/validated_pricing_agent.py --tools-only --product "JBL Bluetooth Speaker" --category Electronics --cost 2400 --margin 25 --competitor 2999

# Append each answer, the reasons it was sent back and its scores to a JSONL log
python 06_validated_pricing_agent/validated_pricing_agent.py --log pricing_log.jsonl
```

| Option | Default | Description |
| --- | --- | --- |
| `--product` | Puma Sneakers | Product name |
| `--category` | Footwear | Category; one with no data in the knowledge base gets the no-history guideline |
| `--cost` | 1800 | Cost price in ₹ |
| `--margin` | 30 | Target gross margin in percent, on selling price (30 or 0.30) |
| `--competitor` | 2699 | Competitor price in ₹ |
| `--current` | none | Current selling price in ₹; adds the expected change in units sold |
| `--elasticity` | from the category benchmark | `low`, `medium` or `high` |
| `--text` | none | Describe the product in plain words instead of the options above |
| `--add-new-category` | Off | Add the Sports & Fitness data first |
| `--no-judge` | Off | Skip the LLM judge's scores (one model call fewer) |
| `--log` | none | Append the answer and its scores to this JSONL file |
| `--quiet` | Off | Print only the answer and its scores, not each step |
| `--tools-only` | Off | Print the calculations without calling a model |

Sample output for the reference product (`--quiet`):

```
Recommended price: ₹2,649
- Margin: 32.0% (target 30%, floor ₹2,571.43)
- Competitor: ₹2,699; this price is 1.9% below it
- Allowed: ₹2,618.03 to ₹2,779.97, within 3% of the competitor (High elasticity guideline)
Why: ₹2,649 falls inside the allowed band and is priced just below the competitor, satisfying the high‑elasticity rule to stay within 3% of the main competitor while keeping the margin floor intact. The choice mirrors a prior successful Puma decision that priced modestly below the rival and follows the psychological‑price‑point guideline for value perception.
Risks: Monitor competitor price moves and sales velocity to avoid a price war or inventory buildup.
Sources: [4] decision:Puma Smash Sneakers, [8] guideline:High elasticity: small price changes, [9] guideline:Psychological price points

Code checks: above_floor ok, in_band ok, price_point ok, margin_in_answer_correct ok, citations_valid ok
Judge: follows_evidence pass, complete pass, clear pass
Sent back once (grounding: "₹2,649 is 2.5% below the main competitor’s price")
Path: retrieve -> grade_evidence -> calculate -> decide -> check_rules -> check_grounding -> decide -> check_rules -> check_grounding -> finalize | 9 model calls (+1 judge), 13,757 tokens, 4.3 s
```

### Walkthrough notebooks

```bash
pip install jupyter
jupyter notebook 01_prompt_only/prompt_only_pricing.ipynb
jupyter notebook 02_rag_chunking/rag_chunking_comparison.ipynb
jupyter notebook 03_pricing_knowledge_rag/knowledge_rag_pricing.ipynb
jupyter notebook 04_tools_and_evaluation/pricing_agent_evaluation.ipynb
jupyter notebook 05_self_correcting_rag/self_correcting_pricing_agent.ipynb
jupyter notebook 06_validated_pricing_agent/validated_pricing_agent.ipynb
```

All six notebooks are saved with their outputs, so you can read the full comparisons on GitHub without running them. Running the first three makes 12, 38 and 6 model calls. The stage 4 evaluation makes 371 calls (about 300,000 tokens; roughly 3 minutes on Cerebras). On Groq's free tier, set `N_TRIALS = 1` in the notebook to stay within the daily token limit. The stage 5 notebook made 213 calls in its saved run (about 200,000 tokens; under 3 minutes on Cerebras); the count varies with how often the agent loops. On Groq's free tier, set `N_RUNS = 1` in its last section. The stage 6 notebook made 128 calls in its saved run (about 155,000 tokens; under 2 minutes on Cerebras); on Groq's free tier, set `N_RUNS = 1` in its section 3.

### From Python

```python
# run from inside 01_prompt_only/
from pricing_agent import PricingAgent

agent = PricingAgent()
print(agent.get_price_recommendation(cost_price=400, current_price=599, target_margin=25,
                                     competitor_price=579, price_elasticity="Medium"))
```

```python
# run from inside 03_pricing_knowledge_rag/
from pricing_knowledge_rag import RAGPricingAgent, build_store, make_embeddings

agent = RAGPricingAgent(build_store(make_embeddings()))
print(agent.recommend("Puma Sneakers", "Footwear", cost_price=1800, target_margin=30, price_elasticity="High"))
print(agent.sources())  # the documents behind the answer, numbered as in its citations
```

```python
# run from inside 06_validated_pricing_agent/
from validated_pricing_agent import KnowledgeBase, ValidatedPricingAgent

agent = ValidatedPricingAgent(KnowledgeBase(), verbose=False)
result = agent.price({"product_name": "Puma Sneakers", "category": "Footwear", "cost_price": 1800,
                      "target_margin": 30, "competitor_price": 2699})
print(result.answer)        # the merchant-facing answer; every number in it is calculated by code
print(result.code_checks)   # the rule checks, re-run on the final answer
print(result.judge)         # the LLM judge's scores
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
- **Not connected to stage 1.** Stage 2 answers questions about a price list; stage 3 is where retrieval meets the recommender.

## Limitations of the knowledge-base stage

- **Rules live in the prompt.** The floor, rounding and elasticity rules are instructions, not checks, so a run can break them, as the luxury answer did. Structured output with the price as a number, validated in code, would catch that (stage 6 does).
- **Arithmetic is still the model's.** Small slips, such as a 30.8% margin for 30.7%, are easy to spot but shouldn't happen. A calculation tool would remove them.
- **Citations are not verified.** An answer can cite the wrong record, or state a number no record contains; nothing checks this automatically yet.
- **Small, hand-written data.** 31 past decisions across 7 categories, and six test products run once each, show the approach, not a validated pricing model.

## Limitations of the evaluation stage

- **Small and tidy.** 12 questions run 3 times each; a difference of one or two runs is within noise. The expected prices follow this stage's own tool rules (5% undercut, 10% premium, 10% floor, fixed elasticity coefficients), not market data.
- **The judge is the same model as the agents.** That could favour its own style. The hand-labelled set and the code cross-check are what make its verdicts trustworthy here, and both are small.
- **The tools are not yet in the recommender.** Stage 4 evaluates a tool-using agent on its own questions; stage 6 wires the same calculations into the knowledge-base recommender.
- **Illustrative costs.** The token counts are real, but the rates and the per-tool-call fee are examples. The tools are local functions and cost nothing.

## Limitations of the self-correcting stage

- **The checks are LLM judgements.** They disagree with themselves on identical text, passed a number borrowed from a guideline's example, and once rejected a cautious answer while passing a bolder one. They are a safety net, not proof.
- **No exact arithmetic.** The graders don't catch small slips, such as $2,570 sold as a 30% margin when it gives 29.96%. Stage 4's tools and a code check of the price would.
- **The prompt does much of the work.** With a strict generator prompt, plain RAG already admitted the grocery gap; the clearest gain from the loop came from rejecting off-topic documents.
- **Small and separate.** Twenty hand-written entries and four test questions, with a knowledge base separate from stage 3's. Stage 6 wires these checks into the stage 3 recommender, with structured output and the stage 4 calculations.

## Limitations of the validated stage

- **The rules are only as good as their encoding.** The bands are this repository's reading of the guidelines (for example, Low elasticity as "at the competitor to 10% above"), and elasticity is set per category, while the bike case shows that part of a category can behave differently. Benchmarks per sub-category, or the merchant's own elasticity, would fix that case.
- **The model still calculates when told not to.** The grounding check catches it, at the cost of a retry. Listing each allowed price's margin and gap to the competitor in the prompt would remove the reason to calculate.
- **The model checks are still model checks.** The relevance grades, the grounding check and the judge are LLM judgements, now limited to what code can't decide. This stage did not re-validate the judge against hand-labelled answers, as stage 4 did.
- **Small and hand-written.** Seven products, each priced twice, on the stage 3 knowledge base; enough to show the design, not to measure it.

## License

Released under the [MIT License](LICENSE).

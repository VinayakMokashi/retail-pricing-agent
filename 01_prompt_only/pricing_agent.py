"""
Prompt-only retail pricing agent (LangChain + Groq).

Recommends a selling price for one product from five inputs - cost price,
current price, target margin, competitor price and price elasticity - using
nothing but the model's own reasoning: no tools, no data lookup, and no code
that checks or post-processes the numbers.

Four prompting strategies are defined here so they can be compared on the
same inputs:

- basic    - a one-paragraph instruction with the inputs filled in.
- cot      - chain-of-thought: a role, an explicit margin formula and four
             analysis steps (margin floor -> competitive position ->
             elasticity -> recommendation) ending in a fixed output format.
- few-shot - two worked pricing decisions, then the new case.
- agent    - the PricingAgent class: a system message that fixes the persona
             and the pricing rules, plus a step-by-step user prompt.

How to run
----------
1. Copy `.env.example` at the repository root to `.env` and add your Groq API key.
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python 01_prompt_only/pricing_agent.py

With no arguments it prices the reference product (cost 400, current price
599, target margin 25%, competitor 579, medium elasticity) with the agent.
Run with --help to set your own inputs or pick a strategy.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load GROQ_API_KEY from the .env file at the repository root.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = "openai/gpt-oss-120b"
ELASTICITY_LEVELS = ("Low", "Medium", "High")
INPUT_VARIABLES = ["cost_price", "current_price", "target_margin", "competitor_price", "price_elasticity"]

# Reference product used by the CLI defaults and the walkthrough notebook.
EXAMPLE_PRODUCT = {
    "cost_price": 400,
    "current_price": 599,
    "target_margin": 25,
    "competitor_price": 579,
    "price_elasticity": "Medium",
}

# --- Strategy 1: basic prompt -------------------------------------------------
BASIC_PROMPT = PromptTemplate(
    input_variables=INPUT_VARIABLES,
    template="""You are a retail pricing agent. Recommend an optimal selling price balancing margin and competition.

Inputs:
- Cost Price: ${cost_price}
- Current Price: ${current_price}
- Target Margin: {target_margin}%
- Competitor Price: ${competitor_price}
- Price Elasticity: {price_elasticity}

Give the recommended selling price and a brief explanation of your reasoning.""",
)

# --- Strategy 2: chain-of-thought prompt --------------------------------------
# Defines margin explicitly (on selling price, not markup on cost) and walks the
# model through the analysis in a fixed order before it commits to a price.
COT_PROMPT = PromptTemplate(
    input_variables=INPUT_VARIABLES,
    template="""ROLE: You are an experienced retail pricing analyst.
TASK: Recommend ONE optimal selling price that balances profit margin and competitiveness.

PRODUCT DATA:
- Cost Price: ${cost_price}
- Current Selling Price: ${current_price}
- Target Margin: {target_margin}% (gross margin on selling price = (price - cost) / price)
- Competitor Price: ${competitor_price}
- Price Elasticity of Demand: {price_elasticity} (Low = customers barely react to price changes, High = customers are very price-sensitive)

ANALYSIS STEPS (think step by step and show your work):
1. Calculate the minimum price based on target margin: minimum price = cost / (1 - target margin). Also compute the current margin.
2. Analyze competitive positioning: compare the current price and the minimum price with the competitor price (gap in $ and %).
3. Consider price elasticity impact: explain how sensitive demand is and whether we should price above, at, or below the competitor.
4. Recommend the optimal price: pick a single price that meets the target margin (never below the minimum price) and fits the competitive and elasticity analysis. State the resulting margin.

RULES:
- Do not use any external data; reason only from the inputs above.
- Give a single price, not a range.

OUTPUT FORMAT:
Show the 4 steps with short headings, then end with exactly these lines:
Recommended Price: $<number>
Expected Margin: <number>%
Summary: <one sentence>""",
)

# --- Strategy 3: few-shot prompt ----------------------------------------------
# No rules at all: the model has to infer the pricing policy from two examples.
FEW_SHOT_PROMPT = PromptTemplate(
    input_variables=INPUT_VARIABLES,
    template="""You are a pricing expert. Here are some examples of good pricing decisions:

Example 1:
Cost: $200, Current: $400, Target Margin: 30%, Competitor: $380, Elasticity: Medium
Recommendation: $375 (maintains margin above 30%, competitive with market, good for medium elasticity)

Example 2:
Cost: $100, Current: $180, Target Margin: 25%, Competitor: $200, Elasticity: Low
Recommendation: $190 (exceeds margin target, leverages low elasticity for higher profit)

Now analyze this case:
Cost: ${cost_price}, Current: ${current_price}, Target Margin: {target_margin}%, Competitor: ${competitor_price}, Elasticity: {price_elasticity}

Recommendation:""",
)

# --- Strategy 4: agent with a system persona ----------------------------------
AGENT_SYSTEM_PROMPT = """You are PriceWise, a senior retail pricing strategist.
You recommend optimal selling prices using ONLY the numbers the user gives you - no tools, no market data lookup.

Your principles:
- Never recommend a price below the minimum price needed to hit the target margin.
- Gross margin is measured on selling price: margin = (price - cost) / price, so minimum price = cost / (1 - target margin).
- Use the competitor price as the market anchor.
- Low elasticity: customers are not price-sensitive, so you may price at or above the competitor.
- Medium elasticity: stay close to the competitor price.
- High elasticity: customers are very price-sensitive, so price at or slightly below the competitor if the margin allows.
- Always show your calculations and reasoning step by step, clearly and concisely.
- Recommend exactly one price, not a range."""

AGENT_PROMPT = PromptTemplate(
    input_variables=INPUT_VARIABLES,
    template="""Please recommend a selling price for this product.

PRODUCT DATA:
- Cost Price: ${cost_price}
- Current Price: ${current_price}
- Target Margin: {target_margin}%
- Competitor Price: ${competitor_price}
- Price Elasticity: {price_elasticity}

Work through these steps:
1. Margin Floor: minimum price to meet the target margin, and the current margin.
2. Competitive Position: how the current price and the margin floor compare with the competitor.
3. Elasticity Impact: how customer price-sensitivity should shape the price.
4. Recommendation: one optimal price, its margin, and its gap vs the competitor.

End with:
Recommended Price: $<number>
Expected Margin: <number>%
Reasoning: <2-3 sentences>""",
)

QUICK_CHECK_PROMPT = """Quick pricing check:
Cost: ${cost_price}, Target Margin: {target_margin}%, Competitor: ${competitor_price}

Provide a quick price recommendation with brief reasoning."""

PROMPT_STRATEGIES = {"basic": BASIC_PROMPT, "cot": COT_PROMPT, "few-shot": FEW_SHOT_PROMPT}
STRATEGIES = ("basic", "cot", "few-shot", "agent")


def make_llm(temperature=0.2, max_tokens=4096):
    """Create the Groq chat model shared by every strategy."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env at the repository root "
            "and add your Groq API key."
        )
    # Low temperature keeps the numeric reasoning stable between runs. gpt-oss
    # reasons before it answers and those tokens count towards max_tokens, so
    # leave headroom; max_retries rides out free-tier rate limits.
    return ChatGroq(model=MODEL, temperature=temperature, max_tokens=max_tokens, max_retries=10)


class PricingAgent:
    """A pricing agent whose behaviour comes entirely from its prompts."""

    def __init__(self, llm=None):
        self.llm = llm or make_llm()
        # The persona and pricing rules live in the system message, so they are
        # applied to every request without repeating them in each prompt.
        self.system_message = SystemMessage(content=AGENT_SYSTEM_PROMPT)
        self.pricing_template = AGENT_PROMPT

    def get_price_recommendation(self, cost_price, current_price, target_margin, competitor_price, price_elasticity):
        """Return a step-by-step analysis that ends in one recommended price."""
        user_prompt = self.pricing_template.format(
            cost_price=cost_price,
            current_price=current_price,
            target_margin=target_margin,
            competitor_price=competitor_price,
            price_elasticity=price_elasticity,
        )
        response = self.llm.invoke([self.system_message, HumanMessage(content=user_prompt)])
        return response.content

    def quick_price_check(self, cost_price, target_margin, competitor_price):
        """Return a short recommendation from three inputs.

        Deliberately sent without the system message, which shows how much of
        the agent's discipline (one price, margin floor) comes from that message.
        """
        prompt = QUICK_CHECK_PROMPT.format(
            cost_price=cost_price, target_margin=target_margin, competitor_price=competitor_price
        )
        return self.llm.invoke([HumanMessage(content=prompt)]).content


def recommend(strategy, product, llm=None):
    """Run one strategy ("basic", "cot", "few-shot" or "agent") on a product dict."""
    llm = llm or make_llm()
    if strategy == "agent":
        return PricingAgent(llm).get_price_recommendation(**product)
    chain = PROMPT_STRATEGIES[strategy] | llm
    return chain.invoke(product).content


def _number(value):
    """Show 400.0 as 400 so prompts read naturally; keep real decimals."""
    return int(value) if float(value).is_integer() else value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Recommend a selling price with a prompt-only LLM pricing agent."
    )
    parser.add_argument("--cost", type=float, default=EXAMPLE_PRODUCT["cost_price"], help="cost price")
    parser.add_argument("--current", type=float, default=EXAMPLE_PRODUCT["current_price"], help="current selling price")
    parser.add_argument("--margin", type=float, default=EXAMPLE_PRODUCT["target_margin"],
                        help="target gross margin in percent, measured on selling price")
    parser.add_argument("--competitor", type=float, default=EXAMPLE_PRODUCT["competitor_price"], help="competitor price")
    parser.add_argument("--elasticity", type=str.capitalize, choices=ELASTICITY_LEVELS,
                        default=EXAMPLE_PRODUCT["price_elasticity"], help="price elasticity of demand")
    parser.add_argument("--strategy", choices=STRATEGIES + ("quick", "all"), default="agent",
                        help="prompting strategy to run; 'all' runs basic, cot, few-shot and agent in turn")
    args = parser.parse_args(argv)

    if min(args.cost, args.current, args.competitor) <= 0:
        parser.error("prices must be greater than zero")
    if not 0 <= args.margin < 100:
        parser.error("--margin must be at least 0 and below 100")
    return args


def main(argv=None):
    # Force UTF-8 stdout so model output with special characters does not crash on Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    product = {
        "cost_price": _number(args.cost),
        "current_price": _number(args.current),
        "target_margin": _number(args.margin),
        "competitor_price": _number(args.competitor),
        "price_elasticity": args.elasticity,
    }
    llm = make_llm()
    if args.strategy == "quick":
        # The quick check only uses three of the five inputs.
        print(
            f"Product: cost ${product['cost_price']}, target margin {product['target_margin']}%, "
            f"competitor ${product['competitor_price']}\n"
        )
        print(PricingAgent(llm).quick_price_check(
            product["cost_price"], product["target_margin"], product["competitor_price"]
        ))
        return

    print(
        f"Product: cost ${product['cost_price']}, current ${product['current_price']}, "
        f"target margin {product['target_margin']}%, competitor ${product['competitor_price']}, "
        f"elasticity {product['price_elasticity']}\n"
    )
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    for strategy in strategies:
        print("=" * 60)
        print(f"Strategy: {strategy}")
        print("=" * 60)
        print(recommend(strategy, product, llm).strip() + "\n")


if __name__ == "__main__":
    main()

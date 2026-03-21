"""
claude_agent.py — Uses Claude to estimate probabilities and detect edges.
Sends market context to Claude and gets back probability estimates.
"""

import json
from typing import Optional

from rich.console import Console

import config

console = Console()

# Only import anthropic if key is set (optional dependency)
_anthropic_client = None


def _get_client():
    global _anthropic_client
    if _anthropic_client is None:
        if not config.ANTHROPIC_API_KEY:
            return None
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


PROBABILITY_PROMPT = """You are an expert prediction market trader and probability estimator.
You will be given a market question and current pricing data from Kalshi.

Your job:
1. Estimate the TRUE probability of the event occurring (0.0 to 1.0)
2. Compare your estimate to the market price to find edges
3. Provide a confidence level in your estimate (low/medium/high)

Rules:
- Be calibrated. If you say 70%, it should happen ~70% of the time.
- Consider base rates, recent trends, and any relevant data.
- For weather markets, consider seasonal patterns and geography.
- For sports, consider team records, matchups, and recent performance.
- For crypto, consider recent volatility and trend direction.
- If you're uncertain, your confidence should be LOW and your edge should be small.
- NEVER estimate probabilities near 0 or 1 unless absolutely certain.

Respond with ONLY valid JSON in this exact format:
{
    "estimated_probability": 0.65,
    "confidence": "medium",
    "edge_vs_market": 0.05,
    "reasoning": "Brief 1-2 sentence explanation",
    "recommended_side": "yes",
    "recommended_action": "buy"
}"""


class ClaudeAgent:
    """Uses Claude to estimate event probabilities for trading decisions."""

    def __init__(self):
        self.client = _get_client()

    def estimate_probability(
        self,
        market_title: str,
        yes_price: float,
        no_price: float,
        volume: int,
        hours_to_settlement: float,
        additional_context: str = "",
    ) -> Optional[dict]:
        """
        Ask Claude to estimate the probability of a market event.
        Returns dict with estimated_probability, confidence, edge, reasoning.
        """
        if not self.client:
            console.print("[yellow]No Anthropic API key set. Using simple heuristic.[/yellow]")
            return self._heuristic_estimate(yes_price, volume)

        market_price_pct = yes_price / 100.0

        user_msg = f"""Market: {market_title}
Current yes price: {yes_price}c (implied probability: {market_price_pct:.1%})
Current no price: {no_price}c
Volume: {volume} contracts
Time to settlement: {hours_to_settlement:.1f} hours
{f'Additional context: {additional_context}' if additional_context else ''}

What is the TRUE probability of this event? Is there an edge vs the market price?"""

        try:
            response = self.client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=500,
                system=PROBABILITY_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )

            text = response.content[0].text.strip()

            # Parse JSON response
            # Handle case where Claude wraps in markdown code blocks
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)

            # Validate response
            prob = float(result.get("estimated_probability", 0.5))
            if prob < 0.01:
                prob = 0.01
            if prob > 0.99:
                prob = 0.99
            result["estimated_probability"] = prob
            result["edge_vs_market"] = prob - market_price_pct

            return result

        except Exception as e:
            console.print(f"[yellow]Claude API error: {e}. Using heuristic.[/yellow]")
            return self._heuristic_estimate(yes_price, volume)

    def _heuristic_estimate(self, yes_price: float, volume: int) -> dict:
        """
        Simple heuristic when Claude is unavailable.
        Assumes market is roughly efficient but applies a small
        mean-reversion bias for extreme prices.
        """
        market_prob = yes_price / 100.0

        # Mean-reversion: extreme prices tend to revert
        if market_prob > 0.85:
            est_prob = market_prob - 0.03
        elif market_prob < 0.15:
            est_prob = market_prob + 0.03
        else:
            est_prob = market_prob  # Market is probably right

        edge = est_prob - market_prob

        # Low volume markets may have more mispricing
        if volume < 50:
            confidence = "low"
        elif volume < 200:
            confidence = "medium"
        else:
            confidence = "low"  # High volume = more efficient

        side = "no" if edge < 0 else "yes"

        return {
            "estimated_probability": round(est_prob, 4),
            "confidence": confidence,
            "edge_vs_market": round(edge, 4),
            "reasoning": "Heuristic mean-reversion estimate (Claude unavailable)",
            "recommended_side": side,
            "recommended_action": "buy",
        }

    def analyze_batch(self, markets_df, top_n: int = 5) -> list[dict]:
        """Analyze the top N most promising markets."""
        results = []
        for _, row in markets_df.head(top_n).iterrows():
            estimate = self.estimate_probability(
                market_title=row.get("title", row.get("ticker", "")),
                yes_price=row.get("yes_price", 50),
                no_price=row.get("no_price", 50),
                volume=row.get("volume", 0),
                hours_to_settlement=row.get("hours_to_settlement", 24),
            )
            if estimate:
                estimate["ticker"] = row["ticker"]
                estimate["market_price"] = row.get("yes_price", 50)
                results.append(estimate)

        return results

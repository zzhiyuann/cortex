"""Research Agent — Simulates web research for demonstration.

This agent responds to research queries with simulated results.
In a real deployment, you would plug in actual web search / RAG logic.

Usage:
    python examples/research_agent.py
"""

from __future__ import annotations

import argparse
import hashlib
import random

from a2a_hub import Agent, capability


class ResearchAgent(Agent):
    """Agent that simulates web research tasks."""

    name = "researcher"

    @capability("web-search", description="Search the web for information")
    async def search(self, query: str, max_results: int = 3) -> dict:
        """Simulate a web search and return results."""
        # Generate deterministic but varied fake results based on query
        seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        results = []
        domains = [
            "arxiv.org", "github.com", "stackoverflow.com",
            "wikipedia.org", "docs.python.org", "medium.com",
        ]
        for i in range(min(max_results, 5)):
            domain = rng.choice(domains)
            results.append({
                "title": f"Result {i + 1} for: {query}",
                "url": f"https://{domain}/result/{seed + i}",
                "snippet": f"Simulated search result about '{query}'. "
                           f"This would contain relevant information from {domain}.",
            })

        return {
            "query": query,
            "results": results,
            "result_count": len(results),
            "note": "Simulated results — replace with real search API",
        }

    @capability("summarize", description="Summarize text content")
    async def summarize(self, text: str, max_length: int = 200) -> dict:
        """Simulate text summarization."""
        words = text.split()
        if len(words) <= 30:
            summary = text
        else:
            # Simple extractive summary: take first and last sentences
            sentences = text.replace("!", ".").replace("?", ".").split(".")
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) <= 2:
                summary = text[:max_length]
            else:
                summary = f"{sentences[0]}. ... {sentences[-1]}."

        return {
            "summary": summary[:max_length],
            "original_length": len(text),
            "summary_length": len(summary[:max_length]),
            "compression_ratio": round(len(summary[:max_length]) / max(len(text), 1), 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Agent")
    parser.add_argument("--host", default="localhost", help="Hub host")
    parser.add_argument("--port", type=int, default=8765, help="Hub port")
    args = parser.parse_args()

    agent = ResearchAgent()
    print(f"Starting Research Agent, connecting to ws://{args.host}:{args.port}")
    agent.run(hub_host=args.host, hub_port=args.port)


if __name__ == "__main__":
    main()

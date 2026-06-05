"""
core/memory.py
───────────────
Token-efficient conversational memory for multi-turn chat.

Goal: Support follow-up questions ("What else does it say about X?",
"Expand on that last point") without blowing up the LLM context window.

Strategy — sliding window + summary compression:
  1. Keep the last MAX_RECENT_TURNS full turns verbatim.
  2. Older turns are summarised by the LLM into a running SUMMARY.
  3. Each prompt to the LLM receives:
       [system prompt]
       [running summary (if any)]
       [last N full turns]
       [retrieved context chunks]
       [current question]

Why this approach?
  - Simple: no vector memory, no embeddings of chat history.
  - Token-efficient: summary replaces N old turns with ~100 tokens.
  - The LLM summary is coherent narrative, not a raw transcript.

Token budget estimates (Llama 3.3 70B via Groq, 128k context):
  - Each turn ≈ 200-400 tokens.
  - 5 full turns ≈ 1,500 tokens.
  - Summary ≈ 100 tokens.
  - Retrieved chunks (5 × 200 chars) ≈ 350 tokens.
  - Total overhead ≈ 2,000 tokens; leaves 126k for the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


MAX_RECENT_TURNS = 5       # Full turns kept verbatim
SUMMARY_TRIGGER  = 8       # Summarise when total turns exceed this


@dataclass
class Turn:
    role:    str   # "user" or "assistant"
    content: str


@dataclass
class ConversationMemory:
    """
    Maintains a summary of old turns + a window of recent turns.

    Usage:
        mem = ConversationMemory()
        mem.add_turn("user", "What is machine learning?")
        mem.add_turn("assistant", "Machine learning is ...")
        prompt = mem.build_context_block()  # inject into LLM prompt
    """

    _turns:   List[Turn] = field(default_factory=list)
    _summary: str        = ""

    def add_turn(self, role: str, content: str) -> None:
        """Append a new turn to memory."""
        self._turns.append(Turn(role=role, content=content))

    def build_messages(
        self,
        system_prompt: str,
        context_block: str,
        current_question: str,
    ) -> List[dict]:
        """
        Build the full messages list to send to the Groq API.

        Structure:
          - system message (instructions + context chunks)
          - optional: summary of old turns as an assistant message
          - recent turns (user/assistant alternating)
          - current user question

        Args:
            system_prompt:    Role + citation instructions.
            context_block:    The RAG-retrieved chunks formatted as text.
            current_question: The user's current question.

        Returns:
            List of {"role": ..., "content": ...} dicts for the Groq API.
        """
        messages: List[dict] = []

        # System: role + retrieved context
        full_system = (
            f"{system_prompt}\n\n"
            f"=== RETRIEVED DOCUMENT CONTEXT ===\n"
            f"{context_block}\n"
            f"==================================="
        )
        messages.append({"role": "system", "content": full_system})

        # Summary of older turns (if we've compressed any)
        if self._summary:
            messages.append({
                "role": "assistant",
                "content": (
                    f"[Summary of earlier conversation]\n{self._summary}"
                ),
            })

        # Recent verbatim turns (keep last MAX_RECENT_TURNS)
        recent = self._turns[-MAX_RECENT_TURNS * 2:]  # *2 for user+assistant pairs
        for turn in recent:
            messages.append({"role": turn.role, "content": turn.content})

        # Current question (always last)
        messages.append({"role": "user", "content": current_question})

        return messages

    def maybe_compress(self, groq_client, model: str) -> None:
        """
        If the turn count exceeds SUMMARY_TRIGGER, compress older turns
        into a summary using the LLM.

        Called after each assistant response; runs at most once per trigger.
        """
        if len(self._turns) <= SUMMARY_TRIGGER:
            return

        # Turns to compress: everything except the last MAX_RECENT_TURNS pairs
        cutoff = len(self._turns) - MAX_RECENT_TURNS * 2
        old_turns = self._turns[:cutoff]
        self._turns = self._turns[cutoff:]

        # Format old turns for the summarisation prompt
        transcript = "\n".join(
            f"{t.role.upper()}: {t.content}" for t in old_turns
        )

        try:
            resp = groq_client.chat.completions.create(
                model=model,
                max_tokens=200,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation summariser. "
                            "Produce a concise (≤150 words) factual summary "
                            "of the following conversation excerpt. "
                            "Preserve key facts, questions asked, and answers given."
                        ),
                    },
                    {"role": "user", "content": transcript},
                ],
            )
            new_summary = resp.choices[0].message.content.strip()
            # Append to any existing summary
            self._summary = (
                f"{self._summary}\n{new_summary}".strip()
                if self._summary
                else new_summary
            )
        except Exception:
            # Compression is best-effort; never break the chat over it.
            pass

    def clear(self) -> None:
        self._turns.clear()
        self._summary = ""

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def summary(self) -> str:
        return self._summary

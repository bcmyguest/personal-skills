"""LoCoMo loader: real conversational memory data with ground truth.

LoCoMo (Maharana et al. 2024, "Evaluating Very Long-Term Conversational Memory
of LLM Agents", arXiv:2402.17753) is 10 multi-session dialogues -- 5882 turns --
with real per-session timestamps and 199 QA pairs whose `evidence` field names
the exact dialogue turns that answer them.

Why it is the right corpus for this system, and the synthetic one was not:

  * **Real natural language.** The synthetic corpus had 12 routine templates, so
    a VAE could learn the schema almost exactly and novelty detection scored
    AUC 1.0000. That number says more about the generator than the method.
  * **Ground-truth retrieval targets.** `evidence` gives the turns a question
    should recall, so recall@k is measured against a label rather than against
    string overlap with the query.
  * **Real timestamps.** Sessions are dated over months, so the 30-day
    forgetting curve acts on real elapsed time instead of simulated ageing.

Fetch once (2.8 MB) with:

    curl -sLo experiments/data/locomo10.json \\
      https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "data" / "locomo10.json"

# "1:56 pm on 8 May, 2023"
_DATE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ampm>am|pm)\s+on\s+"
    r"(?P<day>\d{1,2})\s+(?P<month>\w+),?\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        "january february march april may june july august september "
        "october november december".split()
    )
}


def parse_timestamp(raw: str) -> datetime:
    """LoCoMo session stamps are free text; parse to an aware datetime.

    Falls back to a fixed epoch rather than raising: a malformed stamp should
    degrade one session's decay behaviour, not abort a benchmark run.
    """
    match = _DATE.search(raw or "")
    if not match:
        return datetime(2023, 1, 1, tzinfo=timezone.utc)
    hour = int(match["h"]) % 12
    if match["ampm"].lower() == "pm":
        hour += 12
    name = match["month"].lower()
    month = _MONTHS.get(name) or next(
        (n for m, n in _MONTHS.items() if m.startswith(name[:3])), 0
    )
    if not month:
        return datetime(2023, 1, 1, tzinfo=timezone.utc)
    try:
        return datetime(
            int(match["year"]), month, int(match["day"]), hour, int(match["m"]),
            tzinfo=timezone.utc,
        )
    except ValueError:
        # "31 February" and friends: the docstring promises a fallback, and
        # datetime raising here would abort a whole benchmark run.
        return datetime(2023, 1, 1, tzinfo=timezone.utc)


@dataclass
class Turn:
    """One dialogue turn, the unit this system stores as an episodic memory."""

    dia_id: str
    speaker: str
    text: str
    session: str
    timestamp: datetime

    @property
    def memory_text(self) -> str:
        """Speaker-prefixed, because who said it is part of the episode."""
        return f"{self.speaker}: {self.text}"


@dataclass
class Question:
    question: str
    answer: str
    category: int
    evidence: list[str] = field(default_factory=list)
    """dia_ids of the turns that answer it -- the retrieval ground truth."""


@dataclass
class Conversation:
    sample_id: str
    turns: list[Turn]
    questions: list[Question]

    def turn_by_id(self) -> dict[str, Turn]:
        return {t.dia_id: t for t in self.turns}

    @property
    def span_days(self) -> float:
        if not self.turns:
            return 0.0
        stamps = [t.timestamp for t in self.turns]
        return (max(stamps) - min(stamps)).total_seconds() / 86400.0


def _as_evidence_list(raw) -> list[str]:
    """`evidence` is sometimes a list, sometimes a bare string."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


def load(path: Path | str = DEFAULT_PATH) -> list[Conversation]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it with:\n"
            "  curl -sLo experiments/data/locomo10.json \\\n"
            "    https://raw.githubusercontent.com/snap-research/locomo/"
            "main/data/locomo10.json"
        )
    raw = json.loads(path.read_text())

    conversations = []
    for sample in raw:
        conv = sample["conversation"]
        session_keys = sorted(
            (k for k in conv if k.startswith("session_") and not k.endswith("date_time")),
            key=lambda k: int(k.split("_")[1]),
        )
        turns: list[Turn] = []
        for key in session_keys:
            stamp = parse_timestamp(conv.get(f"{key}_date_time", ""))
            for turn in conv[key]:
                text = (turn.get("text") or "").strip()
                dia_id = turn.get("dia_id") or ""
                if not text or not dia_id:
                    # A blank dia_id would collide with every other blank one in
                    # turn_by_id(), silently inflating apparent evidence
                    # resolution.
                    continue
                turns.append(
                    Turn(
                        dia_id=dia_id,
                        speaker=turn.get("speaker", "?"),
                        text=text,
                        session=key,
                        timestamp=stamp,
                    )
                )

        questions = [
            Question(
                question=q["question"],
                answer=str(q.get("answer", "")),
                category=int(q.get("category", 0)),
                evidence=_as_evidence_list(q.get("evidence")),
            )
            for q in sample.get("qa", [])
            if q.get("question")
        ]
        conversations.append(
            Conversation(sample["sample_id"], turns, questions)
        )
    return conversations


def stats(conversations: list[Conversation]) -> dict:
    turns = sum(len(c.turns) for c in conversations)
    questions = sum(len(c.questions) for c in conversations)
    answerable = 0
    for conv in conversations:
        # Hoisted: rebuilding this per (question, evidence) made stats() O(Q*E*T).
        ids = conv.turn_by_id()
        answerable += sum(
            1 for q in conv.questions if any(e in ids for e in q.evidence)
        )
    return {
        "conversations": len(conversations),
        "turns": turns,
        "questions": questions,
        "questions_with_resolvable_evidence": answerable,
        "mean_turns_per_conversation": turns / max(len(conversations), 1),
        "mean_span_days": sum(c.span_days for c in conversations)
        / max(len(conversations), 1),
    }

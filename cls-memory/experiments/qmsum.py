"""QMSum loader: real meeting transcripts with annotated relevant spans.

QMSum (Zhong et al. 2021, "QMSum: A New Benchmark for Query-based Multi-domain
Meeting Summarization", arXiv:2104.05938) is 232 meetings across three domains --
product design (AMI), academic (ICSI), and Welsh/Canadian parliamentary
committees -- with queries annotated to the *turn ranges* that answer them.

It is the closest public corpus to what this system is actually for. LoCoMo is
personal dialogue between two friends; QMSum is many-speaker organisational
discussion, with the register, jargon and turn-taking of real work. Using both
guards against tuning to one corpus: a change that helps LoCoMo and hurts QMSum
is overfitting, not an improvement.

`relevant_text_span` gives inclusive [start, end] turn indices, so ground truth
covers a *contiguous span* rather than LoCoMo's scattered individual turns --
a usefully different retrieval shape.

Fetch once (2.7 MB):

    curl -sLo experiments/data/qmsum_test.jsonl \\
      https://raw.githubusercontent.com/Yale-LILY/QMSum/main/data/ALL/jsonl/test.jsonl
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from experiments.locomo import Conversation, Question, Turn

DEFAULT_PATH = Path(__file__).parent / "data" / "qmsum_test.jsonl"

# QMSum has no timestamps. Meetings are spaced a week apart and turns a minute
# apart so the decay machinery has a plausible, monotone timeline to act on;
# nothing in the retrieval measurements depends on the exact spacing.
MEETING_INTERVAL = timedelta(days=7)
TURN_INTERVAL = timedelta(minutes=1)
EPOCH = datetime(2023, 1, 2, 9, 0, tzinfo=timezone.utc)


def _spans_to_indices(spans, n_turns: int) -> list[int]:
    """`relevant_text_span` is [["1", "16"], ...] -- inclusive, string indices."""
    out: list[int] = []
    for span in spans or []:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            continue
        if start > end:
            start, end = end, start
        out.extend(i for i in range(start, end + 1) if 0 <= i < n_turns)
    return sorted(set(out))


def load(path: Path | str = DEFAULT_PATH, *, max_meetings: int | None = None) -> list[Conversation]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it with:\n"
            "  curl -sLo experiments/data/qmsum_test.jsonl \\\n"
            "    https://raw.githubusercontent.com/Yale-LILY/QMSum/"
            "main/data/ALL/jsonl/test.jsonl"
        )

    conversations: list[Conversation] = []
    with path.open() as handle:
        for meeting_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if max_meetings is not None and len(conversations) >= max_meetings:
                break
            record = json.loads(line)
            raw_turns = record.get("meeting_transcripts", [])
            base = EPOCH + meeting_index * MEETING_INTERVAL

            turns: list[Turn] = []
            for i, turn in enumerate(raw_turns):
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                turns.append(
                    Turn(
                        # Index-based ids: the ground-truth spans are indices
                        # into the ORIGINAL list, so the id must encode the
                        # original position, not the filtered one.
                        dia_id=f"m{meeting_index}:t{i}",
                        speaker=turn.get("speaker", "?"),
                        text=content,
                        session=f"meeting_{meeting_index}",
                        timestamp=base + i * TURN_INTERVAL,
                    )
                )

            questions = [
                Question(
                    question=q["query"],
                    answer=str(q.get("answer", "")),
                    category=1,
                    evidence=[
                        f"m{meeting_index}:t{i}"
                        for i in _spans_to_indices(
                            q.get("relevant_text_span"), len(raw_turns)
                        )
                    ],
                )
                for q in record.get("specific_query_list", [])
                if q.get("query")
            ]

            if turns and questions:
                conversations.append(
                    Conversation(f"qmsum_{meeting_index}", turns, questions)
                )
    return conversations


def stats(conversations: list[Conversation]) -> dict:
    turns = sum(len(c.turns) for c in conversations)
    questions = sum(len(c.questions) for c in conversations)
    resolvable = 0
    evidence_sizes = []
    for conv in conversations:
        ids = {t.dia_id for t in conv.turns}
        for question in conv.questions:
            present = [e for e in question.evidence if e in ids]
            if present:
                resolvable += 1
                evidence_sizes.append(len(present))
    return {
        "meetings": len(conversations),
        "turns": turns,
        "questions": questions,
        "questions_with_resolvable_evidence": resolvable,
        "mean_turns_per_meeting": turns / max(len(conversations), 1),
        "mean_evidence_turns_per_question": (
            sum(evidence_sizes) / max(len(evidence_sizes), 1)
        ),
    }

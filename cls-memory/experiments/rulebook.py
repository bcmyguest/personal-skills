"""An organizational rulebook: supersession, scope traps, and a token budget.

Why this corpus exists
----------------------
LoCoMo is the wrong benchmark for remembering *rules*. It is episodic chat with
no rule ever superseded, no scope conditions, and no cost to retrieving too
much -- so it cannot measure the two things that actually matter for a policy
memory:

  * **Staleness.** "Expenses over $500 need VP approval" is later revised to
    "$1000". Injecting the old one into a prompt is worse than injecting
    nothing: the model confidently applies an obsolete rule. A retriever that
    scores on similarity alone ranks the two versions almost identically,
    because they differ by one token.
  * **Context bloat.** Rules are only useful injected into a prompt, and prompt
    space is the budget. "hit@10" is not a cost model. The metric here is
    *tokens spent to cover the applicable rules*, which is what a caller pays.

Adversarial by construction
---------------------------
Three trap families, because a rules corpus that is merely a list of distinct
facts measures nothing that a keyword index cannot already do:

  * **Supersession chains** -- v1 -> v2 -> v3 of the same rule, only the last
    current. Lexically near-identical, so similarity cannot separate them.
  * **Scope near-duplicates** -- the same rule with one dimension changed
    (employee/contractor, EU/US, prod/staging). Retrieving the wrong scope is
    a correctness failure, not a ranking nuisance.
  * **Cross-category collisions** -- shared vocabulary ("approval", "30 days",
    "review") across unrelated policy areas.

The `supersedes` links are ground truth for scoring. They are deliberately
*not* given to the retrieval arms: a store that already knows which rules are
dead makes the whole problem trivial, and real policy documents arrive without
that annotation. Detecting supersession from the text is the actual task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    text: str
    category: str
    day: int
    """Days after EPOCH that this rule took effect."""
    supersedes: str | None = None
    """Ground truth only -- never exposed to a retrieval arm."""

    @property
    def effective(self) -> datetime:
        return EPOCH + timedelta(days=self.day)


@dataclass(frozen=True)
class Situation:
    question: str
    gold: tuple[str, ...]
    """rule_ids that a correct answer needs, current versions only."""
    traps: tuple[str, ...] = field(default_factory=tuple)
    """rule_ids that are plausible but wrong -- stale versions or wrong scope.
    Retrieving these is scored as a distinct failure, not just a miss."""


# ---------------------------------------------------------------------------
# Expenses: a three-step supersession chain plus a contractor scope split.

_EXPENSES = [
    Rule("exp.approval.v1",
         "Expense reports over $500 require approval from a Vice President "
         "before reimbursement is issued.",
         "expenses", 0),
    Rule("exp.approval.v2",
         "Expense reports over $1000 require approval from a Vice President "
         "before reimbursement is issued.",
         "expenses", 210, supersedes="exp.approval.v1"),
    Rule("exp.approval.v3",
         "Expense reports over $1000 require approval from a Vice President, "
         "and any report over $5000 additionally requires the CFO to sign off "
         "before reimbursement is issued.",
         "expenses", 430, supersedes="exp.approval.v2"),
    Rule("exp.contractor.scope",
         "Contractors may not submit expense reports directly. All contractor "
         "costs are invoiced through the engaging vendor and are settled on "
         "standard vendor payment terms.",
         "expenses", 60),
    Rule("exp.receipt.v1",
         "Receipts must be attached for any individual expense line above $25.",
         "expenses", 0),
    Rule("exp.receipt.v2",
         "Receipts must be attached for any individual expense line above $75.",
         "expenses", 300, supersedes="exp.receipt.v1"),
    Rule("exp.travel.eu",
         "Travel booked for staff based in the EU must use the European travel "
         "desk, and per-diem rates follow the published EU schedule.",
         "expenses", 90),
    Rule("exp.travel.us",
         "Travel booked for staff based in the US must use the North American "
         "travel desk, and per-diem rates follow the published US schedule.",
         "expenses", 90),
]

# ---------------------------------------------------------------------------
# Data retention: numbers that collide with the expense thresholds on purpose.

_DATA = [
    Rule("data.retention.v1",
         "Customer support transcripts are retained for 30 days and then "
         "permanently deleted.",
         "data", 0),
    Rule("data.retention.v2",
         "Customer support transcripts are retained for 90 days and then "
         "permanently deleted.",
         "data", 365, supersedes="data.retention.v1"),
    Rule("data.pii.eu",
         "Personal data belonging to EU residents may only be processed in EU "
         "regions, and may not be copied into US-hosted analytics systems.",
         "data", 30),
    Rule("data.pii.us",
         "Personal data belonging to US residents may be processed in any "
         "company-controlled region provided the region is SOC 2 in scope.",
         "data", 30),
    Rule("data.export.v1",
         "Bulk export of customer records requires approval from the Data "
         "Protection Officer and is logged for audit.",
         "data", 45),
    Rule("data.backup",
         "Production databases are backed up nightly and backups are retained "
         "for 35 days.",
         "data", 15),
]

# ---------------------------------------------------------------------------
# Access control: prod/staging scope split, plus a revoked exception.

_ACCESS = [
    Rule("access.prod.v1",
         "Production database access requires a ticket approved by the service "
         "owner and is granted for 7 days.",
         "access", 0),
    Rule("access.prod.v2",
         "Production database access requires a ticket approved by the service "
         "owner and by the on-call security engineer, and is granted for "
         "24 hours.",
         "access", 400, supersedes="access.prod.v1"),
    Rule("access.staging",
         "Staging database access is self-service for any engineer and does "
         "not require a ticket.",
         "access", 0),
    Rule("access.contractor.v1",
         "Contractors may be granted production read access for the duration "
         "of their engagement.",
         "access", 20),
    Rule("access.contractor.v2",
         "Contractors may not be granted production access of any kind. "
         "Contractor work requiring production data must be performed by a "
         "supervising employee.",
         "access", 380, supersedes="access.contractor.v1"),
    Rule("access.review",
         "All standing access grants are reviewed quarterly and revoked if not "
         "reconfirmed by the service owner.",
         "access", 120),
]

# ---------------------------------------------------------------------------
# Procurement and HR: shared "approval" / "30 days" vocabulary, unrelated.

_OTHER = [
    Rule("proc.vendor.v1",
         "New vendor engagements above $10000 require approval from Procurement "
         "and a completed security review.",
         "procurement", 0),
    Rule("proc.vendor.v2",
         "New vendor engagements above $25000 require approval from Procurement "
         "and a completed security review. Engagements involving customer data "
         "require a security review at any value.",
         "procurement", 340, supersedes="proc.vendor.v1"),
    Rule("proc.notice",
         "Vendor contracts must be cancelled with at least 30 days written "
         "notice to avoid automatic renewal.",
         "procurement", 50),
    Rule("hr.leave",
         "Requests for leave longer than 10 working days require approval from "
         "the reporting manager at least 30 days in advance.",
         "hr", 0),
    Rule("hr.contractor.term",
         "Contractor engagements are reviewed every 90 days and terminate "
         "automatically unless renewed in writing.",
         "hr", 70),
    Rule("hr.equipment",
         "Company laptops must be returned within 5 working days of a person's "
         "last day.",
         "hr", 100),
]

RULES: list[Rule] = _EXPENSES + _DATA + _ACCESS + _OTHER


SITUATIONS: list[Situation] = [
    Situation(
        "An employee submitted a $6,200 expense report for a conference. "
        "What approvals are needed?",
        gold=("exp.approval.v3",),
        traps=("exp.approval.v1", "exp.approval.v2"),
    ),
    Situation(
        "Someone wants to expense a $700 client dinner. Does that need a VP?",
        gold=("exp.approval.v3",),
        traps=("exp.approval.v1", "exp.approval.v2"),
    ),
    Situation(
        "Does a $40 taxi ride need a receipt attached?",
        gold=("exp.receipt.v2",),
        traps=("exp.receipt.v1",),
    ),
    Situation(
        "A contractor in Berlin wants to be reimbursed for a flight they "
        "booked themselves. How does that work?",
        gold=("exp.contractor.scope",),
        traps=("exp.travel.eu", "exp.approval.v1", "exp.approval.v2"),
    ),
    Situation(
        "An engineer based in Ireland needs to book travel to a customer site. "
        "Which desk and what per-diem?",
        gold=("exp.travel.eu",),
        traps=("exp.travel.us",),
    ),
    Situation(
        "How long do we keep customer support chat logs before deleting them?",
        gold=("data.retention.v2",),
        traps=("data.retention.v1", "data.backup"),
    ),
    Situation(
        "We want to pipe EU customer support transcripts into our US data "
        "warehouse for analysis. Is that allowed, and how long can we keep them?",
        gold=("data.pii.eu", "data.retention.v2"),
        traps=("data.pii.us", "data.retention.v1"),
    ),
    Situation(
        "A analyst needs to pull every customer record into a spreadsheet. "
        "What is required?",
        gold=("data.export.v1",),
        traps=("data.pii.us", "proc.vendor.v2"),
    ),
    Situation(
        "An engineer needs to query the production database to debug an "
        "incident. What do they need and how long does the access last?",
        gold=("access.prod.v2",),
        traps=("access.prod.v1", "access.staging"),
    ),
    Situation(
        "A contractor on the payments project needs to read production data. "
        "Can we grant that?",
        gold=("access.contractor.v2",),
        traps=("access.contractor.v1", "access.prod.v1", "access.prod.v2"),
    ),
    Situation(
        "Does an engineer need a ticket to poke at the staging database?",
        gold=("access.staging",),
        traps=("access.prod.v1", "access.prod.v2"),
    ),
    Situation(
        "We are signing a $30,000 deal with a new analytics vendor who will "
        "process customer data. What is required before we sign?",
        gold=("proc.vendor.v2",),
        traps=("proc.vendor.v1", "data.export.v1"),
    ),
    Situation(
        "We want out of a vendor contract before it renews. How much notice?",
        gold=("proc.notice",),
        traps=("hr.leave", "hr.contractor.term"),
    ),
    Situation(
        "Someone wants three weeks off in the summer. What is the process?",
        gold=("hr.leave",),
        traps=("proc.notice", "hr.contractor.term"),
    ),
    Situation(
        "A contractor's engagement is ending. What happens to their access and "
        "their laptop?",
        gold=("access.contractor.v2", "hr.equipment", "hr.contractor.term"),
        traps=("access.contractor.v1", "access.review"),
    ),
    Situation(
        "How often do we re-check who still has standing access to systems?",
        gold=("access.review",),
        traps=("hr.contractor.term", "data.backup"),
    ),
]


def current_rule_ids() -> set[str]:
    """Rules not superseded by any other rule -- the live policy set."""
    dead = {r.supersedes for r in RULES if r.supersedes}
    return {r.rule_id for r in RULES if r.rule_id not in dead}


def stale_rule_ids() -> set[str]:
    return {r.supersedes for r in RULES if r.supersedes}


def by_id() -> dict[str, Rule]:
    return {r.rule_id: r for r in RULES}


def stats() -> dict:
    current = current_rule_ids()
    return {
        "rules": len(RULES),
        "current": len(current),
        "superseded": len(RULES) - len(current),
        "chains": sum(1 for r in RULES if r.supersedes),
        "situations": len(SITUATIONS),
        "mean_gold_per_situation": sum(len(s.gold) for s in SITUATIONS) / len(SITUATIONS),
    }


if __name__ == "__main__":
    for key, value in stats().items():
        print(f"{key:<28} {value}")
    # Every gold id must be current, every trap id must exist: a corpus bug
    # here would silently score a correct system as wrong.
    ids = by_id()
    current = current_rule_ids()
    for s in SITUATIONS:
        for g in s.gold:
            assert g in ids, f"unknown gold id {g}"
            assert g in current, f"gold id {g} is superseded"
        for t in s.traps:
            assert t in ids, f"unknown trap id {t}"
    print("corpus self-check OK")

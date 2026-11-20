"""Synthetic organizational corpus with ground-truth labels.

The point of a synthetic corpus here is that we know, per item, whether it is
routine (schema-conformant) or anomalous (schema-violating). That makes the
novelty gate measurable as a binary classifier instead of something we squint
at, and it lets retrieval be scored against a known target.

Structure mirrors what an org actually accumulates:

  ROUTINE   high-volume, templated, low-entropy. Deploys, backups, tickets.
            Generated from a small template set with slot fillers, so a
            competent schema learner should reconstruct them well.

  ANOMALY   rare, off-template, lexically distinct. Incidents and oddities.
            These are what the hippocampus should capture.

  RULE      evergreen policy statements. Timeless, must never be forgotten.

Anomalies deliberately reuse *some* organizational vocabulary (service names,
regions) so they are not trivially separable by vocabulary alone -- the schema
has to notice that the *structure* is wrong, not just that a word is unseen.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class Label(str, Enum):
    ROUTINE = "routine"
    ANOMALY = "anomaly"
    RULE = "rule"


@dataclass(frozen=True)
class Item:
    text: str
    label: Label
    template_id: int


SERVICES = [
    "billing", "checkout", "search", "notifications", "inventory",
    "recommendations", "auth", "pricing", "shipping", "catalog",
]
REGIONS = ["eu-west", "us-east", "ap-south", "us-west", "eu-central"]
TEAMS = ["finance", "operations", "engineering", "support", "compliance"]
CLUSTERS = ["primary", "replica", "analytics", "archive"]
ENVIRONMENTS = ["staging", "production", "sandbox", "qa"]

ROUTINE_TEMPLATES = [
    "deployment of the {service} service completed without incident in {region}",
    "nightly batch job finished successfully for {service} in the {region} region",
    "weekly report generated and delivered to the {team} team on schedule",
    "customer support ticket resolved within the standard response window by {team}",
    "routine database backup completed for the {cluster} cluster in {region}",
    "scheduled maintenance window opened for the {env} environment in {region}",
    "health check passed for the {service} service across all {region} nodes",
    "access review completed for the {team} team with no exceptions raised",
    "certificate renewed automatically for the {service} service in {region}",
    "capacity report for the {cluster} cluster shows normal utilisation in {region}",
    "log rotation completed for the {service} service in the {env} environment",
    "monthly invoice batch processed successfully for the {team} team",
]

ANOMALY_TEMPLATES = [
    "the shard rebalancer corrupted the {service} index during an unplanned failover",
    "a contractor deleted the {env} namespace by hand and nobody noticed for hours",
    "the payment provider rotated credentials without notice and {service} began failing",
    "an unfamiliar vendor invoice arrived from a shell company registered in belize",
    "someone disabled audit logging on the {cluster} cluster and left no ticket",
    "a junior engineer force pushed over the release branch erasing three days of work",
    "the {service} service leaked customer records into a public bucket for six hours",
    "an attacker brute forced a dormant admin account and pivoted into {region}",
    "the {env} database was restored from a backup that turned out to be empty",
    "a rogue cron job spawned twelve thousand containers and exhausted the {region} quota",
    "the on call engineer was unreachable for the entire duration of a sev one outage",
    "an expired certificate took down {service} because the renewal alert was muted",
]

RULE_TEMPLATES = [
    "all refunds above 500 dollars require director approval",
    "production deploys are frozen during the december code freeze",
    "customer data may not leave the eu-west region under any circumstances",
    "every privileged access grant expires automatically after 90 days",
    "incident reviews must be published within five working days",
    "no single engineer may both author and approve a production change",
]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        service=rng.choice(SERVICES),
        region=rng.choice(REGIONS),
        team=rng.choice(TEAMS),
        cluster=rng.choice(CLUSTERS),
        env=rng.choice(ENVIRONMENTS),
    )


def generate(
    n_routine: int = 2000,
    n_anomaly: int = 60,
    *,
    include_rules: bool = True,
    unique_anomalies: bool = True,
    seed: int = 0,
) -> list[Item]:
    """Build a labelled corpus. Routine items dominate, as in reality.

    `unique_anomalies` prefixes every anomaly with a unique incident reference.
    This is not a convenience for the metrics -- it is what makes the retrieval
    task well-posed. With 12 anomaly templates and 60 anomalies, roughly five
    incidents share each template and differ only in a slot filler, so a
    contiguous 50% cue frequently matches several stored memories *verbatim*.
    Measured on a 30-anomaly draw without references: only 19 of 30 texts were
    distinct and 22 of 30 cues matched more than one memory. Recall@1 then
    measures corpus degeneracy, not the model. Real incident logs carry
    identifiers for exactly this reason.

    Set False to reproduce the degenerate regime deliberately.
    """
    rng = random.Random(seed)
    items: list[Item] = []

    for _ in range(n_routine):
        tid = rng.randrange(len(ROUTINE_TEMPLATES))
        items.append(Item(_fill(ROUTINE_TEMPLATES[tid], rng), Label.ROUTINE, tid))

    refs = rng.sample(range(1000, 9999), n_anomaly)
    for n in range(n_anomaly):
        tid = rng.randrange(len(ANOMALY_TEMPLATES))
        body = _fill(ANOMALY_TEMPLATES[tid], rng)
        if unique_anomalies:
            body = f"incident inc {refs[n]} {body} lasting {rng.randint(2, 96)} hours"
        items.append(Item(body, Label.ANOMALY, tid))

    if include_rules:
        for tid, text in enumerate(RULE_TEMPLATES):
            items.append(Item(text, Label.RULE, tid))

    return items


def split(items: list[Item]) -> tuple[list[Item], list[Item], list[Item]]:
    routine = [i for i in items if i.label is Label.ROUTINE]
    anomaly = [i for i in items if i.label is Label.ANOMALY]
    rules = [i for i in items if i.label is Label.RULE]
    return routine, anomaly, rules


def partial_cue(text: str, fraction: float, *, rng: random.Random) -> str:
    """A realistic degraded query: a contiguous fragment of the original text.

    Contiguous rather than a random word subset -- a person recalling an
    incident remembers a phrase, not a bag of scattered tokens.
    """
    words = text.split()
    keep = max(2, int(round(len(words) * fraction)))
    if keep >= len(words):
        return text
    start = rng.randrange(0, len(words) - keep + 1)
    return " ".join(words[start : start + keep])

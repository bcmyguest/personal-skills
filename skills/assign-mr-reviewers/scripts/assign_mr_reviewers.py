#!/usr/bin/env python3
"""Assign reviewers to a GitLab MR from CODEOWNERS or team-spectra fallback.

Usage:
  assign_mr_reviewers.py <MR_URL> [--exclude user1,user2]
  assign_mr_reviewers.py <PROJECT_PATH> <MR_IID> [--exclude user1,user2]

Allowed glab API endpoints (read-only):
  - projects/{id_or_path}
  - projects/{id}/merge_requests/{iid}
  - projects/{id}/merge_requests/{iid}/diffs
  - projects/{id}/merge_requests/{iid}/approval_rules
  - projects/{id}/repository/files/{path}?ref=main
  - groups/{TEAM_SPECTRA_GROUP_ID}/members
"""

import argparse
import fnmatch
import json
import random
import re
import subprocess
import sys
from base64 import b64decode

TEAM_SPECTRA_GROUP_ID = 210

ALLOWED_ENDPOINT_PATTERNS = [
    re.compile(r"^projects/[a-zA-Z0-9%._-]+$"),
    re.compile(r"^projects/\d+/merge_requests/\d+$"),
    re.compile(r"^projects/\d+/merge_requests/\d+/diffs$"),
    re.compile(r"^projects/\d+/merge_requests/\d+/approval_rules$"),
    re.compile(r"^projects/\d+/repository/files/[a-zA-Z0-9%._/-]+\?ref=main$"),
    re.compile(rf"^groups/{TEAM_SPECTRA_GROUP_ID}/members$"),
]


def glab_api(endpoint: str) -> dict | list | None:
    if not any(p.match(endpoint) for p in ALLOWED_ENDPOINT_PATTERNS):
        raise ValueError(f"Blocked API endpoint: {endpoint}")
    result = subprocess.run(
        ["glab", "api", endpoint],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def parse_mr_url(url: str) -> tuple[str, str]:
    match = re.match(r"https?://[^/]+/(.+?)/-/merge_requests/(\d+)", url)
    if match:
        return match.group(1), match.group(2)
    raise ValueError(f"Could not parse MR URL: {url}")


def validate_project_path(path: str) -> None:
    if not re.match(r"^[a-zA-Z0-9._/-]+$", path):
        raise ValueError(f"Invalid project path: {path}")


def validate_mr_iid(iid: str) -> None:
    if not re.match(r"^\d+$", iid):
        raise ValueError(f"Invalid MR IID: {iid}")


def validate_username(username: str) -> None:
    if not re.match(r"^[a-zA-Z0-9_-]+$", username):
        raise ValueError(f"Invalid username: {username}")


# --- Step 1: Get MR info ---


def get_project_id(project_path: str) -> int:
    validate_project_path(project_path)
    data = glab_api(f"projects/{project_path.replace('/', '%2F')}")
    if not data:
        raise ValueError(f"Project not found: {project_path}")
    return data["id"]


def get_mr_details(project_id: int, mr_iid: str) -> dict:
    validate_mr_iid(mr_iid)
    mr = glab_api(f"projects/{project_id}/merge_requests/{mr_iid}")
    if not mr:
        raise ValueError(f"MR !{mr_iid} not found")
    return {
        "title": mr.get("title"),
        "author": mr.get("author", {}).get("username"),
        "reviewers": [r.get("username") for r in mr.get("reviewers", [])],
        "state": mr.get("state"),
    }


def get_changed_files(project_id: int, mr_iid: str) -> list[str]:
    validate_mr_iid(mr_iid)
    diffs = glab_api(f"projects/{project_id}/merge_requests/{mr_iid}/diffs")
    if not diffs:
        return []
    return [d.get("new_path", d.get("old_path", "")) for d in diffs]


def get_required_approvals(project_id: int, mr_iid: str) -> int:
    validate_mr_iid(mr_iid)
    rules = glab_api(f"projects/{project_id}/merge_requests/{mr_iid}/approval_rules")
    if not rules:
        return 2
    return max((r.get("approvals_required", 0) for r in rules), default=2)


def get_codeowners(project_id: int) -> str | None:
    for path in ["CODEOWNERS", ".gitlab%2FCODEOWNERS", "docs%2FCODEOWNERS"]:
        data = glab_api(f"projects/{project_id}/repository/files/{path}?ref=main")
        if data and "content" in data:
            return b64decode(data["content"]).decode()
    return None


# --- Step 2: Pick reviewers ---


def parse_codeowners(content: str) -> list[tuple[str, list[str]]]:
    rules = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            pattern = parts[0]
            owners = [o.lstrip("@") for o in parts[1:]]
            rules.append((pattern, owners))
    return rules


def match_codeowners(changed_files: list[str], codeowners_content: str) -> list[str]:
    rules = parse_codeowners(codeowners_content)
    matched_owners = set()
    for filepath in changed_files:
        last_match_owners = None
        for pattern, owners in rules:
            if pattern.startswith("/"):
                if fnmatch.fnmatch("/" + filepath, pattern + "*") or fnmatch.fnmatch(
                    "/" + filepath, pattern
                ):
                    last_match_owners = owners
            elif fnmatch.fnmatch(filepath, pattern) or fnmatch.fnmatch(
                filepath.split("/")[-1], pattern
            ):
                last_match_owners = owners
        if last_match_owners:
            matched_owners.update(last_match_owners)
    return list(matched_owners)


def get_team_members() -> list[str]:
    members = glab_api(f"groups/{TEAM_SPECTRA_GROUP_ID}/members")
    if not members:
        return []
    return [m["username"] for m in members if m.get("state") == "active"]


def pick_reviewers(
    author: str,
    existing_reviewers: list[str],
    required: int,
    changed_files: list[str],
    has_codeowners: bool,
    codeowners_content: str | None,
    exclude: set[str],
) -> tuple[list[str], str, list[str]]:
    """Returns (picks, source, remaining_candidates)."""
    existing = set(existing_reviewers)
    needed = required - len(existing)

    if needed <= 0:
        return [], "none", []

    source = "team-spectra"
    candidates = []

    if has_codeowners and codeowners_content:
        owners = match_codeowners(changed_files, codeowners_content)
        candidates = [
            o for o in owners if o not in {author} | existing | exclude
        ]
        if candidates:
            source = "CODEOWNERS"

    if not candidates:
        all_members = get_team_members()
        candidates = [
            m for m in all_members if m not in {author} | existing | exclude
        ]
        source = "team-spectra"

    picks = random.sample(candidates, min(needed, len(candidates)))
    remaining = [c for c in candidates if c not in picks]
    return picks, source, remaining


# --- Step 3: Assign reviewers ---


def assign_reviewers(project_path: str, mr_iid: str, reviewers: list[str]) -> str:
    validate_project_path(project_path)
    validate_mr_iid(mr_iid)
    for r in reviewers:
        validate_username(r)

    reviewer_arg = ",".join(f"+{r}" for r in reviewers)
    result = subprocess.run(
        ["glab", "mr", "update", mr_iid, "--reviewer", reviewer_arg, "-R", project_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Error assigning reviewers: {result.stderr}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(
        description="Assign reviewers to a GitLab MR from CODEOWNERS or team-spectra."
    )
    parser.add_argument("target", nargs="+", help="MR URL, or PROJECT_PATH MR_IID")
    parser.add_argument(
        "--exclude", default="", help="Comma-separated usernames to exclude"
    )
    args = parser.parse_args()

    if args.target[0].startswith("http"):
        project_path, mr_iid = parse_mr_url(args.target[0])
    elif len(args.target) >= 2:
        project_path, mr_iid = args.target[0], args.target[1]
    else:
        print("Usage: assign_mr_reviewers.py <MR_URL | PROJECT_PATH MR_IID>", file=sys.stderr)
        sys.exit(1)

    validate_project_path(project_path)
    validate_mr_iid(mr_iid)
    exclude = set(filter(None, args.exclude.split(",")))

    # Get MR info
    project_id = get_project_id(project_path)
    details = get_mr_details(project_id, mr_iid)
    changed_files = get_changed_files(project_id, mr_iid)
    required = get_required_approvals(project_id, mr_iid)
    codeowners = get_codeowners(project_id)

    # Pick reviewers
    picks, source, remaining = pick_reviewers(
        author=details["author"],
        existing_reviewers=details["reviewers"],
        required=required,
        changed_files=changed_files,
        has_codeowners=codeowners is not None,
        codeowners_content=codeowners,
        exclude=exclude,
    )

    output = {
        "project_path": project_path,
        "mr_iid": mr_iid,
        "title": details["title"],
        "author": details["author"],
        "existing_reviewers": details["reviewers"],
        "required_approvals": required,
        "source": source,
        "remaining_candidates": remaining,
    }

    if not picks:
        needed = required - len(details["reviewers"])
        output["status"] = (
            "already_has_reviewers" if needed <= 0 else "no_eligible_candidates"
        )
        output["assigned"] = []
        print(json.dumps(output, indent=2))
        return

    # Assign reviewers
    assign_output = assign_reviewers(project_path, mr_iid, picks)
    output["status"] = "ok"
    output["assigned"] = picks
    output["glab_output"] = assign_output.strip()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

"""Group exact normalized clause text and flag federal sample-plan citations."""

from collections import defaultdict
import hashlib
import re
import unicodedata

from text_checks import TABLES

ITEM = re.compile(r"(?m)^\s*(?:(?:\([ivxlcdm]+\)|\([a-z]\)|\(\d+\)|\d+(?:\.\d+)+[.)]?|\d+[.)])\s+)+")
CHAINAGE = re.compile(r"\b(?:ch(?:ainage)?\.?\s*)?(?:km\.?\s*)?\d+\+\d{2,4}(?:\.\d+)?\b")
FEDERAL = re.compile(r"\bfhwa\s*[-– ]\s*hif\s*[-– ]\s*18\s*[-– ]\s*032\b", re.I)
ADAPTATION = re.compile(
    r"\b(?:adapted?|based|developed|modeled)\s+(?:from|on|after|using)\s+(?:the\s+)?"
    r"(?:federal|fhwa)\s+(?:[\w-]+\s+){0,5}sample\s+(?:[\w-]+\s+){0,5}plan\b", re.I)


def normalized(text, project_names=()):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = ITEM.sub("", text)
    for name in sorted(set(project_names), key=len, reverse=True):
        if name.strip():
            text = re.sub(r"(?<!\w)" + re.escape(unicodedata.normalize("NFKC", name).casefold())
                          + r"(?!\w)", "", text)
    text = CHAINAGE.sub("", text)
    aliases = {}
    for unit, options in TABLES["units"].items():
        for option in (*options, unit):
            key = unicodedata.normalize("NFKC", option).casefold()
            if key in aliases and aliases[key] != unit:
                # Keep an ambiguous abbreviation literal.
                aliases[key] = key
            else:
                aliases[key] = unit
    pattern = "|".join(map(re.escape, sorted(aliases, key=len, reverse=True)))
    text = re.sub(r"(?<![a-z])(?:" + pattern + r")(?![a-z])",
                  lambda match: aliases[match[0]], text)
    return re.sub(r"\s+", "", text)


def template_flag(text):
    return bool(FEDERAL.search(text) or ADAPTATION.search(" ".join(text.split())))


def groups(development, heldout):
    index = defaultdict(list)
    observed = set()
    for phase, rows in [("development", development), ("heldout", heldout)]:
        for row in rows:
            key = row["clause_id"]
            if key in observed:
                raise ValueError(f"Repeated clause identifier {key}")
            observed.add(key)
            value = normalized(row["clause_text"], row.get("project_names", ()))
            if not value:
                raise ValueError(f"No clause text after normalization for {key}")
            digest = hashlib.sha256(value.encode()).hexdigest()
            index[digest].append(dict(clause_id=key, source_id=row["source_id"], phase=phase))
    return dict(index)

"""
Deterministic safety-intent filter for the Voice RAG pipeline.

Design principles:
- O(1) per query: pre-compiled regex patterns checked via re.search()
- High-precision patterns: require procedural intent signals ("how to", "instructions",
  "step by step", "make", "build", "create", "synthesize", etc.) paired with
  dangerous-category nouns to minimise false positives on educational queries.
- Never calls an LLM or external API.
- Language-agnostic for English; Indic-language safe queries are extremely unlikely
  to trigger these patterns (patterns are English-only intentionally).
- Returns a named SafetyCheckResult so callers can inspect category and latency.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Pre-compiled danger patterns (English-only) ───────────────────────────────
#
# Each pattern requires BOTH a procedural-intent signal AND a danger-category
# noun to fire, keeping false-positive rate low on factual / educational queries.
#
# Pattern structure: (?:procedural_signal).*(?:danger_noun) OR explicit phrases.
# Re-compiled once at import time.

_PROCEDURAL = r"(?:how\s+(?:to|do\s+i|can\s+i)|make\s+(?:a\s+)?|build\s+(?:a\s+)?|create\s+(?:a\s+)?|instructions?\s+(?:to|for)|steps?\s+(?:to|for)|guide\s+(?:to|for)|recipe\s+for|synthesize|manufacture|produce|assemble|craft)"

_PATTERNS: list[tuple[str, str]] = [
    # ─── Explosive / weapon construction ──────────────────────────────────────
    ("weapons_explosives",
     r"(?i)(?:" + _PROCEDURAL + r").{0,60}(?:bomb|explosive|ied|grenade|detonator|fuse|ammonium\s+nitrate|pipe\s+bomb|c4|semtex|tnt|gunpowder|thermite|napalm|molotov)"),
    ("weapons_explosives_direct",
     r"(?i)\b(?:build|make|construct|assemble)\s+(?:a\s+)?(?:bomb|explosive|ied|grenade|pipe\s+bomb)\b"),
    ("weapons_firearms",
     r"(?i)(?:" + _PROCEDURAL + r").{0,60}(?:3d.{0,5}print.{0,10}gun|unregistered\s+firearm|ghost\s+gun|untraceable\s+weapon|convert.{0,15}semi.{0,5}auto|illegal\s+mod)"),

    # ─── Firearms (additional) ────────────────────────────────────────────────
    ("weapons_firearms_print",
     r"(?i)(?:3d.{0,5}print|print.{0,10}3d).{0,30}(?:gun|pistol|weapon|firearm)"),
    ("weapons_firearms_untraceable",
     r"(?i)untraceable\s+(?:gun|firearm|weapon)"),
    ("chem_bio_weapons",
     r"(?i)(?:" + _PROCEDURAL + r").{0,60}(?:sarin|nerve\s+agent|vx\s+gas|ricin|anthrax|botulinum|chlorine\s+gas\s+weapon|mustard\s+gas|novichok)"),
    ("chem_bio_synthesis",
     r"(?i)(?:synthesize|synthesise|produce|manufacture).{0,40}(?:sarin|ricin|anthrax\s+spore|vx\s+gas|novichok)"),

    # ─── Malware / hacking / credential theft ────────────────────────────────
    ("malware",
     r"(?i)(?:" + _PROCEDURAL + r").{0,60}(?:ransomware|keylogger|rootkit|trojan\s+horse|rat\s+malware|worm\s+virus|botnet|exploit\s+kit|payload|reverse\s+shell|privilege\s+escalation\s+exploit)"),
    ("credential_theft",
     r"(?i)(?:" + _PROCEDURAL + r").{0,60}(?:phish(?:ing)?\s+page|credential\s+harvest|steal\s+password|bypass\s+2fa|session\s+hijack|sql\s+injection\s+attack|xss\s+attack)"),

    # ─── Self-harm / suicide instructions ────────────────────────────────────
    ("self_harm",
     r"(?i)(?:" + _PROCEDURAL + r").{0,60}(?:commit\s+suicide|kill\s+(?:my)?self|self.{0,5}harm|slit\s+(?:my\s+)?wrist|overdose\s+on|lethal\s+dose\s+of)"),
    ("self_harm_direct",
     r"(?i)\b(?:instructions?|method|ways?)\s+(?:to|for)\s+(?:commit\s+)?(?:suicide|self.{0,5}harm|kill\s+(?:my)?self)\b"),

    # ─── Sexual exploitation / CSAM ──────────────────────────────────────────
    ("sexual_exploitation",
     r"(?i)\b(?:child\s+porn|csam|underage\s+sex|minors?\s+(?:sexually|nude)|loli(?:ta)?\s+(?:porn|nude)|grooming\s+(?:a\s+)?(?:child|minor|kid))\b"),
    ("grooming_minor",
     r"(?i)\b(?:how\s+to\s+groom|groom\s+a\s+(?:child|minor|kid|teenage\w*))\b"),

    # ─── Drug synthesis ──────────────────────────────────────────────────────
    ("drug_synthesis",
     r"(?i)(?:" + _PROCEDURAL + r").{0,60}(?:methamphetamine|meth\s+lab|synthesize\s+(?:mdma|heroin|fentanyl|cocaine)|cook\s+meth|illicit\s+drug\s+lab)"),
    ("drug_synthesis_cook",
     r"(?i)\b(?:cooking|cook)\s+meth\b|\binstructions?\s+for\s+(?:cooking|making)\s+meth\b"),
    # ─── Self-harm additional ───────────────────────────────────────────────
    ("self_harm_methods",
     r"(?i)\b(?:methods?|ways?)\s+(?:for\s+)?(?:killing\s+(?:my)?self|ending\s+(?:my\s+)?life|committing\s+suicide|self.{0,5}harm)\b"),
]

# Pre-compile all patterns once at module load
_COMPILED: list[tuple[str, re.Pattern]] = [
    (cat, re.compile(pat)) for cat, pat in _PATTERNS
]


@dataclass
class SafetyCheckResult:
    """Result of the safety intent filter."""
    is_unsafe: bool
    category: Optional[str] = None
    latency_ms: float = 0.0
    # Human-readable refusal reason for the response
    refusal_reason: str = ""


def check_safety(text: str) -> SafetyCheckResult:
    """
    Runs all compiled patterns against `text` in O(n_patterns * text_length).

    Parameters
    ----------
    text : str
        The STT transcript to check (already stripped of leading/trailing whitespace).

    Returns
    -------
    SafetyCheckResult
        is_unsafe=True if any pattern fires, along with the matched category.
        is_unsafe=False otherwise, with sub-millisecond latency.
    """
    t0 = time.perf_counter()

    for category, pattern in _COMPILED:
        if pattern.search(text):
            latency_ms = (time.perf_counter() - t0) * 1000
            return SafetyCheckResult(
                is_unsafe=True,
                category=category,
                latency_ms=latency_ms,
                refusal_reason=f"unsafe_request: category={category}"
            )

    latency_ms = (time.perf_counter() - t0) * 1000
    return SafetyCheckResult(is_unsafe=False, latency_ms=latency_ms)

"""Plain data: what an interrogation record is made of. No AI in here."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

ROLES = ("interrogator", "suspect", "other")
# How a sentence relates to a crime fact. The model only labels this; the ledger decides what it means.
KINDS = ("asserts", "asks", "denies", "assents")


@dataclass
class Session:
    id: str
    label: str            # "Feb 27, 2006, Mishicot High School"
    day: date
    recorded: bool = True  # False when there's no public transcript (it may still have been taped)
    source: str = ""       # citation, e.g. "Calumet County Sheriff's Dept. transcript, pp. 1-55"


@dataclass
class Utterance:
    id: str               # "S2-0412"
    session: str          # Session.id
    seq: int              # global order across all sessions (code assigns it)
    speaker: str          # as printed: "WIEGERT", "BRENDAN"
    role: str             # one of ROLES
    text: str
    page: str = ""        # page as printed in the transcript (for citations)
    sensitive: bool = False  # graphic content: hidden in the UI unless revealed
    page_end: str = ""    # last printed page when the turn continues onto the next page
    split: bool = False   # a nod pulled out of an investigator's printed turn (not a turn of its own)


@dataclass
class Fact:
    id: str               # "F07"
    label: str            # "Shot in the head"
    detail: str = ""      # what exactly counts as stating it
    source: str = ""      # where the fact list says this came from
    public_before: date | None = None  # reported publicly before the interrogation (so not 'inside knowledge')
    holdback: bool = False             # police say they kept it secret
    planted: bool = False              # a deliberately false detail investigators used as a test
    prior_statement: str = ""          # a documented earlier statement by the suspect with no public transcript (with source)
    public_what: str = ""                # what exactly was reported publicly, with its source
    must: list = field(default_factory=list)  # key-term groups: a line carries the fact only if it has one alternate from every group
    # What was reported before the interviews, each with its source and the key-term groups (indices into `must`) whose
    # part of the detail it states: {"what", "source", "page"?, "date"? (a dated report), "before"? and "court_list"? (the
    # court's list of what was reported 'long before' the March 1 confession), "covers": [0, 1]}. Public only if they
    # cover every group; otherwise the groups not covered decide by the transcript.
    reports: list = field(default_factory=list)
    related_reports: list = field(default_factory=list)  # reports the detail could be inferred from, not reports of it
    not_reported: str = ""  # the part of the detail no report states, in words ("the branches")
    court_notes: list = field(default_factory=list)  # the court's own observations: {"page", "text", "quotes": [...]}


@dataclass
class Mention:
    utterance: str        # Utterance.id
    fact: str             # Fact.id
    kind: str             # one of KINDS
    quote: str = ""       # the words that carry the fact
    by: str = "model"     # "model" | "matcher" | "keyword" | "human"
    verified: bool | None = None  # verifier agent's check of the quote (None = not checked)


@dataclass
class Case:
    id: str
    title: str
    suspect: str
    sessions: list[Session] = field(default_factory=list)
    utterances: list[Utterance] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

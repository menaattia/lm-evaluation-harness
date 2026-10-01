"""Shared preprocessing and prompt functions for the AITA perspective tasks."""

from __future__ import annotations

import random


VALID_VERDICTS = {"NTA", "YTA"}
OPTION_SHUFFLE_SEED = 0


def _correct_answer(doc):
    verdict = str(doc.get("verdict", "")).strip().upper()
    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"Expected verdict to be NTA or YTA, got {doc.get('verdict')!r}"
        )
    # Every rendered question asks whether Party A was wrong.
    return "No" if verdict == "NTA" else "Yes"


def _shuffled_options(doc):
    """Seed and shuffle options consistently for every perspective of a source."""
    source_id = str(doc.get("source_id") or doc.get("id") or "").strip()
    if not source_id:
        raise ValueError("Every example must have a non-empty source_id or id")
    rng = random.Random(f"aita-perspective-{OPTION_SHUFFLE_SEED}:{source_id}")
    options = ["Yes", "No"]
    rng.shuffle(options)
    return options


def process_docs(dataset):
    """Add deterministic answer options and the correct option index."""

    def _prepare_doc(doc):
        scenario = str(doc.get("scenario_text", "")).strip()
        if not scenario:
            raise ValueError("Every example must have non-empty scenario_text")

        correct = _correct_answer(doc)
        options = _shuffled_options(doc)
        return {
            "Scenario": scenario,
            "Options": options,
            "Answer": options.index(correct),
        }

    return dataset.map(_prepare_doc)


def process_original_docs(dataset):
    """Prepare original AITA posts for the same judgment task."""

    def _prepare_doc(doc):
        title = str(doc.get("title", "")).strip()
        selftext = str(doc.get("selftext", "")).strip()
        if not selftext:
            raise ValueError("Every original example must have non-empty selftext")

        scenario = f"{title}\n\n{selftext}" if title else selftext
        correct = _correct_answer(doc)
        options = _shuffled_options(doc)
        return {
            "Scenario": scenario,
            "Options": options,
            "Answer": options.index(correct),
        }

    return dataset.map(_prepare_doc)


def doc_to_text(doc):
    """Prompt for a binary judgment of the rendered scenario."""
    return f'{doc["Scenario"]}\n\nOnly output Yes or No, nothing else.'


def doc_to_choice(doc):
    """Return the Yes/No strings scored by lm-eval after the prompt."""
    return doc["Options"]


def doc_to_target(doc):
    """Return the index of the correct item in doc_to_choice."""
    return int(doc["Answer"])


def doc_to_target_text(doc):
    """Return the correct Yes/No answer for generate-until task variants."""
    return doc_to_choice(doc)[doc_to_target(doc)]

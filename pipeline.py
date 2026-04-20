"""
pipeline.py – Production-grade Moderation Pipeline
===================================================
Part 5 deliverable for FAST-NUCES Responsible & Explainable AI Assignment 2.

Three-layer architecture:
  Layer 1 – Regex pre-filter (fast, deterministic, category-labelled)
  Layer 2 – Calibrated DistilBERT classifier (mitigated model from Part 4)
  Layer 3 – Human review queue (low-confidence escalation)

Usage:
    from pipeline import ModerationPipeline
    pipe = ModerationPipeline(model_path='path/to/best_mitigated_model')
    result = pipe.predict("some comment text")
    # Returns dict with keys: decision, layer, category (if Layer 1), confidence
"""

import re
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 – REGEX BLOCKLIST
# Patterns organised by category, each returning a block decision with
# the matched category name (auditable).
# ══════════════════════════════════════════════════════════════════════════════

BLOCKLIST: dict[str, list[re.Pattern]] = {

    # ── Category 1: Direct threats of violence (≥5 patterns) ─────────────────
    "direct_threat": [
        # "I will / I'm gonna / going to [kill|murder|shoot|stab|hurt] you"
        re.compile(
            r"\b(i'?m?\s+)?(will|gonna|going\s+to|should|shall)\s+"
            r"(kill|murder|shoot|stab|hurt|attack|harm|destroy)\s+(you|u|him|her|them)\b",
            re.IGNORECASE
        ),
        # "you're going to die / you will die"
        re.compile(
            r"\byou('?re|r)?\s+(going\s+to|gonna|will)\s+(die|be\s+killed|be\s+dead)\b",
            re.IGNORECASE
        ),
        # "I'll find where you live / I know where you sleep"
        re.compile(
            r"\bi'?ll?\s+find\s+(where|out\s+where)\s+you\b",
            re.IGNORECASE
        ),
        # "someone should shoot / kill / stab you"
        re.compile(
            r"\bsomeone\s+should\s+(kill|shoot|murder|stab|hurt|attack)\s+(you|him|her|them)\b",
            re.IGNORECASE
        ),
        # Generic: [subject] [threat verb] [you|them] tonight/soon/now
        re.compile(
            r"\b(you|he|she|they)\s+(will|won'?t\s+survive|won'?t\s+make\s+it)\b",
            re.IGNORECASE
        ),
        # "I'm coming for you"
        re.compile(
            r"\bi'?m\s+coming\s+for\s+(you|u|him|her|them)\b",
            re.IGNORECASE
        ),
    ],

    # ── Category 2: Calls for self-harm / suicide (≥4 patterns) ──────────────
    "self_harm_directed": [
        # "you should kill yourself / go kill yourself"
        re.compile(
            r"\b(you\s+should|go\s+and?|just)\s+kill\s+yourself\b",
            re.IGNORECASE
        ),
        # "nobody / no one would miss you if you died"
        re.compile(
            r"\b(nobody|no\s+one|noone)\s+(would|will|even)\s+miss\s+you\b",
            re.IGNORECASE
        ),
        # "do everyone a favour and disappear / end it"
        re.compile(
            r"\bdo\s+(everyone|the\s+world)\s+a\s+favou?r\s+and\s+(disappear|end\s+it|die|leave)\b",
            re.IGNORECASE
        ),
        # "the world would be better without you"
        re.compile(
            r"\bworld\s+would\s+be\s+better\s+without\s+(you|u|him|her|them)\b",
            re.IGNORECASE
        ),
        # "you don't deserve to live"
        re.compile(
            r"\byou\s+don'?t\s+deserve\s+to\s+(live|exist|breathe)\b",
            re.IGNORECASE
        ),
    ],

    # ── Category 3: Doxxing and stalking threats (≥4 patterns) ───────────────
    "doxxing_stalking": [
        # "I know where you live / work / sleep"
        re.compile(
            r"\bi\s+(know|found|have)\s+(where|your)\s+(you\s+)?(live|home|address|work|sleep)\b",
            re.IGNORECASE
        ),
        # "I will / I'll post your address / phone / info"
        re.compile(
            r"\bi'?ll?\s+(post|share|leak|publish|release)\s+your\s+"
            r"(address|phone(\s+number)?|info|information|personal|details|number)\b",
            re.IGNORECASE
        ),
        # "I found your real name / identity"
        re.compile(
            r"\bi\s+(found|know|have)\s+your\s+(real\s+)?(name|identity|account|profile)\b",
            re.IGNORECASE
        ),
        # "everyone will know who you really are / your real identity"
        re.compile(
            r"\beveryone\s+will\s+(know|find\s+out)\s+(who\s+you|your\s+real|the\s+truth)\b",
            re.IGNORECASE
        ),
        # "I'm watching you / I've been following you"
        re.compile(
            r"\bi'?m?\s+(watching|following|tracking|monitoring)\s+(you|u|him|her|them)\b",
            re.IGNORECASE
        ),
    ],

    # ── Category 4: Severe dehumanisation (≥4 patterns) ─────────────────────
    "dehumanization": [
        # "[group] are not human / are subhuman"
        re.compile(
            r"\b\w+\s+are\s+(not\s+)?(?:human|people|persons?)\b.*"
            r"|\bsubhuman\b",
            re.IGNORECASE
        ),
        # "[group] are animals / vermin / parasites"
        re.compile(
            r"\b\w+\s+(are|r)\s+(animals?|vermin|parasites?|cockroaches?|rats?|pests?|insects?)\b",
            re.IGNORECASE
        ),
        # "[group] should be exterminated / eliminated / wiped out"
        re.compile(
            r"\b\w+\s+should\s+be\s+(exterminated?|eliminated?|wiped\s+out|eradicated?|cleansed?)\b",
            re.IGNORECASE
        ),
        # "[group] are a disease / cancer / plague"
        re.compile(
            r"\b\w+\s+(are|is|r)\s+(a\s+)?(disease|cancer|plague|infection|virus|tumor|blight)\b",
            re.IGNORECASE
        ),
        # "inferior race / genetic inferiors"
        re.compile(
            r"\b(inferior|lesser|lower)\s+(race|species|breed|stock|genetics?)\b",
            re.IGNORECASE
        ),
    ],

    # ── Category 5: Coordinated harassment signals (≥3 patterns) ─────────────
    "coordinated_harassment": [
        # "everyone report [username] / this account"
        re.compile(
            r"\beveryone\s+(go\s+)?(report|flag|block|spam)\s+\S+",
            re.IGNORECASE
        ),
        # "let's all go after / target / attack [user/account]"
        re.compile(
            r"\blet'?s\s+(all\s+)?(go\s+after|target|attack|mass\s+report|dog\s*pile)\b",
            re.IGNORECASE
        ),
        # "raid their / his / her profile / server / stream"
        re.compile(
            r"\braid\s+(their|his|her|the)\s+(profile|server|stream|channel|page|account)\b",
            re.IGNORECASE
        ),
        # Lookahead: "mass report" regardless of what follows
        re.compile(
            r"\bmass\s+report(?=\b)",
            re.IGNORECASE
        ),
        # "coordinate [harassment|attack|reporting] on/against"
        re.compile(
            r"\bcoordinate\s+(an?\s+)?(harassment|attack|report(ing)?)\s+(on|against|toward)\b",
            re.IGNORECASE
        ),
    ],
}


def input_filter(text: str) -> dict | None:
    """
    Layer 1: Regex pre-filter.

    Returns a block decision dict if any pattern matches, else None.
    The dict includes the matched category for auditability.
    """
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision"   : "block",
                    "layer"      : "input_filter",
                    "category"   : category,
                    "confidence" : 1.0,
                    "reason"     : f"Matched regex category: {category}"
                }
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 – CALIBRATED DISTILBERT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class _BERTSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Thin sklearn wrapper around a HuggingFace model so we can pass it to
    CalibratedClassifierCV.
    """
    def __init__(self, hf_model, tokenizer, device="cpu", max_length=128):
        self.hf_model   = hf_model
        self.tokenizer  = tokenizer
        self.device     = device
        self.max_length = max_length
        self.classes_   = np.array([0, 1])

    def fit(self, X, y=None):
        return self  # already fine-tuned; nothing to do here

    @torch.no_grad()
    def predict_proba(self, X):
        """X is a list of strings."""
        enc = self.tokenizer(
            list(X),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        logits = self.hf_model(**enc).logits
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        return probs  # shape [N, 2]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.enc.items()}


# ══════════════════════════════════════════════════════════════════════════════
# MODERATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class ModerationPipeline:
    """
    Production-grade three-layer content moderation pipeline.

    Parameters
    ----------
    model_path : str
        Path to the best mitigated HuggingFace model (saved with
        model.save_pretrained() and tokenizer.save_pretrained()).
    block_threshold : float
        Calibrated probability above which the model auto-blocks. Default 0.6.
    allow_threshold : float
        Calibrated probability below which the model auto-allows. Default 0.4.
    calibration_texts : list[str] | None
        A representative sample of texts used to fit the isotonic calibrator.
        If None, the raw model probabilities are used (no calibration step).
    calibration_labels : list[int] | None
        Corresponding binary labels for calibration_texts.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model_path: str,
        block_threshold: float = 0.6,
        allow_threshold: float = 0.4,
        calibration_texts=None,
        calibration_labels=None,
        device: str | None = None,
    ):
        self.block_threshold = block_threshold
        self.allow_threshold = allow_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── Load HuggingFace model ────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        hf_model = hf_model.to(self.device)
        hf_model.eval()

        self._hf_model = hf_model
        self._wrapper  = _BERTSklearnWrapper(
            hf_model=hf_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # ── Calibration ────────────────────────────────────────────────────────
        if calibration_texts is not None and calibration_labels is not None:
            print("Getting raw probabilities for calibration set…")
            raw_probs = self._raw_probs_batch(calibration_texts)

            from sklearn.isotonic import IsotonicRegression
            self._iso = IsotonicRegression(out_of_bounds='clip')
            self._iso.fit(raw_probs, np.array(calibration_labels))
            self._calibrated = True
            print("Isotonic calibrator fitted.")
        else:
            self._calibrated = False
            self._iso = None
            print("No calibration data provided; using raw model probabilities.")
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _raw_probs_batch(self, texts: list, batch_size: int = 64) -> np.ndarray:
        """Return raw P(toxic) for a list of texts via batched inference."""
        ds = _TextDataset(texts, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size)
        all_probs = []
        for batch in loader:
            batch  = {k: v.to(self.device) for k, v in batch.items()}
            logits = self._hf_model(**batch).logits
            probs  = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
        return np.array(all_probs)

    def _model_proba(self, text: str) -> float:
        """Return calibrated (or raw) P(toxic) for a single text."""
        raw = float(self._raw_probs_batch([text])[0])
        if self._calibrated:
            return float(self._iso.predict([raw])[0])
        return raw

    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """
        Run the three-layer pipeline on a single comment.

        Returns
        -------
        dict with keys:
            decision    : 'block' | 'allow' | 'review'
            layer       : 'input_filter' | 'model' | 'human_review'
            confidence  : float  (1.0 for Layer 1, model prob for Layers 2/3)
            category    : str    (only present for Layer 1 matches)
            reason      : str    (human-readable explanation)
        """
        text = str(text).strip()

        # ── Layer 1: Regex pre-filter ─────────────────────────────────────────
        filter_result = input_filter(text)
        if filter_result is not None:
            return filter_result

        # ── Layer 2: Calibrated model ─────────────────────────────────────────
        confidence = self._model_proba(text)

        if confidence >= self.block_threshold:
            return {
                "decision"   : "block",
                "layer"      : "model",
                "confidence" : round(float(confidence), 4),
                "reason"     : f"Model confidence {confidence:.3f} ≥ block threshold {self.block_threshold}"
            }

        if confidence <= self.allow_threshold:
            return {
                "decision"   : "allow",
                "layer"      : "model",
                "confidence" : round(float(confidence), 4),
                "reason"     : f"Model confidence {confidence:.3f} ≤ allow threshold {self.allow_threshold}"
            }

        # ── Layer 3: Human review queue ───────────────────────────────────────
        return {
            "decision"   : "review",
            "layer"      : "human_review",
            "confidence" : round(float(confidence), 4),
            "reason"     : (
                f"Model confidence {confidence:.3f} in uncertainty band "
                f"({self.allow_threshold}–{self.block_threshold}); escalated for human review."
            )
        }

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_batch(self, texts: list[str], batch_size: int = 64) -> list[dict]:
        """
        Run the pipeline on a list of texts efficiently.

        Layer 1 is applied per-text; Layer 2 uses batched inference.
        """
        results = [None] * len(texts)
        needs_model_idx = []
        needs_model_text = []

        # Apply Layer 1 first (fast, no GPU needed)
        for i, text in enumerate(texts):
            r = input_filter(str(text).strip())
            if r is not None:
                results[i] = r
            else:
                needs_model_idx.append(i)
                needs_model_text.append(str(text).strip())

        # Batch inference for texts that passed Layer 1
        if needs_model_text:
            ds     = _TextDataset(needs_model_text, self.tokenizer)
            loader = DataLoader(ds, batch_size=batch_size)
            all_probs = []
            for batch in loader:
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                logits = self._hf_model(**batch).logits
                probs  = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)

            for idx, text, confidence in zip(needs_model_idx, needs_model_text, all_probs):
                confidence = float(confidence)
                if confidence >= self.block_threshold:
                    results[idx] = {
                        "decision": "block", "layer": "model",
                        "confidence": round(confidence, 4),
                        "reason": f"Model confidence {confidence:.3f} ≥ {self.block_threshold}"
                    }
                elif confidence <= self.allow_threshold:
                    results[idx] = {
                        "decision": "allow", "layer": "model",
                        "confidence": round(confidence, 4),
                        "reason": f"Model confidence {confidence:.3f} ≤ {self.allow_threshold}"
                    }
                else:
                    results[idx] = {
                        "decision": "review", "layer": "human_review",
                        "confidence": round(confidence, 4),
                        "reason": (
                            f"Uncertainty band ({self.allow_threshold}–{self.block_threshold}); "
                            "escalated for human review."
                        )
                    }

        return results


# ══════════════════════════════════════════════════════════════════════════════
# Quick smoke-test (run when executed directly)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Pipeline Smoke Test (Layer 1 only) ===\n")

    test_cases = [
        # Should trigger direct_threat
        ("I will kill you tonight.",                  "direct_threat"),
        ("Someone should shoot him.",                 "direct_threat"),
        # Should trigger self_harm_directed
        ("You should just kill yourself.",            "self_harm_directed"),
        ("The world would be better without you.",    "self_harm_directed"),
        # Should trigger doxxing_stalking
        ("I'll post your address online.",            "doxxing_stalking"),
        ("Everyone will know who you really are.",    "doxxing_stalking"),
        # Should trigger dehumanization
        ("They are vermin and should be eliminated.", "dehumanization"),
        # Should trigger coordinated_harassment
        ("Everyone mass report this account now.",    "coordinated_harassment"),
        ("Let's all go after his profile.",           "coordinated_harassment"),
        # Should NOT trigger Layer 1
        ("I love the weather today.",                 None),
        ("This is a perfectly normal comment.",       None),
    ]

    passed = 0
    for text, expected_category in test_cases:
        result = input_filter(text)
        actual = result["category"] if result else None
        status = "✓" if actual == expected_category else "✗"
        print(f"  {status} [{actual or 'no match':25s}] {text[:60]}")
        if actual == expected_category:
            passed += 1

    print(f"\n{passed}/{len(test_cases)} tests passed.")

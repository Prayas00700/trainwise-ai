"""Multilingual (English / Hindi / Hinglish) intent classifier for the support assistant."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from trainwise.models.base import BaseModel, FeatureSpec

_DIGITS = re.compile(r"\d+")


def normalise(text: str) -> str:
    return _DIGITS.sub(" num ", str(text).lower()).strip()


class IntentClassifier(BaseModel):
    name = "intent_classifier"
    task = "classification"
    target = "intent"
    primary_metric = "accuracy"
    features = (FeatureSpec("text", "text", pipeline="chat_channel"),)

    def fit(self, df: pd.DataFrame) -> "IntentClassifier":
        text = df["text"].map(normalise)
        self.pipeline = Pipeline([
            ("tfidf", FeatureUnion([
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")),
            ])),
            ("clf", LogisticRegression(max_iter=3000, C=8.0)),
        ]).fit(text, df[self.target])
        self.reference_ = {"text": "hello"}
        self.trained_rows_ = len(df)
        return self

    def proba(self, X) -> pd.DataFrame:
        text = self.frame(X)["text"].astype(str).map(normalise)
        return pd.DataFrame(self.pipeline.predict_proba(text), columns=self.pipeline.classes_)

    def score(self, X) -> np.ndarray:
        return self.proba(X).max(axis=1).to_numpy()

    def explain(self, record: dict, top_k: int = 5) -> list:
        return []   # occlusion over a single text field is not informative

    def predict_frame(self, X) -> list:
        P = self.proba(X)
        out = []
        for _, row in P.iterrows():
            ranked = row.sort_values(ascending=False)
            out.append({"intent": str(ranked.index[0]), "confidence": round(float(ranked.iloc[0]), 4),
                        "alternatives": [{"intent": str(i), "p": round(float(p), 4)} for i, p in ranked.iloc[1:3].items()],
                        "_monitor_value": str(ranked.index[0])})
        return out

    def evaluate(self, df: pd.DataFrame) -> dict:
        pred = [r["intent"] for r in self.predict_frame(df)]
        return {"accuracy": round(float(accuracy_score(df[self.target], pred)), 4),
                "macro_f1": round(float(f1_score(df[self.target], pred, average="macro")), 4), "n": int(len(df))}

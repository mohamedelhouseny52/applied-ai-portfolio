"""English request intent detection, conservative routing and reference retrieval."""
from pathlib import Path
import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from .common import ROOT, write_json, versions, digest, bar_chart


def redact(text):
    text = unicodedata.normalize("NFKC", str(text)).strip().casefold()
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " [email] ", text)
    text = re.sub(r"(?<!\w)(?:\+?\d[\d -]{7,}\d)(?!\w)", " [number] ", text)
    return re.sub(r"\s+", " ", text).strip()


def pipeline():
    return Pipeline([
        ("text", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=35000)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=40000)),
        ])),
        ("classifier", LogisticRegression(C=4, max_iter=1000, random_state=42)),
    ])


def team_for(intent):
    if any(word in intent for word in ["cash", "atm", "withdraw"]):
        return "cash_operations"
    if "transfer" in intent or "beneficiary" in intent:
        return "transfers"
    if "card" in intent or "pin" in intent:
        return "cards"
    if "top_up" in intent:
        return "funding"
    return "account_support"


def evaluate(model, texts, labels, threshold, margin):
    prob = model.predict_proba(texts)
    order = np.sort(prob, axis=1)
    accepted = (order[:, -1] >= threshold) & (order[:, -1] - order[:, -2] >= margin)
    predicted = model.classes_[prob.argmax(axis=1)]
    report = {"accuracy": accuracy_score(labels, predicted), "macro_f1": f1_score(labels, predicted, average="macro"), "auto_route_coverage": float(accepted.mean()), "auto_route_accuracy": float((np.asarray(labels)[accepted] == predicted[accepted]).mean()) if accepted.any() else None, "review_share": float((~accepted).mean()), "auto_route_count": int(accepted.sum())}
    return report, predicted, accepted, prob.max(axis=1)


def infer(bundle, text):
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        raise ValueError("text must contain 1–4000 characters")
    clean = redact(text)
    model = bundle["model"]
    p = model.predict_proba([clean])[0]
    ranked = np.argsort(-p)
    top, second = ranked[:2]
    features = model.named_steps["text"].transform([clean])
    similarity = cosine_similarity(features, bundle["reference_vectors"])[0]
    nearest = np.argsort(-similarity)[:3]
    lexical = float(similarity[nearest[0]])
    reasons = []
    if p[top] < bundle["threshold"]:
        reasons.append("low_model_score")
    if p[top] - p[second] < bundle["margin"]:
        reasons.append("ambiguous_intents")
    if lexical < .2:
        reasons.append("low_reference_similarity")
    if len(clean.split()) < 3:
        reasons.append("insufficient_context")
    intent = str(model.classes_[top])
    urgency = "high" if re.search(r"\b(stolen|fraud|unauthori[sz]ed)\b", clean) else "normal"
    if urgency == "high":
        reasons.append("sensitive_request_requires_review")
    return {"intent": intent, "recommended_team": team_for(intent), "urgency": urgency,
            "summary": clean[:180], "summary_type": "redacted_excerpt", "model_score": float(p[top]),
            "score_margin": float(p[top] - p[second]), "action": "manual_review" if reasons else "route",
            "review_reasons": reasons, "alternatives": [{"intent": str(model.classes_[i]), "score": float(p[i])} for i in ranked[:3]],
            "references": [{"reference_id": int(i), "intent": bundle["reference_labels"][i], "similarity": float(similarity[i])} for i in nearest]}


def run(train_path=None, test_path=None, out=None):
    train_path = Path(train_path or ROOT / "data/raw/banking_train.csv")
    test_path = Path(test_path or ROOT / "data/raw/banking_test.csv")
    out = Path(out or ROOT / "artifacts/routing"); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_path); test = pd.read_csv(test_path)
    raw["clean"] = raw.text.map(redact); test["clean"] = test.text.map(redact)
    if raw.category.isna().any() or test.category.isna().any():
        raise ValueError("Missing category")
    # Contradictory labels for the same normalized text cannot be learned consistently.
    conflict = set(raw.groupby("clean").category.nunique().loc[lambda s: s > 1].index)
    raw = raw[~raw.clean.isin(conflict)].drop_duplicates("clean")
    overlaps = test.clean.isin(raw.clean)
    held = test[~overlaps].drop_duplicates("clean")
    fit, validation = train_test_split(raw, test_size=.2, stratify=raw.category, random_state=42)
    model = pipeline(); model.fit(fit.clean, fit.category)
    margin = .15
    choices = []
    for t in [.4, .5, .6, .7, .8, .9]:
        m, *_ = evaluate(model, validation.clean, validation.category, t, margin)
        if m["auto_route_count"] >= 50 and m["auto_route_accuracy"] >= .95:
            choices.append((m["auto_route_coverage"], t))
    threshold = max(choices)[1] if choices else 1.0
    val_report, *_ = evaluate(model, validation.clean, validation.category, threshold, margin)
    test_report, predicted, accepted, scores = evaluate(model, held.clean, held.category, threshold, margin)
    references = fit.reset_index(drop=True)
    bundle = {"model": model, "threshold": threshold, "margin": margin,
              "reference_vectors": model.named_steps["text"].transform(references.clean),
              "reference_labels": references.category.tolist()}
    probes = ["Please recommend a pasta recipe", "What is the weather in Cairo", "I want to book a hotel room", "My card was stolen", "help", "I sent money to another bank and it has not arrived", "My card has not arrived yet", "Email me at nadia@example.test about card delivery"]
    probe_results = [{"input": p, "result": infer(bundle, p)} for p in probes]
    pd.DataFrame({"actual": held.category.values, "predicted": predicted, "model_score": scores, "score_gate_pass": accepted}).to_csv(out / "holdout_predictions.csv", index=False)
    write_json(out / "per_intent.json", classification_report(held.category, predicted, output_dict=True, zero_division=0))
    write_json(out / "probe_results.json", probe_results)
    report = {"validation": val_report, "test": test_report, "threshold": threshold, "margin": margin, "threshold_selection": "Maximum validation coverage with >=95% routed accuracy and >=50 cases", "split_sizes": {"fit": len(fit), "validation": len(validation), "test": len(held)}, "test_overlaps_removed": int(overlaps.sum()), "contradictory_train_texts_removed": len(conflict), "classes": len(model.classes_), "source_sha256": {"train": digest(train_path), "test": digest(test_path)}, "versions": versions(), "limitations": ["English BANKING77 benchmark, not a general multilingual helpdesk model", "Scores are not independently calibrated probabilities", "Coverage metrics apply score/margin gates only; inference adds lexical, context and sensitivity gates", "Regex redaction is partial, not a privacy guarantee", "Out-of-domain rejection is heuristic and only smoke-tested", "Reference retrieval returns examples and labels, not verified answers or generative RAG", "Urgency/team mapping is a demo policy requiring business approval"]}
    write_json(out / "metrics.json", report)
    joblib.dump(bundle, out / "model.joblib")
    bar_chart({"All-query accuracy": test_report["accuracy"], "Routed accuracy": test_report["auto_route_accuracy"] or 0, "Routing coverage": test_report["auto_route_coverage"]}, "77-intent routing: deduplicated held-out queries", out / "evaluation.png", "Fraction")
    return report

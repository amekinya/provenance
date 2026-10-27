"""Fine-tune NVIDIA's llama-nemotron-rerank-1b-v2 into a task model, on a Nebius Serverless Job (one L40S).
Shared by Provenance (sentence-detail matcher) and Faithful (interpretation fidelity checker).

Data: JSONL pairs {"query", "passage", "label" (0/1), "split" ("train"/"dev"/"test"), "group"} downloaded from a URL
(our public GitHub repo). Writes the model + metrics to OUT_DIR (a mounted Nebius bucket), and prints metrics.
Env: DATA_URL, NAME (e.g. faithful-checker), OUT_DIR (default /models), EPOCHS, LR, BATCH, MAX_LEN,
REPORT_CLASS ("1" reports metrics for label 1; "0" for label 0, e.g. Faithful's critical errors, scored as 1 - p).
"""
from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

BASE = "nvidia/llama-nemotron-rerank-1b-v2"
NAME = os.environ.get("NAME", "provenance-matcher")
OUT = Path(os.environ.get("OUT_DIR", "/models")) / NAME
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "1e-5"))
BATCH = int(os.environ.get("BATCH", "32"))
MAX_LEN = int(os.environ.get("MAX_LEN", "320"))
REPORT_CLASS = int(os.environ.get("REPORT_CLASS", "1"))
torch.manual_seed(0)
random.seed(0)


def template(q: str, p: str) -> str:
    return f"question:{q} \n \n passage:{p}"


def load() -> dict[str, list[dict]]:
    raw = urllib.request.urlopen(os.environ["DATA_URL"], timeout=120).read().decode()
    rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
    out: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for r in rows:
        out[r.get("split", "train")].append(r)
    print({k: len(v) for k, v in out.items()}, "positives in train:", sum(r["label"] for r in out["train"]), flush=True)
    return out


@torch.inference_mode()
def predict(model, tok, rows: list[dict], dev: str) -> list[float]:
    model.eval()
    scores = []
    for i in range(0, len(rows), 128):
        b = rows[i:i + 128]
        enc = tok([template(r["query"], r["passage"]) for r in b], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(**enc).logits
        scores += torch.sigmoid(logits.float().view(-1)).tolist()
    return scores


def _flip(rows: list[dict], scores: list[float]) -> tuple[list[dict], list[float]]:
    if REPORT_CLASS == 1:
        return rows, scores
    return [r | {"label": 1 - r["label"]} for r in rows], [1 - s for s in scores]


def metrics(rows: list[dict], scores: list[float]) -> dict:
    """Accuracy/F1 at the threshold chosen on dev, plus AUC (rank-based), for the reported class."""
    rows, scores = _flip(rows, scores)
    def at(th):
        tp = sum(1 for r, s in zip(rows, scores) if s >= th and r["label"]); fp = sum(1 for r, s in zip(rows, scores) if s >= th and not r["label"])
        fn = sum(1 for r, s in zip(rows, scores) if s < th and r["label"]); tn = len(rows) - tp - fp - fn
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        return {"threshold": th, "accuracy": round((tp + tn) / max(len(rows), 1), 4), "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4)}
    pos = [s for r, s in zip(rows, scores) if r["label"]]; neg = [s for r, s in zip(rows, scores) if not r["label"]]
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / max(len(pos) * len(neg), 1) if len(pos) * len(neg) < 4e7 else None
    best = max((at(t / 100) for t in range(5, 96, 5)), key=lambda m: m["f1"])
    return {"auc": round(auc, 4) if auc is not None else None, **best}


def main() -> None:
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
    data = load()
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # fp32 master weights; the forward pass runs in bf16 under autocast. (Pure bf16 weights round away most AdamW
    # updates at this learning rate.)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, trust_remote_code=True, dtype=torch.float32).to(dev)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.eos_token_id

    base_dev = predict(model, tok, data["dev"], dev)
    base_test = predict(model, tok, data["test"], dev)
    m_base_dev = metrics(data["dev"], base_dev)
    print("base model dev:", m_base_dev, flush=True)

    train = data["train"]
    steps = math.ceil(len(train) / BATCH * EPOCHS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    pos_w = torch.tensor((len(train) - sum(r["label"] for r in train)) / max(sum(r["label"] for r in train), 1), device=dev).clamp(max=10)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    step = 0
    model.train()
    while step < steps:
        random.shuffle(train)
        for i in range(0, len(train), BATCH):
            if step >= steps:
                break
            b = train[i:i + BATCH]
            enc = tok([template(r["query"], r["passage"]) for r in b], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
            y = torch.tensor([float(r["label"]) for r in b], device=dev)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(**enc).logits
            logits = logits.float().view(-1)
            loss = lossf(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        model.train()

    m_dev = metrics(data["dev"], predict(model, tok, data["dev"], dev))
    th = m_dev["threshold"]
    tuned_test = predict(model, tok, data["test"], dev)
    m_test = metrics(data["test"], tuned_test)
    m_base_test = metrics(data["test"], base_test)
    # Report test at the dev-chosen threshold (no peeking).
    def at_th(rows, scores, t):
        rows, scores = _flip(rows, scores)
        tp = sum(1 for r, s in zip(rows, scores) if s >= t and r["label"]); fp = sum(1 for r, s in zip(rows, scores) if s >= t and not r["label"])
        fn = sum(1 for r, s in zip(rows, scores) if s < t and r["label"])
        p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        return {"precision": round(p, 4), "recall": round(rc, 4), "f1": round(2 * p * rc / max(p + rc, 1e-9), 4)}
    report = {"name": NAME, "base": BASE, "reported_class": REPORT_CLASS, "train_pairs": len(train), "epochs": EPOCHS, "lr": LR, "batch": BATCH,
              "base_model": {"dev": m_base_dev, "test_auc": m_base_test["auc"], "test_at_base_dev_threshold": at_th(data["test"], base_test, m_base_dev["threshold"])},
              "fine_tuned": {"dev": m_dev, "test_auc": m_test["auc"], "test_at_dev_threshold": at_th(data["test"], tuned_test, th)},
              "threshold": th, "seconds": round(time.time() - t0, 1),
              "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=1))  # metrics first, so a save problem can't lose them
    print("REPORT " + json.dumps(report), flush=True)
    # transformers 5 refuses to save a config holding generation parameters; the reranker's remote code carries
    # temperature=1.0, which the class default restores on load.
    temp = getattr(model.config, "temperature", None)
    if temp is not None:
        del model.config.temperature
    model.to(torch.bfloat16).save_pretrained(OUT)
    if temp is not None:
        model.config.temperature = temp
    tok.save_pretrained(OUT)
    print("saved", sorted(p.name for p in OUT.iterdir()), flush=True)


if __name__ == "__main__":
    main()

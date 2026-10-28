"""Copy a training job's report (metrics.json next to the weights in the bucket) from our Nebius endpoint into the repo.
Usage: python train/report.py <model-name> <out.json>   (needs ENDPOINT_URL, and ENDPOINT_TOKEN if the endpoint has auth)"""
import json
import os
import sys
import urllib.request


def fetch(name: str) -> dict:
    req = urllib.request.Request(os.environ["ENDPOINT_URL"].rstrip("/") + f"/metrics/{name}")
    if os.environ.get("ENDPOINT_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['ENDPOINT_TOKEN']}")
    d = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    if d.get("name") != name or "fine_tuned" not in d or "base_model" not in d:
        raise SystemExit(f"unexpected report for {name}: {str(d)[:300]}")
    return d


if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    d = fetch(name)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(d, f, indent=1)
    print(f"{name}: {d['train_pairs']} training pairs, {d['epochs']} epochs on {d['gpu']} ({d['seconds']:.0f} s); "
          f"test AUC {d['base_model']['test_auc']} before fine-tuning, {d['fine_tuned']['test_auc']} after; "
          f"F1 at the dev threshold {d['base_model']['test_at_base_dev_threshold']['f1']} -> {d['fine_tuned']['test_at_dev_threshold']['f1']}")

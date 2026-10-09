"""Download public reference data and record exact source hashes."""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Casework-data-reference/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = ROOT / "data/sources.json"
    if frozen.exists():
        manifest = json.loads(frozen.read_text(encoding="utf-8"))
        for target, source in manifest.items():
            data = get(source["url"])
            if "archive_sha256" in source:
                if hashlib.sha256(data).hexdigest() != source["archive_sha256"]:
                    raise ValueError("Source archive hash changed")
                z = zipfile.ZipFile(io.BytesIO(data))
                data = z.read(next(n for n in z.namelist() if n.endswith("hour.csv")))
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"Source content hash changed: {target}")
            (OUT / target).write_bytes(data)
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Downloaded and verified frozen data sources")
        return
    manifest = {}
    for repo, files in {
        "IBM/telco-customer-churn-on-icp4d": [("data/Telco-Customer-Churn.csv", "telco.csv"), ("LICENSE", "telco-LICENSE")],
        "PolyAI-LDN/task-specific-datasets": [("banking_data/train.csv", "banking_train.csv"), ("banking_data/test.csv", "banking_test.csv"), ("LICENSE", "banking-LICENSE")],
    }.items():
        sha = json.loads(get(f"https://api.github.com/repos/{repo}/commits/master"))["sha"]
        for source, target in files:
            url = f"https://raw.githubusercontent.com/{repo}/{sha}/{source}"
            data = get(url); (OUT / target).write_bytes(data)
            manifest[target] = {"url": url, "repository_commit": sha, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    data = get(url)
    z = zipfile.ZipFile(io.BytesIO(data))
    filename = next(n for n in z.namelist() if n.endswith("hour.csv"))
    content = z.read(filename); (OUT / "bike_hour.csv").write_bytes(content)
    manifest["bike_hour.csv"] = {"url": url, "archive_sha256": hashlib.sha256(data).hexdigest(), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    frozen.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v["bytes"] for k, v in manifest.items()}, indent=2))


if __name__ == "__main__":
    main()

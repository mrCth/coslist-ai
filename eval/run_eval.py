"""Evaluation harness for the CosList AI service.

Sends every image in ``eval/samples/`` to a *running* ``/api/analyze`` endpoint
and reports JSON-validity, retry rate, latency percentiles and how often
deliberately-bad photos trigger warnings. It also writes a manual rubric CSV
and a Markdown report.

Usage::

    # terminal 1
    uvicorn app.main:app --host 127.0.0.1 --port 8000

    # terminal 2
    python eval/run_eval.py --base-url http://127.0.0.1:8000

If ``eval/samples/`` is empty the script generates 6 *synthetic* placeholder
images (including one blurry and one very dark) so the pipeline can be
exercised. Synthetic images are **not** a meaningful evaluation - real photos
are required for a real result. This is stated in the generated report.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
SAMPLES_DIR = HERE / "samples"
RUBRIC_CSV = HERE / "rubric.csv"
RESULTS_MD = HERE / "results.md"

SYNTHETIC = [
    ("01_synthetic_dress.jpg", "normal"),
    ("02_synthetic_wig.jpg", "normal"),
    ("03_synthetic_prop.jpg", "normal"),
    ("04_synthetic_top.jpg", "normal"),
    ("05_synthetic_blurry.jpg", "blurry"),
    ("06_synthetic_dark.jpg", "dark"),
]


# --------------------------------------------------------------------------- #
# Synthetic image generation (only when the samples folder is empty)
# --------------------------------------------------------------------------- #
def _draw_item(kind: str) -> Image.Image:
    img = Image.new("RGB", (768, 768), (238, 236, 250))
    d = ImageDraw.Draw(img)
    if kind == "dress":
        d.polygon([(384, 120), (250, 300), (300, 680), (468, 680), (518, 300)], fill=(150, 60, 190))
        d.rectangle((330, 120, 438, 170), fill=(90, 40, 130))
    elif kind == "wig":
        d.ellipse((220, 150, 548, 560), fill=(60, 40, 30))
        d.ellipse((300, 300, 468, 520), fill=(235, 220, 200))
    elif kind == "prop":
        d.rectangle((260, 260, 508, 620), fill=(70, 70, 90))
        d.line((260, 440, 508, 440), fill=(200, 200, 220), width=8)
    else:  # top
        d.rectangle((240, 200, 528, 560), fill=(210, 80, 130))
        d.rectangle((180, 200, 240, 340), fill=(210, 80, 130))
        d.rectangle((528, 200, 588, 340), fill=(210, 80, 130))
    d.text((24, 24), "synthetic placeholder - NOT a real product photo", fill=(20, 20, 20))
    return img


def generate_synthetic() -> List[Dict]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    kinds = ["dress", "wig", "prop", "top", "dress", "dress"]
    made = []
    for (name, tag), kind in zip(SYNTHETIC, kinds):
        img = _draw_item(kind)
        if tag == "blurry":
            img = img.filter(ImageFilter.GaussianBlur(radius=9))
        elif tag == "dark":
            img = img.point(lambda p: int(p * 0.12))
        path = SAMPLES_DIR / name
        img.save(path, quality=88)
        made.append({"path": path, "tag": tag, "synthetic": True})
    return made


def collect_samples() -> tuple[List[Dict], bool]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    real = sorted(
        p for p in SAMPLES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if real:
        samples = []
        any_synthetic = False
        for p in real:
            tag = "normal"
            low = p.name.lower()
            synthetic = "synthetic" in low
            any_synthetic = any_synthetic or synthetic
            if "blur" in low:
                tag = "blurry"
            elif "dark" in low:
                tag = "dark"
            samples.append({"path": p, "tag": tag, "synthetic": synthetic})
        return samples, any_synthetic
    return generate_synthetic(), True


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def analyze_one(client: httpx.Client, base_url: str, path: Path) -> Dict:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    started = time.perf_counter()
    try:
        with path.open("rb") as fh:
            files = {"images": (path.name, fh.read(), mime)}
        res = client.post(
            f"{base_url}/api/analyze",
            files=files,
            data={"notes": "สินค้ามือสอง สภาพตามภาพ"},
        )
    except httpx.HTTPError as exc:
        return {
            "ok": False, "status": None, "latency_ms": int((time.perf_counter() - started) * 1000),
            "retries": None, "warnings": 0, "error": str(exc),
        }
    latency = int((time.perf_counter() - started) * 1000)
    if res.status_code != 200:
        try:
            detail = res.json().get("detail", "")
        except Exception:
            detail = res.text[:200]
        return {
            "ok": False, "status": res.status_code, "latency_ms": latency,
            "retries": None, "warnings": 0, "error": detail,
        }
    body = res.json()
    meta = body.get("meta") or {}
    return {
        "ok": True,
        "status": 200,
        "latency_ms": latency,
        "server_latency_ms": meta.get("latency_ms"),
        "retries": meta.get("retries", 0),
        "warnings": len(body.get("warnings") or []),
        "title": body.get("title_th", ""),
        "grade": body.get("condition_grade"),
        "category": body.get("category"),
        "price_min": (body.get("suggested_price_thb") or {}).get("min"),
        "price_max": (body.get("suggested_price_thb") or {}).get("max"),
        "tags": body.get("tags") or [],
        "error": None,
    }


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def write_rubric(rows: List[Dict]) -> None:
    with RUBRIC_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "sample", "tag", "ai_title", "ai_grade", "ai_price_min", "ai_price_max",
            "title_accuracy_1_5", "grade_plausibility_1_5", "hallucination_yes_no", "notes",
        ])
        for r in rows:
            writer.writerow([
                r["name"], r["tag"], r.get("title", ""), r.get("grade", ""),
                r.get("price_min", ""), r.get("price_max", ""),
                "", "", "", "",
            ])


def write_results(rows: List[Dict], synthetic: bool, base_url: str) -> Dict:
    ok = [r for r in rows if r["ok"]]
    n = len(rows)
    latencies = [r["latency_ms"] for r in ok]
    retried = [r for r in ok if (r.get("retries") or 0) > 0]
    bad = [r for r in rows if r["tag"] in {"blurry", "dark"}]
    bad_warned = [r for r in bad if r["ok"] and r["warnings"] > 0]

    json_validity = len(ok) / n if n else 0.0
    retry_rate = len(retried) / len(ok) if ok else 0.0
    bad_warn_rate = len(bad_warned) / len(bad) if bad else 0.0

    lines: List[str] = []
    lines.append("# CosList AI - Evaluation Results\n")
    lines.append(f"- Endpoint: `{base_url}/api/analyze`")
    lines.append(f"- Samples evaluated: **{n}**")
    lines.append(f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if synthetic:
        lines.append(
            "> **NOTE - synthetic images.** `eval/samples/` was empty, so the harness generated "
            "6 placeholder images (4 normal, 1 blurry, 1 very dark). These are simple drawings, "
            "**not real product photos**, so the numbers below only prove the pipeline works "
            "end-to-end. **Real photos are required for a meaningful evaluation.** "
            "Drop your own photos into `eval/samples/` and re-run.\n"
        )

    lines.append("## Automated metrics\n")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| JSON validity rate (HTTP 200 + schema-valid) | {json_validity:.0%} |")
    lines.append(f"| Retry rate (repair prompt used) | {retry_rate:.0%} |")
    lines.append(f"| Median latency | {statistics.median(latencies):.0f} ms |" if latencies else "| Median latency | n/a |")
    lines.append(f"| p95 latency | {percentile(latencies, 95):.0f} ms |" if latencies else "| p95 latency | n/a |")
    lines.append(f"| Warning rate on deliberately bad photos | {bad_warn_rate:.0%} ({len(bad_warned)}/{len(bad)}) |")
    lines.append("")

    lines.append("## Per-sample results\n")
    lines.append("| Sample | Tag | OK | Retries | Latency (ms) | Grade | Price range | Warnings |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        price = ""
        if r["ok"] and r.get("price_min") is not None:
            price = f"{r['price_min']}-{r['price_max']}"
        lines.append(
            f"| {r['name']} | {r['tag']} | {'yes' if r['ok'] else 'no'} | "
            f"{r.get('retries', '-')} | {r['latency_ms']} | {r.get('grade', '-') or '-'} | "
            f"{price or '-'} | {r.get('warnings', '-')} |"
        )
    lines.append("")

    failures = [r for r in rows if not r["ok"]]
    if failures:
        lines.append("## Failures\n")
        for r in failures:
            lines.append(f"- `{r['name']}` (tag={r['tag']}): status={r['status']} - {r['error']}")
        lines.append("")

    lines.append("## Limitations (be honest)\n")
    lines.append("- Vision-language models can mis-describe garments, colours and materials; every field is a guess from a photo.")
    lines.append("- Price is a heuristic estimate only - **not** market data. Treat it as a starting point.")
    lines.append("- The model never verifies brand or authenticity and is instructed not to claim it.")
    lines.append("- Latency on a CPU/partial-GPU laptop is high (tens of seconds); a small quantised 4B model is used.")
    lines.append("- The synthetic samples here are simple drawings; real-world accuracy needs real photos (see rubric.csv).")
    lines.append("- `rubric.csv` must be scored by a human; automated metrics do not measure correctness, only validity/latency/robustness.")
    lines.append("")

    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    return {
        "json_validity": json_validity,
        "retry_rate": retry_rate,
        "median_latency": statistics.median(latencies) if latencies else None,
        "p95_latency": percentile(latencies, 95) if latencies else None,
        "bad_warn_rate": bad_warn_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CosList AI evaluation harness")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    samples, synthetic = collect_samples()
    print(f"[eval] {len(samples)} samples ({'synthetic placeholders' if synthetic else 'real photos'})")

    rows: List[Dict] = []
    with httpx.Client(timeout=args.timeout) as client:
        try:
            health = client.get(f"{args.base_url}/health")
            health.raise_for_status()
            print(f"[eval] server: {health.json().get('provider')} / {health.json().get('model')}")
        except httpx.HTTPError as exc:
            print(f"[eval] ERROR: cannot reach {args.base_url} ({exc}). Start uvicorn first.")
            raise SystemExit(2)

        for i, sample in enumerate(samples, 1):
            result = analyze_one(client, args.base_url, sample["path"])
            result["name"] = sample["path"].name
            result["tag"] = sample["tag"]
            rows.append(result)
            state = "ok" if result["ok"] else f"FAIL({result['status']})"
            print(f"[eval] {i}/{len(samples)} {sample['path'].name:32s} {state:10s} {result['latency_ms']} ms")

    write_rubric(rows)
    summary = write_results(rows, synthetic, args.base_url)

    print("\n[eval] summary")
    print(f"  JSON validity : {summary['json_validity']:.0%}")
    print(f"  retry rate    : {summary['retry_rate']:.0%}")
    if summary["median_latency"] is not None:
        print(f"  median latency: {summary['median_latency']:.0f} ms")
        print(f"  p95 latency   : {summary['p95_latency']:.0f} ms")
    print(f"  bad-photo warning rate: {summary['bad_warn_rate']:.0%}")
    print(f"\n[eval] wrote {RESULTS_MD.name} and {RUBRIC_CSV.name}")


if __name__ == "__main__":
    main()

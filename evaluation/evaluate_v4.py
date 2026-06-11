#!/usr/bin/env python3
"""
evaluate_v4.py — ПРИЁМКА v4 против v3, ЯБЛОКО-К-ЯБЛОКУ. Один протокол для обеих
моделей: КАЛИБРОВАННАЯ p (изотоника версии) + LENGTH-AWARE вердиктный порог версии
(база для >=cutoff слов, строже short для <cutoff). gpt4/saiga — НЕТРОНУТЫЙ hold-out.

Срезы:
  - test_self: FPR на test-human (length-aware), recall ИИ на test-AI;
  - holdout per-generator recall: gpt4, saiga (бенчи v3), llama, mistral (новые);
  - короткие 30/40/50 слов: FPR (test-human) и recall (все holdout-AI);
  - docx: test_diploma (FPR на абзацах прозы — должен быть 0), pure_ai/ai_edited/mixed
    (recall — доля абзацев прозы с p>=вердикт).
Регистрируем v4 в реестре transformer_score В РАНТАЙМЕ (прод-файл не трогаем).
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from pathlib import Path

import numpy as np

from aidetector import transformer_score as TF
from aidetector.document_scan import _WORD_RE
from aidetector.paths import model_path

# регистрируем v4 в рантайме — прод transformer_score.py НЕ редактируем
TF.REGISTRY["v4"] = dict(model=model_path("v4_model"), calib=model_path("v4_calibrator.joblib"),
                         thr=model_path("threshold_v4.json"))

VERSIONS = [v for v in ("v3", "v4") if os.path.exists(TF.REGISTRY[v]["model"])]
SHORT_LENS = [50, 40, 30]
DOCX = {
    "test_diploma": ("samples/test_diploma.docx", "human"),
    "pure_ai": ("samples/test_pure_ai.docx", "ai"),
    "pure_ai_ghost": ("samples/test_pure_ai_ghost.docx", "ai"),
    "ai_edited": ("samples/test_ai_edited.docx", "ai"),
    "mixed": ("samples/test_mixed.docx", "mixed"),
}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def wc(t):
    return len(_WORD_RE.findall(t))


def trunc(t, n):
    sp = [(m.start(), m.end()) for m in _WORD_RE.finditer(t)]
    return t.strip() if len(sp) <= n else t[:sp[n - 1][1]].strip()


def flag_rate(sc, texts):
    """Доля текстов с КАЛИБРОВАННОЙ p >= LENGTH-AWARE порогом (вердикт версии)."""
    if not texts:
        return float("nan"), 0
    p = sc.proba(texts)
    thr = np.array([sc.threshold_for(wc(t)) for t in texts])
    return float((p >= thr).mean()), len(texts)


def flag_rate_flat(sc, texts):
    """Доля с p >= БАЗОВЫМ порогом (без строгого short-порога) — так считались
    цитируемые v3-бенчи gpt4=0.74/saiga=0.83, по ним заданы пороги приёмки 0.70/0.78."""
    if not texts:
        return float("nan"), 0
    p = sc.proba(texts)
    return float((p >= sc.threshold()).mean()), len(texts)


def flag_rate_fixed_len(sc, texts, N):
    """Урезаем до N слов, затем length-aware порог по факт. длине (N< cutoff -> short)."""
    tt = [trunc(t, N) for t in texts]
    return flag_rate(sc, tt)


def scan_docx(version, path):
    """Сканируем docx трансформером версии (как show_flagged): окна -> p -> абзацы.
    Возвращает список p_ai абзацев прозы (с вердиктом, p не None) и базовый порог."""
    from aidetector import document_scan as ds
    from aidetector.extract_text import extract_text
    sc = TF.get(version)
    verdict_thr = sc.threshold()           # база (длинные окна -> длинная проза)
    text, _ = extract_text(path)
    paras = ds.split_paragraphs(text)
    split = ds.classify_sections(paras)
    stream, spans = ds._build_content_stream(split.content)
    wins = ds.build_windows(stream)
    if wins:
        ps = sc.proba([stream[w.cstart:w.cend] for w in wins])
        for w, p in zip(wins, ps):
            w.p_ai = float(p)
    res = ds.project_to_paragraphs(split.content, spans, wins, verdict_thr)
    verdicted = [r.p_ai for r in res if r.p_ai is not None]
    n_flag = sum(1 for p in verdicted if p >= verdict_thr)
    return dict(verdict_thr=round(verdict_thr, 4), n_prose=len(verdicted),
                n_flagged=n_flag, flag_rate=round(n_flag / len(verdicted), 4) if verdicted else None)


def main():
    print(f"[versions] {VERSIONS}\n", flush=True)

    test = load("data/test.jsonl")
    test_h = [r["text"] for r in test if r["label"] == 0]
    test_ai = [r["text"] for r in test if r["label"] == 1]
    test_ai_long = [r["text"] for r in test if r["label"] == 1 and len(r["text"]) >= 200]
    holdouts = {}
    for name, fp in [("gpt4", "data/holdout_gpt4.jsonl"), ("saiga", "data/holdout_saiga.jsonl"),
                     ("llama", "data/holdout_llama.jsonl"), ("mistral", "data/holdout_mistral.jsonl")]:
        if os.path.exists(fp):
            holdouts[name] = [r["text"] for r in load(fp)]

    results = {}
    for ver in VERSIONS:
        sc = TF.get(ver)
        r = {"thresholds": {"base": sc.threshold(), "short": sc._thr_short, "cutoff": sc._cutoff}}
        # test_self
        fpr, n = flag_rate(sc, test_h)
        rec_all, _ = flag_rate(sc, test_ai)
        rec_long, _ = flag_rate(sc, test_ai_long)
        r["test_self"] = dict(fpr=round(fpr, 4), n_human=n, recall_ai=round(rec_all, 4),
                              recall_ai_long=round(rec_long, 4), n_ai=len(test_ai))
        # holdouts full-length, length-aware
        r["holdout"] = {}
        for name, txts in holdouts.items():
            rec, nn = flag_rate(sc, txts)            # honest deploy (length-aware)
            rec_flat, _ = flag_rate_flat(sc, txts)   # flat base — для критериев приёмки
            r["holdout"][name] = dict(recall=round(rec, 4), recall_flat=round(rec_flat, 4),
                                      n=nn, med_wc=int(np.median([wc(t) for t in txts])))
        # короткие длины: FPR (test-human) + recall (объединённые holdout-AI)
        all_ai = sum(holdouts.values(), [])
        r["short"] = {}
        for N in SHORT_LENS:
            fN, _ = flag_rate_fixed_len(sc, test_h, N)
            rN, _ = flag_rate_fixed_len(sc, all_ai, N)
            # отдельно невиданные новые
            rl = flag_rate_fixed_len(sc, holdouts.get("llama", []), N)[0] if "llama" in holdouts else None
            rm = flag_rate_fixed_len(sc, holdouts.get("mistral", []), N)[0] if "mistral" in holdouts else None
            r["short"][N] = dict(fpr=round(fN, 4), recall_all=round(rN, 4),
                                 recall_llama=None if rl is None else round(rl, 4),
                                 recall_mistral=None if rm is None else round(rm, 4))
        # docx
        r["docx"] = {}
        for name, (path, kind) in DOCX.items():
            if os.path.exists(path):
                try:
                    r["docx"][name] = {**scan_docx(ver, path), "kind": kind}
                except Exception as e:
                    r["docx"][name] = {"error": str(e), "kind": kind}
        results[ver] = r
        print(f"[{ver}] готов", flush=True)

    json.dump(results, open("data/v4_eval.json", "w"), ensure_ascii=False, indent=2)

    # ---------- печать ----------
    def g(ver, *keys, default=None):
        d = results.get(ver, {})
        for k in keys:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d if d != {} else default

    print("\n" + "=" * 78)
    print("  ПРИЁМКА v4 vs v3 (калиброванная p, length-aware порог версии)")
    print("=" * 78)
    print(f"\n  {'срез':<26} {'v3':>12} {'v4':>12}   крит.")
    print("  " + "-" * 64)

    def row(label, v3v, v4v, crit=""):
        s3 = f"{v3v:.3f}" if isinstance(v3v, float) and v3v == v3v else str(v3v)
        s4 = f"{v4v:.3f}" if isinstance(v4v, float) and v4v == v4v else str(v4v)
        print(f"  {label:<26} {s3:>12} {s4:>12}   {crit}")

    row("test_self FPR", g("v3", "test_self", "fpr"), g("v4", "test_self", "fpr"), "<=0.03")
    row("test_self recall ИИ", g("v3", "test_self", "recall_ai"), g("v4", "test_self", "recall_ai"))
    row("test_self recall длинные", g("v3", "test_self", "recall_ai_long"), g("v4", "test_self", "recall_ai_long"))
    for name in ("gpt4", "saiga", "llama", "mistral"):
        c = {"gpt4": ">=0.70", "saiga": ">=0.78", "llama": ">=0.50", "mistral": ">=0.50"}[name]
        row(f"holdout {name} (flat)", g("v3", "holdout", name, "recall_flat"),
            g("v4", "holdout", name, "recall_flat"), c + " (крит.)")
        row(f"holdout {name} (len-aware)", g("v3", "holdout", name, "recall"),
            g("v4", "holdout", name, "recall"), "honest deploy")
    print("  " + "-" * 64)
    print("  короткие (recall объединённых holdout / FPR test-human):")
    for N in SHORT_LENS:
        row(f"  {N}w recall_all", g("v3", "short", N, "recall_all"), g("v4", "short", N, "recall_all"))
        row(f"  {N}w FPR", g("v3", "short", N, "fpr"), g("v4", "short", N, "fpr"), "<=0.03")
        row(f"  {N}w recall_llama", g("v3", "short", N, "recall_llama"), g("v4", "short", N, "recall_llama"))
        row(f"  {N}w recall_mistral", g("v3", "short", N, "recall_mistral"), g("v4", "short", N, "recall_mistral"))
    print("  " + "-" * 64)
    print("  docx (доля абзацев прозы с p>=вердикт):")
    for name in DOCX:
        d3 = g("v3", "docx", name) or {}
        d4 = g("v4", "docx", name) or {}
        f3 = d3.get("flag_rate"); f4 = d4.get("flag_rate")
        nf3 = d3.get("n_flagged"); nf4 = d4.get("n_flagged")
        np3 = d3.get("n_prose"); np4 = d4.get("n_prose")
        crit = "FPR=0" if name == "test_diploma" else ""
        print(f"  {name:<26} {str(nf3)+'/'+str(np3):>12} {str(nf4)+'/'+str(np4):>12}   {crit}  (flagged/prose)")

    # ---------- вердикт ----------
    print("\n" + "=" * 78)
    if "v4" in results:
        v4 = results["v4"]
        # критерии приёмки — по FLAT-порогу (как заданы пороги 0.70/0.78 от v3 0.74/0.83)
        checks = {
            "test_self FPR<=3%": v4["test_self"]["fpr"] <= 0.03 + 1e-9,
            "holdout_gpt4>=0.70": v4["holdout"].get("gpt4", {}).get("recall_flat", 0) >= 0.70,
            "holdout_saiga>=0.78": v4["holdout"].get("saiga", {}).get("recall_flat", 0) >= 0.78,
            "holdout_llama>=0.50": v4["holdout"].get("llama", {}).get("recall_flat", 0) >= 0.50,
            "holdout_mistral>=0.50": v4["holdout"].get("mistral", {}).get("recall_flat", 0) >= 0.50,
            "test_diploma FPR=0": (v4.get("docx", {}).get("test_diploma", {}).get("n_flagged", 1) == 0),
        }
        for k, ok in checks.items():
            print(f"   [{'OK ' if ok else 'FAIL'}] {k}")
        verdict = "ПРИЁМКА ПРОЙДЕНА -> v4 В ПРОД" if all(checks.values()) else "ОТКАЗ -> v4 НЕ в прод, v3 остаётся боевой"
        print(f"\n   ВЕРДИКТ: {verdict}")
        results["_verdict"] = {"checks": checks, "passed": all(checks.values())}
        json.dump(results, open("data/v4_eval.json", "w"), ensure_ascii=False, indent=2)
    print("=" * 78)
    print("\n[сохранено] data/v4_eval.json")


if __name__ == "__main__":
    main()

// Тонкий клиент к POST /detect — никакой ML-логики на фронте, только отрисовка.
"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  text: $("text"), model: $("model"), run: $("run"),
  loading: $("loading"), error: $("error"), result: $("result"),
  verdict: $("verdict"), prob: $("prob"), barFill: $("barFill"),
  barThr: $("barThr"), meta: $("meta"), note: $("note"), features: $("features"),
};

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function fail(msg) {
  hide(els.loading);
  hide(els.result);
  els.error.textContent = msg;
  show(els.error);
}

function render(data) {
  hide(els.loading);
  hide(els.error);

  // v1 без обученной модели возвращает ai_probability = -1
  if (typeof data.ai_probability !== "number" || data.ai_probability < 0) {
    fail(data.note || "Модель недоступна.");
    return;
  }

  const ai = data.verdict && data.verdict.includes("ИИ");
  const pct = (data.ai_probability * 100).toFixed(1);
  const thrPct = (data.threshold * 100).toFixed(1);

  els.verdict.textContent = data.verdict;
  els.verdict.className = "verdict " + (ai ? "verdict-ai" : "verdict-human");

  els.prob.textContent = pct + "% — вероятность ИИ";

  els.barFill.className = "bar-fill " + (ai ? "ai" : "human");
  els.barFill.style.width = pct + "%";
  els.barThr.style.left = thrPct + "%";
  els.barThr.title = "порог " + thrPct + "%";

  els.meta.textContent = `модель: ${data.model} · порог ИИ: ${thrPct}%`;
  els.note.textContent = data.note || "";

  els.features.replaceChildren();   // очистка без innerHTML
  const feats = data.features || {};
  for (const k of Object.keys(feats)) {
    const tr = document.createElement("tr");
    const td1 = document.createElement("td");
    const td2 = document.createElement("td");
    td1.textContent = k;
    td2.textContent = feats[k];
    tr.appendChild(td1);
    tr.appendChild(td2);
    els.features.appendChild(tr);
  }

  show(els.result);
}

async function detect() {
  const text = els.text.value.trim();
  if (!text) { fail("Введите текст для проверки."); return; }

  const model = els.model.value;
  hide(els.error);
  hide(els.result);
  show(els.loading);
  els.run.disabled = true;

  try {
    const res = await fetch("/detect?model=" + encodeURIComponent(model), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      let detail = "Ошибка " + res.status;
      try { const j = await res.json(); if (j.detail) detail += ": " + JSON.stringify(j.detail); } catch (e) {}
      fail(detail);
      return;
    }
    render(await res.json());
  } catch (e) {
    fail("Сеть недоступна или сервер не отвечает.");
  } finally {
    els.run.disabled = false;
  }
}

els.run.addEventListener("click", detect);
// Ctrl/Cmd+Enter в textarea — быстрый запуск
els.text.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") detect();
});

// ===== Проверка документа (.docx/.pdf/.txt) -> карта по абзацам + отчёт =====
const doc = {
  file: $("docFile"), run: $("scanRun"), loading: $("scanLoading"),
  error: $("scanError"), result: $("scanResult"), title: $("scanTitle"),
  meta: $("scanMeta"), summary: $("scanSummary"),
  dlHtml: $("dlHtml"), dlMd: $("dlMd"),
};
let lastReport = null;   // {filename, report_html, report_md}

const SUM_CLASS = {
  "подозрительно": "verdict-ai", "неопределённо": "sum-uncertain",
  "человек": "verdict-human", "недостаточно текста": "sum-dim",
};

function docFail(msg) {
  hide(doc.loading); hide(doc.result);
  doc.error.textContent = msg; show(doc.error);
}

function renderScan(data) {
  hide(doc.loading); hide(doc.error);
  if (data.error) { docFail(data.error); return; }
  lastReport = data;

  doc.title.textContent = data.filename || "Документ";
  doc.title.className = "verdict";
  doc.meta.textContent =
    `модель: ${data.model} · абзацев: ${data.n_total} · служебных отсеяно: ${data.n_dropped}`;

  doc.summary.replaceChildren();
  (data.summary || []).forEach((row) => {
    if (!row.n && !row.chars) return;
    const li = document.createElement("li");
    const chip = document.createElement("span");
    chip.className = "chip " + (SUM_CLASS[row.cat] || "sum-dim");
    chip.textContent = row.cat;
    const txt = document.createElement("span");
    txt.textContent = ` ${(row.share * 100).toFixed(1)}% (${row.n} абз., ${row.chars} симв)`;
    li.appendChild(chip); li.appendChild(txt);
    doc.summary.appendChild(li);
  });

  show(doc.result);
}

function downloadReport(kind) {
  if (!lastReport) return;
  const isHtml = kind === "html";
  const content = isHtml ? lastReport.report_html : lastReport.report_md;
  if (!content) return;
  const base = (lastReport.filename || "отчёт").replace(/\.[^.]+$/, "");
  const blob = new Blob([content], {
    type: isHtml ? "text/html;charset=utf-8" : "text/markdown;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${base}.report.${isHtml ? "html" : "md"}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function scanDoc() {
  const f = doc.file.files && doc.file.files[0];
  if (!f) { docFail("Выберите файл .docx, .pdf или .txt."); return; }

  hide(doc.error); hide(doc.result); show(doc.loading);
  doc.run.disabled = true;
  try {
    const fd = new FormData();
    fd.append("file", f);
    const res = await fetch("/scan?model=" + encodeURIComponent(els.model.value), {
      method: "POST", body: fd,
    });
    const data = await res.json().catch(() => ({ error: "Ошибка " + res.status }));
    renderScan(data);
  } catch (e) {
    docFail("Сеть недоступна или сервер не отвечает.");
  } finally {
    doc.run.disabled = false;
  }
}

doc.run.addEventListener("click", scanDoc);
doc.dlHtml.addEventListener("click", () => downloadReport("html"));
doc.dlMd.addEventListener("click", () => downloadReport("md"));

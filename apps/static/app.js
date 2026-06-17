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

  els.verdict.textContent = (ai ? "🔴 " : "🟢 ") + data.verdict;
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

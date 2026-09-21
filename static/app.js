const messageEl = document.querySelector("#message");
const charCount = document.querySelector("#char-count");
const singleForm = document.querySelector("#single-form");
const scanBtn = document.querySelector("#scan-btn");
const clearBtn = document.querySelector("#clear-btn");
const samplesEl = document.querySelector("#samples");
const emptyState = document.querySelector("#empty-state");
const errorState = document.querySelector("#error-state");
const verdictEl = document.querySelector("#verdict");
const resultCard = document.querySelector(".result");
const singleStatus = document.querySelector("#single-status");
const historyList = document.querySelector("#history-list");
const HISTORY_KEY = "scam-detector-history";
let latestResult = null;
let latestBatch = null;

function setCharCount() {
  charCount.textContent = `${messageEl.value.length} / 5000`;
}

function showError(target, message) {
  target.hidden = false;
  target.textContent = message;
}

function hideError(target) {
  target.hidden = true;
  target.textContent = "";
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    if (typeof detail === "string") throw new Error(detail);
    if (Array.isArray(detail) && detail[0]?.msg) throw new Error(detail[0].msg);
    throw new Error("Request failed.");
  }
  return data;
}

function renderHighlighted(text, highlights) {
  const sorted = [...(highlights || [])].sort((a, b) => a.start - b.start);
  let html = "";
  let cursor = 0;
  for (const span of sorted) {
    const start = Math.max(0, Math.min(span.start, text.length));
    const end = Math.max(start, Math.min(span.end, text.length));
    if (start < cursor) continue;
    html += escapeHtml(text.slice(cursor, start));
    html += `<mark title="${escapeHtml(span.term)}">${escapeHtml(text.slice(start, end))}</mark>`;
    cursor = end;
  }
  html += escapeHtml(text.slice(cursor));
  return html || escapeHtml(text);
}

function renderSignals(signals) {
  const list = document.querySelector("#signals");
  list.innerHTML = "";
  if (!signals.length) {
    list.innerHTML = "<li>No standout phrases for this short text.</li>";
    return;
  }
  for (const item of signals) {
    const li = document.createElement("li");
    li.textContent = item.term;
    list.appendChild(li);
  }
}

function renderCues(cues) {
  const list = document.querySelector("#cues");
  list.innerHTML = "";
  for (const cue of cues || []) {
    const li = document.createElement("li");
    li.textContent = cue.label;
    list.appendChild(li);
  }
}

function resetFeedbackUi() {
  document.querySelector("#fb-status").textContent = "";
  document.querySelector("#fb-correct").hidden = true;
}

function renderVerdict(result) {
  latestResult = result;
  emptyState.hidden = true;
  hideError(errorState);
  verdictEl.hidden = false;
  resultCard.classList.toggle("is-scam", result.prediction === "Scam");
  resultCard.classList.toggle("is-ok", result.prediction === "Legitimate");
  document.querySelector("#verdict-label").textContent =
    result.prediction === "Scam" ? "Likely scam" : "Looks legitimate";
  document.querySelector("#verdict-title").textContent = result.prediction;
  document.querySelector("#verdict-advice").textContent = result.advice;
  document.querySelector("#highlight-quote").innerHTML = renderHighlighted(
    result.original_text,
    result.highlights,
  );
  const pct = Math.round(result.scam_probability * 100);
  document.querySelector("#scam-pct").textContent = `${pct}%`;
  document.querySelector("#scam-bar").style.width = `${pct}%`;
  renderCues(result.cues);
  renderSignals(result.signals || []);
  resetFeedbackUi();
}

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function writeHistory(items) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 8)));
}

function renderHistory() {
  const items = readHistory();
  historyList.innerHTML = "";
  if (!items.length) {
    historyList.innerHTML =
      '<li class="muted">Scans from this browser stay here until you clear them.</li>';
    return;
  }
  for (const item of items) {
    const li = document.createElement("li");
    const pill = document.createElement("span");
    pill.className = `pill ${item.prediction === "Scam" ? "scam" : "ok"}`;
    pill.textContent = item.prediction;
    const snippet = document.createElement("span");
    snippet.className = "snippet";
    snippet.textContent = item.original_text;
    const conf = document.createElement("span");
    conf.className = "muted";
    conf.textContent = `${Math.round(item.confidence_score * 100)}%`;
    li.append(pill, snippet, conf);
    li.addEventListener("click", () => {
      messageEl.value = item.original_text;
      setCharCount();
      renderVerdict(item);
      showTab("single");
    });
    historyList.appendChild(li);
  }
}

function pushHistory(result) {
  const next = [
    result,
    ...readHistory().filter((item) => item.original_text !== result.original_text),
  ];
  writeHistory(next);
  renderHistory();
}

async function loadSamples() {
  try {
    const samples = await fetchJson("/api/v1/examples");
    samplesEl.innerHTML = "";
    for (const sample of samples) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "chip";
      button.textContent = sample.title;
      button.addEventListener("click", () => {
        messageEl.value = sample.text;
        setCharCount();
        messageEl.focus();
      });
      samplesEl.appendChild(button);
    }
  } catch (error) {
    samplesEl.innerHTML = "";
    singleStatus.hidden = false;
    singleStatus.textContent = error.message;
  }
}

async function sendFeedback(actual) {
  if (!latestResult) return;
  const body = await fetchJson("/api/v1/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: latestResult.original_text,
      predicted: latestResult.prediction,
      actual,
    }),
  });
  document.querySelector("#fb-correct").hidden = true;
  document.querySelector("#fb-status").textContent = body.record.correct
    ? "Saved: you agreed."
    : "Saved: marked as a miss.";
  renderFeedback(body.summary);
}

async function loadReview() {
  try {
    const stats = await fetchJson("/api/v1/model/stats");
    document.querySelector("#stat-size").textContent = stats.corpus_size;
    document.querySelector("#stat-acc").textContent = `${Math.round(stats.accuracy * 100)}%`;
    document.querySelector("#stat-recall").textContent = `${Math.round(stats.scam_recall * 100)}%`;
    document.querySelector("#stats-note").textContent = stats.note;
  } catch (error) {
    document.querySelector("#stats-note").textContent = error.message;
  }
  try {
    renderFeedback(await fetchJson("/api/v1/feedback"));
  } catch {
    /* empty store is fine */
  }
}

function renderFeedback(summary) {
  const list = document.querySelector("#feedback-list");
  const line = document.querySelector("#feedback-summary");
  if (!summary || !summary.count) {
    line.textContent = "Agree or correct a verdict after you scan.";
    list.innerHTML = '<li class="muted">No feedback yet.</li>';
    return;
  }
  const rate = summary.agreement_rate == null ? "—" : `${Math.round(summary.agreement_rate * 100)}%`;
  line.textContent = `${summary.count} marks · ${rate} agreement`;
  list.innerHTML = "";
  for (const item of summary.recent || []) {
    const li = document.createElement("li");
    const pill = document.createElement("span");
    pill.className = `pill ${item.correct ? "ok" : "scam"}`;
    pill.textContent = item.correct ? "Agreed" : "Corrected";
    const snippet = document.createElement("span");
    snippet.className = "snippet";
    snippet.textContent = `${item.predicted} → ${item.actual}: ${item.text}`;
    li.append(pill, snippet);
    list.appendChild(li);
  }
}

function showTab(name) {
  const tabs = {
    single: document.querySelector("#tab-single"),
    batch: document.querySelector("#tab-batch"),
    review: document.querySelector("#tab-review"),
  };
  const panels = {
    single: document.querySelector("#panel-single"),
    batch: document.querySelector("#panel-batch"),
    review: document.querySelector("#panel-review"),
  };
  for (const key of Object.keys(tabs)) {
    const on = key === name;
    tabs[key].classList.toggle("is-active", on);
    tabs[key].setAttribute("aria-selected", on ? "true" : "false");
    panels[key].hidden = !on;
  }
  if (name === "review") loadReview();
}

singleForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError(errorState);
  scanBtn.disabled = true;
  scanBtn.textContent = "Scanning…";
  try {
    const result = await fetchJson("/api/v1/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: messageEl.value }),
    });
    renderVerdict(result);
    pushHistory(result);
  } catch (error) {
    emptyState.hidden = true;
    verdictEl.hidden = true;
    showError(errorState, error.message);
  } finally {
    scanBtn.disabled = false;
    scanBtn.textContent = "Scan message";
  }
});

clearBtn.addEventListener("click", () => {
  messageEl.value = "";
  setCharCount();
  emptyState.hidden = false;
  verdictEl.hidden = true;
  hideError(errorState);
  resultCard.classList.remove("is-scam", "is-ok");
  latestResult = null;
});

document.querySelector("#clear-history").addEventListener("click", () => {
  writeHistory([]);
  renderHistory();
});

document.querySelector("#fb-yes").addEventListener("click", () => {
  if (!latestResult) return;
  sendFeedback(latestResult.prediction);
});

document.querySelector("#fb-no").addEventListener("click", () => {
  document.querySelector("#fb-correct").hidden = false;
  document.querySelector("#fb-status").textContent = "Choose the label you would give it.";
});

document.querySelector("#fb-scam").addEventListener("click", () => sendFeedback("Scam"));
document.querySelector("#fb-legit").addEventListener("click", () => sendFeedback("Legitimate"));

messageEl.addEventListener("input", setCharCount);

document.querySelector("#tab-single").addEventListener("click", () => showTab("single"));
document.querySelector("#tab-batch").addEventListener("click", () => showTab("batch"));
document.querySelector("#tab-review").addEventListener("click", () => showTab("review"));

document.querySelector("#batch-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const batchError = document.querySelector("#batch-error");
  const table = document.querySelector("#batch-table");
  const empty = document.querySelector("#batch-empty");
  const summary = document.querySelector("#batch-summary");
  hideError(batchError);
  const lines = document
    .querySelector("#batch-text")
    .value.split(/\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 50);
  if (!lines.length) {
    showError(batchError, "Add at least one non-empty line.");
    return;
  }
  try {
    const payload = await fetchJson("/api/v1/detect/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: lines.map((text) => ({ text })) }),
    });
    latestBatch = payload;
    empty.hidden = true;
    table.hidden = false;
    summary.hidden = false;
    document.querySelector("#download-csv").hidden = false;
    summary.textContent = `${payload.scam_count} scam · ${payload.legitimate_count} legitimate`;
    const tbody = table.querySelector("tbody");
    tbody.innerHTML = "";
    for (const row of payload.results) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="pill ${row.prediction === "Scam" ? "scam" : "ok"}">${row.prediction}</span></td>
        <td>${Math.round(row.confidence_score * 100)}%</td>
        <td>${escapeHtml(row.original_text)}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (error) {
    showError(batchError, error.message);
  }
});

document.querySelector("#download-csv").addEventListener("click", async () => {
  const lines = document
    .querySelector("#batch-text")
    .value.split(/\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 50);
  if (!lines.length) return;
  const response = await fetch("/api/v1/detect/batch.csv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: lines.map((text) => ({ text })) }),
  });
  if (!response.ok) return;
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "scam-scan.csv";
  link.click();
  URL.revokeObjectURL(url);
});

setCharCount();
loadSamples();
renderHistory();

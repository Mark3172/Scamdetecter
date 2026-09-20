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

function renderVerdict(result) {
  emptyState.hidden = true;
  hideError(errorState);
  verdictEl.hidden = false;
  resultCard.classList.toggle("is-scam", result.prediction === "Scam");
  resultCard.classList.toggle("is-ok", result.prediction === "Legitimate");
  document.querySelector("#verdict-label").textContent =
    result.prediction === "Scam" ? "Likely scam" : "Looks legitimate";
  document.querySelector("#verdict-title").textContent = result.prediction;
  document.querySelector("#verdict-advice").textContent = result.advice;
  const pct = Math.round(result.scam_probability * 100);
  document.querySelector("#scam-pct").textContent = `${pct}%`;
  document.querySelector("#scam-bar").style.width = `${pct}%`;
  renderSignals(result.signals || []);
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
});

document.querySelector("#clear-history").addEventListener("click", () => {
  writeHistory([]);
  renderHistory();
});

messageEl.addEventListener("input", setCharCount);

document.querySelector("#tab-single").addEventListener("click", () => {
  document.querySelector("#panel-single").hidden = false;
  document.querySelector("#panel-batch").hidden = true;
  document.querySelector("#tab-single").classList.add("is-active");
  document.querySelector("#tab-batch").classList.remove("is-active");
  document.querySelector("#tab-single").setAttribute("aria-selected", "true");
  document.querySelector("#tab-batch").setAttribute("aria-selected", "false");
});

document.querySelector("#tab-batch").addEventListener("click", () => {
  document.querySelector("#panel-single").hidden = true;
  document.querySelector("#panel-batch").hidden = false;
  document.querySelector("#tab-batch").classList.add("is-active");
  document.querySelector("#tab-single").classList.remove("is-active");
  document.querySelector("#tab-batch").setAttribute("aria-selected", "true");
  document.querySelector("#tab-single").setAttribute("aria-selected", "false");
});

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
    empty.hidden = true;
    table.hidden = false;
    summary.hidden = false;
    summary.textContent = `${payload.scam_count} scam · ${payload.legitimate_count} legitimate`;
    const tbody = table.querySelector("tbody");
    tbody.innerHTML = "";
    for (const row of payload.results) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="pill ${row.prediction === "Scam" ? "scam" : "ok"}">${row.prediction}</span></td>
        <td>${Math.round(row.confidence_score * 100)}%</td>
        <td>${row.original_text.replace(/[<>&]/g, (ch) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[ch]))}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (error) {
    showError(batchError, error.message);
  }
});

setCharCount();
loadSamples();
renderHistory();

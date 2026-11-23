"use strict";

const COLORS = {
  "Groceries": "#3a8f5c", "Dining": "#d9822b", "Fuel & Transport": "#4a6fb5", "Health & Pharmacy": "#c2484d",
  "Shopping": "#9b59b6", "Electronics": "#2a9bb0", "Entertainment": "#d4a72c", "Bills & Utilities": "#6b7a8f",
  "Uncategorized": "#a0a4ab",
};
const inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 2 });
const inrShort = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", notation: "compact", maximumFractionDigits: 1 });
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = { month: "", categories: [], editing: null, mode: null, knownMonths: new Set() };

// ------------------------------------------------------------------ api

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}
const jsonBody = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function toast(msg, isError = false) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("error", isError);
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), isError ? 5000 : 3200);
}

// ------------------------------------------------------------------ rendering

function monthLabel(m) {
  const [y, mo] = m.split("-").map(Number);
  return new Date(y, mo - 1, 1).toLocaleDateString("en-IN", { month: "short", year: "numeric" });
}

function shortMonth(m) {
  const [y, mo] = m.split("-").map(Number);
  return `${new Date(y, mo - 1, 1).toLocaleDateString("en-IN", { month: "short" }).slice(0, 3)}<small>'${String(y).slice(2)}</small>`;
}

function renderMonthPicker(byMonth) {
  byMonth.forEach((m) => m.month && state.knownMonths.add(m.month));
  const now = new Date();
  state.knownMonths.add(`${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`);
  const months = [...state.knownMonths].sort().reverse();
  const sel = $("month");
  sel.innerHTML = `<option value="">All time</option>` +
    months.map((m) => `<option value="${m}">${monthLabel(m)}</option>`).join("");
  sel.value = state.month;
}

function renderTiles(s) {
  $("t-spent").textContent = inr.format(s.total);
  $("t-count").textContent = s.count;
  $("t-review").textContent = s.needs_review;
  document.querySelector(".warn-tile").classList.toggle("has", s.needs_review > 0);
  $("t-shops").textContent = s.model.known_shops.toLocaleString("en-IN");
  $("t-retrain").textContent = `retrains in ${s.model.corrections_until_retrain} correction${s.model.corrections_until_retrain === 1 ? "" : "s"}`;
}

function renderCategoryChart(byCat) {
  const el = $("cat-chart");
  if (!byCat.length) { el.innerHTML = `<p class="chart-empty">Nothing in this period yet.</p>`; return; }
  const max = Math.max(...byCat.map((c) => c.total), 1);
  el.innerHTML = byCat.map((c) => `
    <div class="hbar" data-cat="${esc(c.category)}" title="${esc(c.category)}: ${c.count} receipt(s)">
      <span class="chip"><span class="dot" style="background:${COLORS[c.category] || COLORS.Uncategorized}"></span>${esc(c.category)}</span>
      <span class="track"><span class="fill" style="width:${(c.total / max) * 100}%;background:${COLORS[c.category] || COLORS.Uncategorized}"></span></span>
      <span class="val">${inr.format(c.total)}</span>
    </div>`).join("");
  el.querySelectorAll(".hbar").forEach((b) => b.addEventListener("click", () => {
    const cat = b.dataset.cat === "Uncategorized" ? "" : b.dataset.cat;
    $("f-cat").value = $("f-cat").value === cat ? "" : cat;
    loadExpenses();
  }));
}

function renderMonthChart(byMonth) {
  const el = $("month-chart");
  if (!byMonth.length) { el.innerHTML = `<p class="chart-empty">Monthly totals appear once receipts have dates.</p>`; return; }
  const max = Math.max(...byMonth.map((m) => m.total), 1);
  el.innerHTML = byMonth.map((m) => `
    <div class="vbar ${m.month === state.month ? "active" : ""}" data-month="${m.month}" title="${monthLabel(m.month)}: ${inr.format(m.total)} · ${m.count} receipt(s)">
      <span class="amt">${inrShort.format(m.total)}</span>
      <span class="col" style="height:${Math.max((m.total / max) * 120, 2)}px"></span>
      <span class="lbl">${shortMonth(m.month)}</span>
    </div>`).join("");
  el.querySelectorAll(".vbar").forEach((b) => b.addEventListener("click", () => {
    state.month = state.month === b.dataset.month ? "" : b.dataset.month;
    $("month").value = state.month;
    refresh();
  }));
}

function renderRows(rows) {
  $("empty").hidden = rows.length > 0;
  $("rows").innerHTML = rows.map((e) => {
    const cat = e.category || "Uncategorized";
    const date = e.date ? new Date(e.date + "T00:00").toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "2-digit" }) : `<span class="muted">no date</span>`;
    return `<tr data-id="${e.id}">
      <td class="date-cell">${date}</td>
      <td><div class="shop">${esc(e.merchant) || `<span class="muted">unknown shop</span>`}</div>
          <div class="items-preview">${esc(e.items.join(", "))}</div></td>
      <td title="${esc(cat)}"><span class="chip"><span class="dot" style="background:${COLORS[cat]}"></span><span class="cat-name">${esc(cat)}</span></span></td>
      <td class="num">${e.total != null ? inr.format(e.total) : `<span class="muted">—</span>`}</td>
      <td>${e.needs_review ? `<span class="flag">review</span>` : ""}</td>
    </tr>`;
  }).join("");
  $("rows").querySelectorAll("tr").forEach((tr) => tr.addEventListener("click", async () => {
    try { openEditor(await api(`/api/expenses/${tr.dataset.id}`), "edit"); } catch (e) { toast(e.message, true); }
  }));
}

// ------------------------------------------------------------------ loading

async function loadExpenses() {
  const p = new URLSearchParams();
  if (state.month) p.set("month", state.month);
  if ($("f-cat").value) p.set("category", $("f-cat").value);
  if ($("f-review").checked) p.set("needs_review", "true");
  if ($("q").value.trim()) p.set("q", $("q").value.trim());
  renderRows(await api(`/api/expenses?${p}`));
}

async function refresh() {
  try {
    const [summary, all] = await Promise.all([
      api(`/api/summary${state.month ? `?month=${state.month}` : ""}`),
      api(`/api/summary`),
    ]);
    renderMonthPicker(all.by_month);
    renderTiles(summary);
    renderCategoryChart(summary.by_category);
    renderMonthChart(all.by_month);
    await loadExpenses();
  } catch (e) {
    toast(`Could not load data: ${e.message}`, true);
  }
}

// ------------------------------------------------------------------ uploading

async function uploadFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) { toast("Please choose an image file.", true); return; }
  const fd = new FormData();
  fd.append("file", file, file.name || "receipt.jpg");
  $("busy").hidden = false;
  try {
    const expense = await api("/api/receipts", { method: "POST", body: fd });
    await refresh();
    openEditor(expense, "upload");
  } catch (e) {
    toast(e.message, true);
  } finally {
    $("busy").hidden = true;
    $("file").value = "";
  }
}

async function trySample() {
  const btn = $("sample-btn");
  btn.disabled = true;
  try {
    const res = await fetch(`/api/sample?level=${$("sample-level").value}`);
    if (!res.ok) throw new Error("could not make a sample receipt");
    const blob = await res.blob();
    await uploadFile(new File([blob], "sample.jpg", { type: "image/jpeg" }));
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------------ editor

function categoryReason(e) {
  if (!e.category) return "The shop name could not be read, so pick a category. Your choice teaches the model.";
  const pct = e.confidence != null ? `${Math.round(e.confidence * 100)}%` : "";
  switch (e.source) {
    case "rule": return "Known shop: matched from the shop list.";
    case "ml": return e.needs_review
      ? `Model's best guess (${pct} confident, below the review threshold). Confirm or correct it.`
      : `Predicted by the model (${pct} confident).`;
    case "user": return "Set by you.";
    case "manual": return "Entered by hand.";
    default: return "";
  }
}

function openEditor(e, mode) {
  state.editing = e;
  state.mode = mode;
  const isNew = mode === "manual";
  $("edit-title").textContent = mode === "upload" ? "Check what was read" : isNew ? "Add an expense" : "Edit expense";
  $("e-merchant").value = e.merchant || "";
  $("e-date").value = e.date || "";
  $("e-total").value = e.total ?? "";
  $("e-items").value = (e.items || []).join("\n");

  const sel = $("e-category");
  sel.innerHTML = (e.category ? "" : `<option value="" disabled selected>${isNew ? "Let the model decide" : "Choose a category…"}</option>`) +
    state.categories.map((c) => `<option ${c === e.category ? "selected" : ""}>${esc(c)}</option>`).join("");
  if (isNew) { sel.required = false; sel.querySelector("option").disabled = false; } else { sel.required = true; }
  $("e-why").textContent = isNew ? "Leave it to the model, or choose one yourself." : categoryReason(e);

  const notes = [...(e.warnings || [])];
  if (e.needs_review && !isNew) notes.unshift("Please check the category: the system isn't sure about it.");
  const n = $("notice");
  if (mode === "upload" && !notes.length) {
    n.className = "notice ok"; n.hidden = false;
    n.textContent = "All fields found. Compare them with the photo before saving: OCR can misread blurry or creased bills.";
  } else if (notes.length) {
    n.className = "notice"; n.hidden = false;
    n.innerHTML = `<strong>Check these:</strong><ul>${notes.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>`;
  } else n.hidden = true;

  const hasImg = !!e.has_image;
  $("preview").hidden = !hasImg;
  document.querySelector(".dlg-body").classList.toggle("no-image", !hasImg);
  if (hasImg) $("preview-img").src = `/api/expenses/${e.id}/image?v=${e.id}`;

  $("ocr-box").hidden = !(e.ocr_text && e.ocr_text.length);
  $("ocr-box").open = false;
  $("ocr-text").textContent = (e.ocr_text || []).join("\n");
  $("timing").textContent = e.timings_ms ? `Processed in ${(e.timings_ms.total / 1000).toFixed(2)} s ` +
    `(clean-up ${e.timings_ms.preprocess} ms, OCR ${e.timings_ms.ocr} ms)` : "";

  $("delete-btn").hidden = isNew;
  $("delete-btn").textContent = mode === "upload" ? "Discard" : "Delete";
  $("save-btn").textContent = isNew ? "Add" : mode === "upload" ? "Save" : "Save changes";
  $("editor").showModal();
  $(isNew ? "e-merchant" : "save-btn").focus();
}

function readForm() {
  const items = $("e-items").value.split("\n").map((s) => s.trim()).filter(Boolean);
  const total = $("e-total").value;
  return {
    merchant: $("e-merchant").value.trim(),
    date: $("e-date").value || null,
    total: total === "" ? null : Number(total),
    items,
    category: $("e-category").value || null,
  };
}

async function saveEditor(ev) {
  ev.preventDefault();
  const f = readForm();
  if (!f.merchant) { toast("Please enter the shop name.", true); return; }
  const btn = $("save-btn");
  btn.disabled = true;
  try {
    if (state.mode === "manual") {
      const body = { ...f };
      if (!body.category) delete body.category;
      const e = await api("/api/expenses", jsonBody("POST", body));
      toast(f.category ? `Added. Learned “${e.merchant}” → ${e.category}.` : `Added as ${e.category}.`);
    } else {
      const old = state.editing;
      const body = { merchant: f.merchant, date: f.date, total: f.total, items: f.items };
      // A changed category is a correction; confirming a flagged one is feedback too.
      if (f.category && (f.category !== old.category || old.needs_review)) body.category = f.category;
      const e = await api(`/api/expenses/${old.id}`, jsonBody("PATCH", body));
      if (e.retrained) toast(`Saved. The model was retrained with your corrections.`);
      else if (e.learned) toast(`Saved. Learned “${e.merchant}” → ${e.category}.`);
      else toast("Saved.");
    }
    $("editor").close();
    await refresh();
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
  }
}

async function deleteEditing() {
  const e = state.editing;
  if (state.mode !== "upload" && !confirm(`Delete ${e.merchant || "this expense"}?`)) return;
  try {
    await api(`/api/expenses/${e.id}`, { method: "DELETE" });
    $("editor").close();
    toast(state.mode === "upload" ? "Discarded." : "Deleted.");
    await refresh();
  } catch (err) { toast(err.message, true); }
}

// ------------------------------------------------------------------ wiring

async function init() {
  state.categories = await api("/api/categories");
  $("f-cat").innerHTML += state.categories.map((c) => `<option>${esc(c)}</option>`).join("");

  const drop = $("drop");
  $("file").addEventListener("change", (e) => uploadFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => uploadFile(e.dataTransfer.files[0]));
  document.addEventListener("paste", (e) => {
    const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
    if (f && !$("editor").open) uploadFile(f);
  });

  $("sample-btn").addEventListener("click", trySample);
  $("manual-btn").addEventListener("click", () => openEditor({ items: [], date: new Date().toISOString().slice(0, 10) }, "manual"));
  $("month").addEventListener("change", (e) => { state.month = e.target.value; refresh(); });
  let qt;
  $("q").addEventListener("input", () => { clearTimeout(qt); qt = setTimeout(loadExpenses, 200); });
  $("f-cat").addEventListener("change", loadExpenses);
  $("f-review").addEventListener("change", loadExpenses);
  document.querySelector(".warn-tile").addEventListener("click", () => {
    $("f-review").checked = true; loadExpenses();
    document.querySelector(".list-head").scrollIntoView({ behavior: "smooth" });
  });

  $("edit-form").addEventListener("submit", saveEditor);
  $("close-btn").addEventListener("click", () => $("editor").close());
  $("cancel-btn").addEventListener("click", () => $("editor").close());
  $("delete-btn").addEventListener("click", deleteEditing);

  await refresh();
}

init().catch((e) => toast(`Could not start: ${e.message}`, true));

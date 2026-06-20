"use strict";

const $ = (id) => document.getElementById(id);

// Live value labels for the sliders.
const bindRange = (id, labelId, fmt = (v) => v) => {
  const el = $(id), label = $(labelId);
  const update = () => (label.textContent = fmt(el.value));
  el.addEventListener("input", update);
  update();
};
bindRange("length", "lenVal");
bindRange("temperature", "tempVal", (v) => Number(v).toFixed(2));
bindRange("topk", "topkVal", (v) => (Number(v) === 0 ? "off" : v));
bindRange("tempo", "tempoVal");
bindRange("cstr", "cstrVal", (v) => Number(v).toFixed(1));

const msg = $("msg");
const setMsg = (text, isError = false) => {
  msg.textContent = text || "";
  msg.classList.toggle("error", isError);
};

// --- view switching (sidebar nav) ---
const VIEW_REFRESH = {
  training: () => { loadModels(); ensureKeepersChip(); },
  generations: () => loadLibrary(),
  mymidi: () => loadMyMidi(),
  corpus: () => { if (!corpusWorks.length) loadComposers(); },
  constraints: () => loadConstraints(),
  counterpoint: () => initCounterpoint(),
};

// --- counterpoint generator ---
const KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
function fillKeyOptions(sel) {
  for (const mode of ["major", "minor"]) {
    for (const n of KEY_NAMES) {
      const o = document.createElement("option");
      o.value = `${n} ${mode}`;
      o.textContent = `${n} ${mode}`;
      sel.appendChild(o);
    }
  }
}

function initCounterpoint() {
  const sel = $("cpKey");
  if (sel.dataset.loaded) return;
  fillKeyOptions(sel);
  sel.value = "C major";
  sel.dataset.loaded = "1";
}

async function generateCounterpoint() {
  const btn = $("cpBtn");
  btn.disabled = true;
  $("cpStatus").textContent = "Generating counterpoint…";
  try {
    const info = await window.pywebview.api.generate_counterpoint({
      key: $("cpKey").value,
      length: Number($("cpLength").value) || 10,
      above: $("cpVoice").value === "above",
      species: Number($("cpSpecies").value) || 1,
    });
    if (info.error) {
      $("cpStatus").textContent = "Failed: " + info.error;
      return;
    }
    showView("generate");
    $("player").src = info.data_uri;
    $("trackName").textContent = `${info.name} · counterpoint`;
    $("player-wrap").classList.remove("hidden");
    $("genEmpty").style.display = "none";
    renderSheet(info.musicxml);
    loadLibrary();
    setMsg("Counterpoint generated (cantus firmus + counter-voice).");
  } catch (e) {
    $("cpStatus").textContent = "Failed: " + (e && e.message ? e.message : e);
  } finally {
    btn.disabled = false;
  }
}
$("cpBtn").addEventListener("click", generateCounterpoint);

async function generateFugue() {
  const btn = $("fugueBtn");
  btn.disabled = true;
  $("fugueStatus").textContent = "Generating subject + tonal answer…";
  try {
    const info = await window.pywebview.api.generate_fugue({
      key: $("cpKey").value,
      length: Number($("cpLength").value) || 8,
      voices: Number($("fugueVoices").value) || 4,
    });
    if (info.error) {
      $("fugueStatus").textContent = "Failed: " + info.error;
      return;
    }
    showView("generate");
    $("player").src = info.data_uri;
    $("trackName").textContent = `${info.name} · fugue exposition`;
    $("player-wrap").classList.remove("hidden");
    $("genEmpty").style.display = "none";
    renderSheet(info.musicxml);
    loadLibrary();
    setMsg("Fugue exposition: subject + tonal answer + countersubject.");
    $("fugueStatus").textContent = "idle";
  } catch (e) {
    $("fugueStatus").textContent = "Failed: " + (e && e.message ? e.message : e);
  } finally {
    btn.disabled = false;
  }
}
$("fugueBtn").addEventListener("click", generateFugue);

// --- voice-leading constraint rules ---
function enabledRules() {
  return [...document.querySelectorAll("#rulesList .rule-check:checked")]
    .map((c) => c.dataset.rule);
}

function updateConstraintBadge() {
  const n = enabledRules().length;
  $("cstrBadge").textContent = n ? `${n} constraint${n > 1 ? "s" : ""} on` : "no constraints";
}

function populateKeys() {
  const sel = $("keySelect");
  if (sel.dataset.loaded) return;
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  for (const mode of ["major", "minor"]) {
    for (const n of names) {
      const o = document.createElement("option");
      o.value = `${n} ${mode}`;
      o.textContent = `${n} ${mode}`;
      sel.appendChild(o);
    }
  }
  sel.dataset.loaded = "1";
}

async function loadConstraints() {
  populateKeys();
  const ul = $("rulesList");
  if (ul.dataset.loaded) { updateConstraintBadge(); return; }
  try {
    const res = await window.pywebview.api.list_constraints();
    ul.innerHTML = "";
    for (const r of res.rules) {
      const li = document.createElement("li");
      li.className = "rule-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.className = "rule-check";
      cb.dataset.rule = r.id;
      cb.checked = true;
      cb.addEventListener("change", updateConstraintBadge);
      const txt = document.createElement("div");
      const name = document.createElement("div");
      name.className = "rule-name";
      name.textContent = r.name;
      const desc = document.createElement("div");
      desc.className = "rule-desc";
      desc.textContent = r.desc;
      txt.append(name, desc);
      li.append(cb, txt);
      li.addEventListener("click", (e) => {
        if (e.target !== cb) { cb.checked = !cb.checked; updateConstraintBadge(); }
      });
      ul.appendChild(li);
    }
    ul.dataset.loaded = "1";
    updateConstraintBadge();
  } catch (e) {
    ul.innerHTML = "<li class='lib-empty'>constraints unavailable</li>";
  }
}
function showView(name) {
  document.querySelectorAll(".view").forEach((v) =>
    v.classList.toggle("active", v.id === "view-" + name));
  document.querySelectorAll(".nav-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === name));
  if (VIEW_REFRESH[name]) VIEW_REFRESH[name]();
}
document.querySelectorAll(".nav-btn").forEach((b) =>
  b.addEventListener("click", () => showView(b.dataset.view)));

// pywebview injects the bridge asynchronously; wait for it.
function whenApiReady() {
  return new Promise((resolve) => {
    if (window.pywebview && window.pywebview.api) return resolve();
    window.addEventListener("pywebviewready", () => resolve(), { once: true });
  });
}

async function refreshStatus() {
  const el = $("status");
  try {
    const s = await window.pywebview.api.status();
    if (s.trained) {
      el.textContent = `Generating with: ${s.active_model_name}`;
      el.classList.remove("warn");
    } else {
      el.textContent = "No model selected — train one below (generating now sounds random).";
      el.classList.add("warn");
    }
  } catch (e) {
    el.textContent = "Bridge unavailable (open via the app, not a browser).";
  }
}

async function generate() {
  const btn = $("generate");
  btn.disabled = true;
  showView("generate");
  setMsg("Generating…");
  const seedRaw = $("seed").value.trim();
  const params = {
    length: Number($("length").value),
    temperature: Number($("temperature").value),
    top_k: Number($("topk").value),
    tempo: Number($("tempo").value),
    seed: seedRaw === "" ? null : Number(seedRaw),
    constraints: enabledRules(),
    constraint_strength: Number($("cstr").value),
    cadence: $("cadence").value,
    key: $("keySelect").value,
  };
  try {
    const info = await window.pywebview.api.generate(params);
    const player = $("player");
    player.src = info.data_uri;
    $("trackName").textContent =
      `${info.name} · ${info.n_notes} notes` +
      (info.trained ? "" : " · ⚠ untrained model");
    $("player-wrap").classList.remove("hidden");
    $("genEmpty").style.display = "none";
    renderSheet(info.musicxml);
    loadLibrary();
    setMsg(info.trained ? "Done." : "Done (untrained — train the model for musical results).");
  } catch (e) {
    setMsg("Generation failed: " + (e && e.message ? e.message : e), true);
  } finally {
    btn.disabled = false;
  }
}

// --- staff notation (OpenSheetMusicDisplay) ---
let osmd = null;
async function renderSheet(musicxml) {
  const el = $("sheet");
  if (!musicxml || typeof opensheetmusicdisplay === "undefined") {
    el.classList.add("hidden");
    return;
  }
  try {
    if (!osmd) {
      osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(el, {
        autoResize: true,
        drawTitle: false,
        drawPartNames: false,
        followCursor: true,   // auto-scroll the panel to keep the cursor in view
        backend: "svg",
      });
    }
    el.classList.remove("hidden");
    await osmd.load(musicxml);
    osmd.render();
    setupScoreFollow();
  } catch (e) {
    el.classList.add("hidden");
    console.error("sheet render failed:", e);
  }
}

// --- follow-along: move the OSMD cursor + colour notes in sync with playback ---
let followRAF = null;
let rawSteps = [];         // each cursor step's ABSOLUTE musical position (unitless)
let cursorSteps = [];      // rawSteps converted to seconds (calibrated at play time)
let coloredEls = [];       // SVG notehead elements currently tinted
let followIdx = -1;        // step index the OSMD cursor is currently sitting on
let followGen = 0;         // generation token — only the latest loop survives
const HILITE = "#7c5cff";

function setupScoreFollow() {
  const player = $("player");
  rawSteps = [];
  cursorSteps = [];
  try {
    if (!osmd || !osmd.cursor) return;
    osmd.cursor.reset();
    let guard = 0;
    while (!osmd.cursor.iterator.EndReached && guard++ < 100000) {
      rawSteps.push(osmd.cursor.iterator.currentTimeStamp.RealValue);
      osmd.cursor.next();
    }
    osmd.cursor.reset();
    osmd.cursor.hide();
  } catch (e) {
    console.error("cursor setup failed:", e);
    rawSteps = [];
    return;
  }
  if (!player._followWired) {
    player._followWired = true;
    player.addEventListener("start", startFollow);
    player.addEventListener("stop", stopFollow);
  }
}

// Seconds-per-musical-unit, calibrated from the audio the player will actually
// produce — so it's correct regardless of the score's notated tempo or units.
function calibrateRate(player) {
  const lastRaw = rawSteps[rawSteps.length - 1] || 0;
  try {
    const ns = player.noteSequence;
    if (ns && ns.notes && ns.notes.length && lastRaw > 0) {
      const lastOnset = Math.max(...ns.notes.map((n) => n.startTime || 0));
      if (lastOnset > 0) return lastOnset / lastRaw;
    }
  } catch (e) {}
  const bpm = (osmd.Sheet && osmd.Sheet.DefaultStartTempoInBpm)
    || Number($("tempo").value) || 120;
  return 240 / bpm;   // fallback: assume RealValue is in whole notes
}

function clearColors() {
  coloredEls.forEach((el) =>
    el.querySelectorAll("path, ellipse, rect").forEach((n) => (n.style.fill = "")));
  coloredEls = [];
}

function colorCurrentNotes() {
  clearColors();
  try {
    const gnotes = osmd.cursor.GNotesUnderCursor ? osmd.cursor.GNotesUnderCursor() : [];
    gnotes.forEach((gn) => {
      const svg = gn.getSVGGElement && gn.getSVGGElement();
      if (svg) {
        svg.querySelectorAll("path, ellipse, rect").forEach((n) => (n.style.fill = HILITE));
        coloredEls.push(svg);
      }
    });
  } catch (e) { /* notehead colouring not available — cursor bar still follows */ }
}

// Move the OSMD cursor to an exact step index (forward via next(), or
// reset()+forward if the clock rewound). Returns true if it moved.
function moveCursorTo(target) {
  target = Math.max(0, Math.min(target, cursorSteps.length - 1));
  if (target === followIdx) return false;
  if (target < followIdx) {            // clock went backwards -> restart from 0
    osmd.cursor.reset();
    followIdx = 0;
  }
  let guard = 0;
  while (followIdx < target && !osmd.cursor.iterator.EndReached && guard++ < 100000) {
    osmd.cursor.next();
    followIdx++;
  }
  followIdx = target;
  return true;
}

function startFollow() {
  if (!rawSteps.length || !osmd || !osmd.cursor) return;
  const player = $("player");
  const rate = calibrateRate(player);
  cursorSteps = rawSteps.map((r) => r * rate);

  cancelAnimationFrame(followRAF);       // kill any prior loop
  const gen = ++followGen;               // invalidate older loops
  osmd.cursor.reset();
  followIdx = 0;
  osmd.cursor.show();
  colorCurrentNotes();

  const tick = () => {
    if (gen !== followGen) return;       // a newer start() superseded us
    const t = Number(player.currentTime) || 0;
    // target = the last step whose start time has been reached (pure fn of t)
    let target = 0;
    while (target < cursorSteps.length - 1 && cursorSteps[target + 1] <= t) target++;
    if (moveCursorTo(target)) colorCurrentNotes();
    followRAF = requestAnimationFrame(tick);
  };
  followRAF = requestAnimationFrame(tick);
}

function stopFollow() {
  followGen++;                           // stop any running loop
  cancelAnimationFrame(followRAF);
  followRAF = null;
  clearColors();
  try { osmd.cursor.hide(); } catch (e) {}
}

$("generate").addEventListener("click", generate);
$("reveal").addEventListener("click", () => window.pywebview.api.reveal_output());

// --- saved-generations library ---
function mkBtn(label, title, onClick) {
  const b = document.createElement("button");
  b.textContent = label;
  b.title = title;
  if (onClick) b.addEventListener("click", (e) => { e.stopPropagation(); onClick(b); });
  return b;
}

async function loadLibrary() {
  try {
    const res = await window.pywebview.api.list_generations();
    $("libCount").textContent = res.count + " saved";
    renderLibrary(res.generations || []);
  } catch (e) {
    $("libCount").textContent = "unavailable";
  }
}

function renderLibrary(items) {
  const ul = $("libList");
  ul.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "lib-empty";
    li.textContent = "No generations yet — click Generate above.";
    ul.appendChild(li);
    return;
  }
  for (const e of items) {
    const li = document.createElement("li");

    const main = document.createElement("div");
    main.className = "lib-main";
    const nm = document.createElement("div");
    nm.className = "lib-name";
    nm.textContent = e.name;
    const meta = document.createElement("div");
    meta.className = "lib-meta";
    const p = e.params || {};
    const model = p.model_name || p.model;
    if (model) {
      const tag = document.createElement("span");
      tag.className = "lib-model";
      tag.textContent = model;
      nm.append(" ", tag);
    }
    meta.textContent =
      e.created.replace("T", " ") +
      (e.n_notes != null ? ` · ${e.n_notes} notes` : "") +
      (p.temperature != null ? ` · temp ${p.temperature}` : "") +
      (p.seed != null ? ` · seed ${p.seed}` : "");
    main.append(nm, meta);
    main.addEventListener("click", () => replayGeneration(e.file));

    const actions = document.createElement("div");
    actions.className = "lib-actions";
    const keepBtn = mkBtn(e.kept ? "★ Kept" : "☆ Keep",
      "Add to the training keepers pool", (btn) => toggleKeep(btn, e));
    if (e.kept) keepBtn.classList.add("kept-on");
    actions.append(
      mkBtn("▶", "Replay", () => replayGeneration(e.file)),
      keepBtn,
      mkBtn("Rename", "Rename", () => startRename(li, e.file, e.name)),
      mkBtn("🗑", "Delete", (btn) => confirmDelete(btn, e.file)),
    );

    li.append(main, actions);
    ul.appendChild(li);
  }
}

async function replayGeneration(file) {
  showView("generate");
  setMsg("Loading…");
  try {
    const info = await window.pywebview.api.load_generation(file);
    if (info.error) {
      setMsg("That file is gone.", true);
      showView("generations");
      loadLibrary();
      return;
    }
    $("player").src = info.data_uri;
    $("trackName").textContent =
      `${info.name}` + (info.n_notes != null ? ` · ${info.n_notes} notes` : "") + " · replay";
    $("player-wrap").classList.remove("hidden");
    $("genEmpty").style.display = "none";
    renderSheet(info.musicxml);
    setMsg("");
  } catch (e) {
    setMsg("Replay failed: " + (e && e.message ? e.message : e), true);
  }
}

function startRename(li, file, currentName) {
  const nameEl = li.querySelector(".lib-name");
  if (!nameEl) return;
  const input = document.createElement("input");
  input.className = "lib-name-input";
  input.value = currentName;
  let done = false;
  const commit = async (save) => {
    if (done) return;
    done = true;
    if (save && input.value.trim() && input.value.trim() !== currentName) {
      await window.pywebview.api.rename_generation(file, input.value.trim());
    }
    loadLibrary();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit(true);
    else if (e.key === "Escape") commit(false);
  });
  input.addEventListener("blur", () => commit(true));
  nameEl.replaceWith(input);
  input.focus();
  input.select();
}

function confirmDelete(btn, file) {
  if (btn.dataset.confirm === "1") {
    window.pywebview.api.delete_generation(file).then(loadLibrary);
    return;
  }
  btn.dataset.confirm = "1";
  btn.textContent = "Sure?";
  setTimeout(() => {
    btn.dataset.confirm = "0";
    btn.textContent = "🗑";
  }, 2500);
}

async function toggleKeep(btn, e) {
  btn.disabled = true;
  try {
    if (e.kept) await window.pywebview.api.unkeep_generation(e.file);
    else await window.pywebview.api.keep_generation(e.file);
  } finally {
    await loadLibrary();
    ensureKeepersChip();
  }
}

// --- My MIDI (drop folder) ---
async function loadMyMidi() {
  try {
    const res = await window.pywebview.api.list_my_midi();
    if (res.dir) $("myMidiDir").textContent = res.dir;
    $("myMidiCount").textContent = res.count + " file" + (res.count === 1 ? "" : "s");
    renderMyMidi(res.files || []);
  } catch (e) {
    $("myMidiCount").textContent = "unavailable";
  }
}

function renderMyMidi(items) {
  const ul = $("myMidiList");
  ul.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "lib-empty";
    li.textContent = "No files yet — drop .mid files into the folder above, then reopen this view.";
    ul.appendChild(li);
    return;
  }
  for (const e of items) {
    const li = document.createElement("li");
    const main = document.createElement("div");
    main.className = "lib-main";
    const nm = document.createElement("div");
    nm.className = "lib-name";
    nm.textContent = e.name;
    const meta = document.createElement("div");
    meta.className = "lib-meta";
    meta.textContent =
      e.created.replace("T", " ") +
      (e.n_notes != null ? ` · ${e.n_notes} notes` : "") +
      (e.size_kb != null ? ` · ${e.size_kb} KB` : "");
    main.append(nm, meta);
    main.addEventListener("click", () => playMyMidi(e.file));

    const actions = document.createElement("div");
    actions.className = "lib-actions";
    actions.append(
      mkBtn("▶", "Play", () => playMyMidi(e.file)),
      mkBtn("🗑", "Remove from My MIDI", (btn) => confirmDeleteMyMidi(btn, e.file)),
    );
    li.append(main, actions);
    ul.appendChild(li);
  }
}

async function playMyMidi(file) {
  showView("generate");
  setMsg("Loading…");
  try {
    const info = await window.pywebview.api.load_my_midi(file);
    if (info.error) {
      setMsg("That file is gone.", true);
      showView("mymidi");
      loadMyMidi();
      return;
    }
    $("player").src = info.data_uri;
    $("trackName").textContent =
      `${info.name}` + (info.n_notes != null ? ` · ${info.n_notes} notes` : "") + " · My MIDI";
    $("player-wrap").classList.remove("hidden");
    $("genEmpty").style.display = "none";
    renderSheet(info.musicxml);
    setMsg("");
  } catch (e) {
    setMsg("Playback failed: " + (e && e.message ? e.message : e), true);
  }
}

function confirmDeleteMyMidi(btn, file) {
  if (btn.dataset.confirm === "1") {
    window.pywebview.api.delete_my_midi(file).then(() => {
      loadMyMidi();
      ensureKeepersChip();
    });
    return;
  }
  btn.dataset.confirm = "1";
  btn.textContent = "Sure?";
  setTimeout(() => {
    btn.dataset.confirm = "0";
    btn.textContent = "🗑";
  }, 2500);
}

$("revealMyMidi").addEventListener("click", () => window.pywebview.api.reveal_my_midi());

// --- music21 corpus browser ---
let corpusWorks = [];
let currentComposer = "bach";
let selectedWorkPath = null;

function renderCorpusList(filter = "") {
  const list = $("corpusList");
  const q = filter.trim().toLowerCase();
  const items = q
    ? corpusWorks.filter((w) => w.name.toLowerCase().includes(q))
    : corpusWorks;
  list.innerHTML = "";
  for (const w of items) {
    const li = document.createElement("li");
    li.textContent = w.name;
    const fmt = document.createElement("span");
    fmt.className = "fmt";
    fmt.textContent = w.format;
    li.appendChild(fmt);
    li.addEventListener("click", () => selectWork(li, w));
    list.appendChild(li);
  }
  if (!items.length) {
    const li = document.createElement("li");
    li.textContent = "no matches";
    li.style.color = "var(--muted)";
    list.appendChild(li);
  }
}

async function selectWork(li, work) {
  document.querySelectorAll(".corpus-list li.active")
    .forEach((el) => el.classList.remove("active"));
  li.classList.add("active");
  selectedWorkPath = work.path;
  $("analyzeBtn").disabled = false;
  $("corpusAnalysis").textContent = `Click “Analyze with Mistral” to analyze ${work.name}.`;
  const raw = $("corpusRaw");
  raw.textContent = "Loading " + work.name + " …";
  const playerWrap = $("corpusPlayerWrap");
  playerWrap.classList.add("hidden");
  try {
    const info = await window.pywebview.api.corpus_raw(work.path);
    raw.textContent = info.text;
    raw.scrollTop = 0;
    if (info.midi_data_uri) {
      $("corpusPlayer").src = info.midi_data_uri;
      $("corpusPlayLabel").textContent =
        `▶ Original — ${info.name} · ${info.n_parts} parts, ${info.n_notes} notes`;
      playerWrap.classList.remove("hidden");
    }
  } catch (e) {
    raw.textContent = "Failed to load: " + (e && e.message ? e.message : e);
  }
}

const selectedComposers = new Set();

async function loadComposers() {
  const sel = $("composer");
  const picker = $("composerPicker");
  try {
    const res = await window.pywebview.api.list_composers();
    currentComposer = res.default || "bach";
    sel.innerHTML = "";
    picker.innerHTML = "";
    for (const c of res.composers) {
      const opt = document.createElement("option");
      opt.value = c.name;
      opt.textContent = `${c.name} (${c.count})`;
      if (c.name === currentComposer) opt.selected = true;
      sel.appendChild(opt);

      const chip = document.createElement("div");
      chip.className = "composer-chip";
      chip.innerHTML = `${c.name} <span class="cc-count">${c.count}</span>`;
      chip.addEventListener("click", () => {
        if (selectedComposers.has(c.name)) {
          selectedComposers.delete(c.name);
          chip.classList.remove("on");
        } else {
          selectedComposers.add(c.name);
          chip.classList.add("on");
        }
        updatePickCount();
      });
      picker.appendChild(chip);
    }
    // default the training selection to the browsed composer
    selectedComposers.clear();
    selectedComposers.add(currentComposer);
    picker.querySelectorAll(".composer-chip").forEach((ch) => {
      if (ch.textContent.trim().startsWith(currentComposer + " ")) ch.classList.add("on");
    });
    updatePickCount();
    ensureKeepersChip();
  } catch (e) {
    sel.innerHTML = "<option>corpus unavailable</option>";
    picker.textContent = "corpus unavailable";
  }
  await loadWorks(currentComposer);
}

// Prepend a "★ My keepers (N)" chip to the picker when the curated pool is
// non-empty, so kept generations can be trained on like any composer.
async function ensureKeepersChip() {
  let n = 0;
  try {
    n = (await window.pywebview.api.keepers_count()).count;
  } catch (e) {}
  const picker = $("composerPicker");
  let chip = document.getElementById("keepersChip");
  if (n === 0) {
    if (chip) chip.remove();
    selectedComposers.delete("curated");
    updatePickCount();
    return;
  }
  if (!chip) {
    chip = document.createElement("div");
    chip.id = "keepersChip";
    chip.className = "composer-chip keepers";
    chip.addEventListener("click", () => {
      if (selectedComposers.has("curated")) {
        selectedComposers.delete("curated");
        chip.classList.remove("on");
      } else {
        selectedComposers.add("curated");
        chip.classList.add("on");
      }
      updatePickCount();
    });
    picker.insertBefore(chip, picker.firstChild);
    if (selectedComposers.has("curated")) chip.classList.add("on");
  }
  chip.innerHTML = `★ My MIDI <span class="cc-count">${n}</span>`;
}

function updatePickCount() {
  const n = selectedComposers.size;
  $("pickCount").textContent =
    n === 0 ? "" : n === 1 ? "· 1 source" : `· ${n} sources (mixed)`;
}

async function loadWorks(composer) {
  const countEl = $("corpusCount");
  countEl.textContent = "loading…";
  // reset the detail view when switching composers
  $("corpusRaw").textContent = "Select a work on the left to view its raw data…";
  $("corpusPlayerWrap").classList.add("hidden");
  $("corpusFilter").value = "";
  selectedWorkPath = null;
  $("analyzeBtn").disabled = true;
  $("corpusAnalysis").textContent =
    "Select a work, then analyze — Mido extracts the notes and Mistral returns a key & interval analysis.";
  try {
    const res = await window.pywebview.api.list_corpus(composer);
    corpusWorks = res.works || [];
    countEl.textContent = res.count + " works available";
    renderCorpusList();
  } catch (e) {
    countEl.textContent = "corpus unavailable";
  }
}

$("composer").addEventListener("change", (e) => {
  currentComposer = e.target.value;
  loadWorks(currentComposer);
});

$("corpusFilter").addEventListener("input", (e) => renderCorpusList(e.target.value));

async function analyzeWork() {
  if (!selectedWorkPath) return;
  const out = $("corpusAnalysis");
  const btn = $("analyzeBtn");
  btn.disabled = true;
  out.textContent = "Extracting notes with Mido and analyzing with Mistral…";
  try {
    const question = $("mistralQuery").value;
    const info = await window.pywebview.api.analyze_corpus_work(selectedWorkPath, question);
    out.textContent = info.text || info.error || "(no response)";
  } catch (e) {
    out.textContent = "Analysis failed: " + (e && e.message ? e.message : e);
  } finally {
    btn.disabled = false;
  }
}
$("analyzeBtn").addEventListener("click", analyzeWork);

// --- training a model (one or more composers) ---
function setTrainStatus(text) {
  $("trainStatus").textContent = text;
}

function setTrainingDisabled(disabled) {
  $("trainBtn").disabled = disabled;
  $("generate").disabled = disabled;
  $("composerPicker").style.pointerEvents = disabled ? "none" : "";
  $("composerPicker").style.opacity = disabled ? "0.5" : "";
}

function applyTrainState(s) {
  setTrainStatus(s.message || s.phase);
  let pct = 0;
  if (s.phase === "importing") pct = 8;
  else if (s.epochs) pct = Math.round((s.epoch / s.epochs) * 100);
  if (s.phase === "done") pct = 100;
  if (s.phase === "error") pct = 0;
  $("progressBar").style.width = pct + "%";
}

async function pollTraining() {
  let s;
  try {
    s = await window.pywebview.api.training_status();
  } catch (e) {
    return;
  }
  applyTrainState(s);
  if (s.running) {
    setTimeout(pollTraining, 800);
  } else {
    setTrainingDisabled(false);
    refreshStatus();
    loadModels();
  }
}

async function trainModel() {
  const composers = [...selectedComposers];
  if (!composers.length) {
    setTrainStatus("pick at least one composer");
    return;
  }
  setTrainingDisabled(true);
  setTrainStatus("starting…");
  const nRaw = $("nWorks").value.trim();
  const params = { composers, epochs: Number($("epochs").value) || 30 };
  if (nRaw) params.n_works = Number(nRaw);
  const name = $("modelName").value.trim();
  if (name) params.name = name;
  try {
    const r = await window.pywebview.api.train_corpus(params);
    if (!r.started) {
      setTrainStatus(r.reason || "could not start");
      setTrainingDisabled(false);
      return;
    }
    pollTraining();
  } catch (e) {
    setTrainStatus("failed: " + (e && e.message ? e.message : e));
    setTrainingDisabled(false);
  }
}

$("trainBtn").addEventListener("click", trainModel);

// --- saved-model library ---
async function loadModels() {
  try {
    const res = await window.pywebview.api.list_models();
    renderModels(res.models || [], res.active);
  } catch (e) {
    $("activeModel").textContent = "models unavailable";
  }
}

function renderModels(items, active) {
  const ul = $("modelList");
  ul.innerHTML = "";
  $("activeModel").textContent = active
    ? `${items.length} model${items.length > 1 ? "s" : ""}`
    : "no models yet";
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "lib-empty";
    li.textContent = "No models yet — pick composers above and train one.";
    ul.appendChild(li);
    return;
  }
  for (const m of items) {
    const li = document.createElement("li");
    const isActive = m.id === active;

    const main = document.createElement("div");
    main.className = "lib-main";
    const nm = document.createElement("div");
    nm.className = "lib-name";
    nm.textContent = (isActive ? "● " : "") + m.name;
    if (isActive) nm.style.color = "var(--accent-2)";
    const meta = document.createElement("div");
    meta.className = "lib-meta";
    meta.textContent =
      m.composers.join(" + ") +
      (m.val_loss != null ? ` · val ${m.val_loss}` : "") +
      (m.epochs != null ? ` · ${m.epochs} epochs` : "") +
      " · " + m.created.replace("T", " ");
    main.append(nm, meta);
    main.addEventListener("click", () => useModel(m.id));

    const actions = document.createElement("div");
    actions.className = "lib-actions";
    const useBtn = mkBtn(isActive ? "active" : "Use", "Generate with this model",
      () => useModel(m.id));
    if (isActive) useBtn.disabled = true;
    actions.append(
      useBtn,
      mkBtn("Rename", "Rename", () => startRenameModel(li, m.id, m.name)),
      mkBtn("🗑", "Delete", (btn) => confirmDeleteModel(btn, m.id)),
    );
    li.append(main, actions);
    ul.appendChild(li);
  }
}

async function useModel(id) {
  await window.pywebview.api.set_active_model(id);
  await refreshStatus();
  loadModels();
}

function startRenameModel(li, id, currentName) {
  const nameEl = li.querySelector(".lib-name");
  if (!nameEl) return;
  const input = document.createElement("input");
  input.className = "lib-name-input";
  input.value = currentName;
  let done = false;
  const commit = async (save) => {
    if (done) return;
    done = true;
    if (save && input.value.trim() && input.value.trim() !== currentName) {
      await window.pywebview.api.rename_model(id, input.value.trim());
    }
    loadModels();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") commit(true);
    else if (e.key === "Escape") commit(false);
  });
  input.addEventListener("blur", () => commit(true));
  nameEl.replaceWith(input);
  input.focus();
  input.select();
}

function confirmDeleteModel(btn, id) {
  if (btn.dataset.confirm === "1") {
    window.pywebview.api.delete_model(id).then(() => {
      loadModels();
      refreshStatus();
    });
    return;
  }
  btn.dataset.confirm = "1";
  btn.textContent = "Sure?";
  setTimeout(() => {
    btn.dataset.confirm = "0";
    btn.textContent = "🗑";
  }, 2500);
}

showView("generate");
whenApiReady().then(() => {
  refreshStatus();
  loadLibrary();
  loadModels();
  loadComposers();
  loadConstraints();
  // reflect any training already in flight (e.g. after a reload)
  window.pywebview.api.training_status().then((s) => {
    if (s.running) {
      setTrainingDisabled(true);
      pollTraining();
      showView("training");
    }
  });
});

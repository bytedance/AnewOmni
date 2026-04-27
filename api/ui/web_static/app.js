async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function setSvg(container, svg) {
  if (svg) {
    container.innerHTML = svg;
    container.classList.remove("empty");
  } else {
    container.textContent = "No SVG available";
    container.classList.add("empty");
  }
}

function setText(container, text, fallback) {
  container.textContent = text && text.length > 0 ? text : fallback;
}

function setEditorCode(code) {
  document.getElementById("code-editor").value = code;
  lastLoadedCodeFromTask = null;
}

function renderFragmentMappings(mappings) {
  if (!mappings || mappings.length === 0) {
    return "No fragment mapping";
  }
  return mappings.map((item) => `${item.name} -> ${item.smiles}`).join("\n");
}

function renderPeptideExtras(prompt) {
  const lines = [];
  if (prompt.random_motif) {
    lines.push(`motif (random): ${prompt.random_motif}`);
    if (prompt.random_motif_positions && prompt.random_motif_positions.length > 0) {
      lines.push(`motif positions: ${prompt.random_motif_positions.join(", ")}`);
    }
  }
  if (prompt.random_nsaa_count && prompt.random_nsaa_count > 0) {
    lines.push(`nsAA (random): count=${prompt.random_nsaa_count}`);
    if (prompt.random_nsaa_positions && prompt.random_nsaa_positions.length > 0) {
      lines.push(`nsAA positions: ${prompt.random_nsaa_positions.join(", ")}`);
    }
  }
  if (prompt.cyclize_mode) {
    lines.push(`cyclize: ${prompt.cyclize_mode}`);
  }
  return lines.length > 0 ? lines.join("\n") : "No peptide extras";
}

function renderAntibodyExtras(prompt) {
  const lines = [];
  const resolved = prompt.resolved_constraints || {};
  const keys = Object.keys(resolved).sort();
  if (keys.length === 0) {
    return prompt.enforcement_note || "No antibody constraints";
  }
  for (const cdrType of keys) {
    const item = resolved[cdrType] || {};
    const motif = item.motif;
    const nsaa = item.nsaa;
    lines.push(`${cdrType}:`);
    if (motif) {
      const positions = item.motif_positions && item.motif_positions.length > 0
        ? item.motif_positions.join(", ")
        : "unresolved";
      lines.push(`  motif ${motif.seq} @ ${positions}`);
    }
    if (nsaa) {
      const positions = item.nsaa_positions && item.nsaa_positions.length > 0
        ? item.nsaa_positions.join(", ")
        : "unresolved";
      lines.push(`  nsAA x${nsaa.count} @ ${positions}`);
    }
    if (!motif && !nsaa) {
      lines.push("  none");
    }
  }
  if (prompt.enforcement_note) {
    lines.push("");
    lines.push(prompt.enforcement_note);
  }
  return lines.join("\n");
}

function renderPeptideNsaaPreviews(prompt) {
  const container = document.getElementById("peptide-nsaa-previews");
  if (!container) return;
  container.innerHTML = "";

  if (!prompt || prompt.kind !== "peptide" || !prompt.random_nsaa_previews || prompt.random_nsaa_previews.length === 0) {
    return;
  }

  prompt.random_nsaa_previews.forEach((item, index) => {
    const card = document.createElement("div");
    card.className = "preview-card";

    const title = document.createElement("div");
    title.className = "preview-title";
    const pos = prompt.random_nsaa_positions && prompt.random_nsaa_positions[index] !== undefined
      ? `pos ${prompt.random_nsaa_positions[index]}`
      : `nsAA ${index + 1}`;
    title.textContent = pos;
    card.appendChild(title);

    const smiles = document.createElement("div");
    smiles.className = "preview-smiles";
    smiles.textContent = item.smiles;
    card.appendChild(smiles);

    const svgBox = document.createElement("div");
    svgBox.className = "preview-svg";
    if (item.svg) {
      svgBox.innerHTML = item.svg;
    } else {
      svgBox.textContent = "No SVG";
    }
    card.appendChild(svgBox);

    container.appendChild(card);
  });
}

function renderAntibodyNsaaPreviews(prompt) {
  const container = document.getElementById("peptide-nsaa-previews");
  if (!container) return;
  container.innerHTML = "";
  if (!prompt || prompt.kind !== "antibody" || !prompt.nsaa_previews || prompt.nsaa_previews.length === 0) {
    return;
  }
  prompt.nsaa_previews.forEach((item) => {
    const card = document.createElement("div");
    card.className = "preview-card";

    const title = document.createElement("div");
    title.className = "preview-title";
    const pos = item.positions && item.positions.length > 0 ? item.positions.join(", ") : "unresolved";
    title.textContent = `${item.cdr_type} @ ${pos}`;
    card.appendChild(title);

    const smiles = document.createElement("div");
    smiles.className = "preview-smiles";
    smiles.textContent = item.smiles;
    card.appendChild(smiles);

    const svgBox = document.createElement("div");
    svgBox.className = "preview-svg";
    if (item.svg) {
      svgBox.innerHTML = item.svg;
    } else {
      svgBox.textContent = "No SVG";
    }
    card.appendChild(svgBox);
    container.appendChild(card);
  });
}

let lastLoadedCodeFromTask = null;
let taskSelectFrozen = false;
let resultPageIndex = 0;
let resultPageSignature = null;
let currentResultPages = [];
let cdrPageIndex = 0;
let cdrPageSignature = null;
let currentCdrPages = [];
let manualBusyMessage = null;
let programOutputBase = "";

function updateProgramOutput(logText = "") {
  const output = document.getElementById("exec-output");
  if (!output) return;
  const sections = [];
  if (programOutputBase && programOutputBase.length > 0) {
    sections.push(programOutputBase);
  }
  if (logText && logText.length > 0) {
    sections.push(`Run Log:\n${logText}`);
  }
  output.textContent = sections.length > 0 ? sections.join("\n\n") : "Ready";
}

function setProgramOutputBase(text, logText = "") {
  programOutputBase = text || "";
  updateProgramOutput(logText);
}

function setBusyState(isBusy) {
  const lock = document.getElementById("interaction-lock");
  const execBtn = document.getElementById("exec-btn");
  const runBtn = document.getElementById("run-btn");
  const resetBtn = document.getElementById("reset-btn");
  const taskSelect = document.getElementById("task-select");
  const saveDirInput = document.getElementById("save-dir-input");
  const codeEditor = document.getElementById("code-editor");
  if (lock) {
    lock.classList.toggle("hidden", !isBusy);
    lock.setAttribute("aria-hidden", isBusy ? "false" : "true");
  }
  [execBtn, runBtn, resetBtn, taskSelect, saveDirInput].forEach((button) => {
    if (button) {
      button.disabled = isBusy;
    }
  });
  if (codeEditor) {
    codeEditor.readOnly = isBusy;
  }
}

function clearPromptVisuals() {
  const promptSvg = document.getElementById("prompt-svg");
  const promptMapping = document.getElementById("prompt-mapping");
  const promptText = document.getElementById("prompt-text");
  const previews = document.getElementById("peptide-nsaa-previews");
  if (promptSvg) {
    promptSvg.textContent = "";
    promptSvg.classList.add("empty");
  }
  if (promptMapping) {
    promptMapping.textContent = "";
  }
  if (promptText) {
    promptText.textContent = "";
  }
  if (previews) {
    previews.innerHTML = "";
  }
}

function renderTasks(tasks, selectedTaskId) {
  const select = document.getElementById("task-select");
  if (!select) return;
  const items = tasks || [];
  // Avoid re-rendering the select while the user is interacting with it. Rebuilding options
  // causes the dropdown to flicker/close and forces double-click behavior.
  if (taskSelectFrozen) {
    return;
  }

  const signature = JSON.stringify(items.map((t) => [t.id, t.status, t.submitted_at]));
  if (select.dataset.signature === signature) {
    // Keep selected value synced without rebuilding the menu.
    if (select.value !== (selectedTaskId || "")) {
      select.value = selectedTaskId || "";
    }
    return;
  }
  select.dataset.signature = signature;

  select.innerHTML = "";

  const liveOption = document.createElement("option");
  liveOption.value = "";
  liveOption.textContent = "[live] current editor";
  select.appendChild(liveOption);

  for (const task of items) {
    const option = document.createElement("option");
    option.value = task.id;
    if (task.status === "demo") {
      option.textContent = `[demo] ${task.id.replace("demo:", "")}`;
    } else {
      const ts = new Date(task.submitted_at * 1000).toLocaleTimeString();
      option.textContent = `${task.id} [${task.status}] @ ${ts}`;
    }
    select.appendChild(option);
  }

  select.value = selectedTaskId || "";
}

function renderMolstar(cifUrl) {
  const frame = document.getElementById("molstar-frame");
  const desiredSrc = cifUrl
    ? `./molstar_viewer.html?cif_url=${encodeURIComponent(cifUrl)}`
    : "./molstar_viewer.html";

  // Avoid reloading the iframe on every polling tick. Reloading Mol* causes
  // the placeholder ("No CIF available") to flash.
  if (frame.dataset && frame.dataset.desiredSrc === desiredSrc) {
    return;
  }

  if (!cifUrl) {
    frame.src = desiredSrc;
    if (frame.dataset) frame.dataset.desiredSrc = desiredSrc;
    frame.classList.add("empty");
    return;
  }
  frame.src = desiredSrc;
  if (frame.dataset) frame.dataset.desiredSrc = desiredSrc;
  frame.classList.remove("empty");
}

function updateResultNav() {
  const total = Math.max(currentResultPages.length, 1);
  const prevBtn = document.getElementById("result-prev-btn");
  const nextBtn = document.getElementById("result-next-btn");
  const label = document.getElementById("result-page-label");
  if (!prevBtn || !nextBtn || !label) return;
  label.textContent = `${Math.min(resultPageIndex + 1, total)} / ${total}`;
  prevBtn.disabled = currentResultPages.length <= 1 || resultPageIndex <= 0;
  nextBtn.disabled = currentResultPages.length <= 1 || resultPageIndex >= currentResultPages.length - 1;
}

function renderResultSvg(fallbackSvg) {
  const container = document.getElementById("result-svg");
  if (currentResultPages.length > 0) {
    const pageIndex = Math.max(0, Math.min(resultPageIndex, currentResultPages.length - 1));
    setSvg(container, currentResultPages[pageIndex]);
  } else {
    setSvg(container, fallbackSvg || null);
  }
  updateResultNav();
}

function updateCdrNav() {
  const total = Math.max(currentCdrPages.length, 1);
  const prevBtn = document.getElementById("cdr-prev-btn");
  const nextBtn = document.getElementById("cdr-next-btn");
  const label = document.getElementById("cdr-page-label");
  const panel = document.getElementById("cdr-control-panel");
  if (!prevBtn || !nextBtn || !label || !panel) return;
  if (currentCdrPages.length === 0) {
    panel.style.display = "none";
    return;
  }
  panel.style.display = "block";
  label.textContent = `${Math.min(cdrPageIndex + 1, total)} / ${total}`;
  prevBtn.disabled = currentCdrPages.length <= 1 || cdrPageIndex <= 0;
  nextBtn.disabled = currentCdrPages.length <= 1 || cdrPageIndex >= currentCdrPages.length - 1;
}

function renderCdrControlSvg() {
  const panel = document.getElementById("cdr-control-panel");
  const container = document.getElementById("cdr-control-svg");
  if (!panel || !container) return;
  if (currentCdrPages.length === 0) {
    panel.style.display = "none";
    setSvg(container, null);
    updateCdrNav();
    return;
  }
  const pageIndex = Math.max(0, Math.min(cdrPageIndex, currentCdrPages.length - 1));
  setSvg(container, currentCdrPages[pageIndex].svg || null);
  updateCdrNav();
}

function renderAppState(state) {
  const promptBusyMessage = manualBusyMessage;
  const isBusy = Boolean(promptBusyMessage);
  setBusyState(isBusy);
  renderTasks(state.tasks, state.selected_task_id);

  if (isBusy) {
    clearPromptVisuals();
    currentCdrPages = [];
    cdrPageSignature = null;
    cdrPageIndex = 0;
    renderCdrControlSvg();
  } else {
    setText(document.getElementById("prompt-text"), state.state_text, "No active prompt");
    setText(
      document.getElementById("prompt-mapping"),
      state.prompt?.kind === "molecule"
        ? renderFragmentMappings(state.prompt?.fragment_mappings || [])
        : state.prompt?.kind === "peptide"
          ? renderPeptideExtras(state.prompt)
          : state.prompt?.kind === "antibody"
            ? renderAntibodyExtras(state.prompt)
          : "No mapping",
      "No mapping",
    );
    if (state.prompt?.kind === "peptide") {
      renderPeptideNsaaPreviews(state.prompt);
    } else if (state.prompt?.kind === "antibody") {
      renderAntibodyNsaaPreviews(state.prompt);
    } else {
      const container = document.getElementById("peptide-nsaa-previews");
      if (container) container.innerHTML = "";
    }
    setSvg(document.getElementById("prompt-svg"), state.prompt?.svg || null);
    const cdrPages = state.prompt?.kind === "antibody" ? (state.prompt?.cdr_control_pages || []) : [];
    const cdrSignature = JSON.stringify(cdrPages.map((item) => item.cdr_type));
    if (cdrSignature !== cdrPageSignature) {
      cdrPageSignature = cdrSignature;
      cdrPageIndex = 0;
    }
    currentCdrPages = cdrPages;
    renderCdrControlSvg();
  }

  // When a task is selected, show the task's code in the editor for reproducibility.
  const viewCode = state.view?.code || null;
  if (viewCode && viewCode !== lastLoadedCodeFromTask) {
    document.getElementById("code-editor").value = viewCode;
    lastLoadedCodeFromTask = viewCode;
  }
  if (!viewCode) {
    lastLoadedCodeFromTask = null;
  }

  const logText = state.view?.log || "";
  updateProgramOutput(logText);

  const latestResult = state.latest_result;
  if (!latestResult) {
    document.getElementById("result-text").textContent = "No generation has been run";
    currentResultPages = [];
    resultPageIndex = 0;
    resultPageSignature = null;
    renderResultSvg(null);
    renderMolstar(null);
    return;
  }

  const lines = [latestResult.summary || "Generation finished"];
  if (latestResult.records_path) {
    lines.push("");
    lines.push(`Display source: ${latestResult.display_source || "raw"}`);
    if (latestResult.used_fallback) {
      lines.push(`Selection: ${latestResult.selection_reason || "best confidence fallback"}`);
    }
  }
  if (latestResult.records && latestResult.records.length > 0) {
    const top = latestResult.records[0];
    lines.push("");
    lines.push("Top record:");
    lines.push(`PDE: ${top.confidence ?? "n/a"}`);
    lines.push(`likelihood: ${top.normalized_likelihood ?? top.likelihood ?? "n/a"}`);
    lines.push(`smiles: ${top.smiles ?? "n/a"}`);
    lines.push(`gen_seq: ${top.gen_seq ?? "n/a"}`);
    lines.push(`tgt_chains: ${JSON.stringify(top.tgt_chains ?? [])}`);
    lines.push(`lig_chains: ${JSON.stringify(top.lig_chains ?? [])}`);
    const hiddenFilterNames = new Set(["AbnormalConfidenceFilter"]);
    const filterOutputs = (Array.isArray(top.filter_outputs) ? top.filter_outputs : []).filter(
      (output) => !hiddenFilterNames.has(output.name),
    );
    if (filterOutputs.length > 0) {
      lines.push("filter_outputs:");
      for (const output of filterOutputs) {
        lines.push(`- ${output.name}: ${output.status}`);
        const detail = output.detail ?? {};
        const detailText = JSON.stringify(detail);
        if (detailText && detailText !== "{}") {
          lines.push(`  detail: ${detailText}`);
        }
      }
    }
  }
  document.getElementById("result-text").textContent = lines.join("\n");
  const svgPages = latestResult.svg_pages || [];
  const pageSignature = JSON.stringify([latestResult.sdf_path || "", svgPages.length]);
  if (pageSignature !== resultPageSignature) {
    resultPageSignature = pageSignature;
    resultPageIndex = 0;
  }
  currentResultPages = svgPages;
  renderResultSvg(latestResult.svg || null);
  renderMolstar(latestResult.cif_url || null);
}

async function refreshState() {
  const state = await fetchJson("/api/state");
  renderAppState(state);
}

async function executeCode(options = {}) {
  const { statusMessage = "Executing program...", showResultMessage = true } = options;
  const code = document.getElementById("code-editor").value;
  manualBusyMessage = statusMessage;
  setBusyState(true, manualBusyMessage);
  setProgramOutputBase(statusMessage);
  const result = await fetchJson("/api/eval", {
    method: "POST",
    body: JSON.stringify({ code }),
  });
  manualBusyMessage = null;
  if (showResultMessage) {
    setProgramOutputBase(result.output || "OK");
  }
  renderAppState(result.app_state);
  return result;
}

async function runGeneration() {
  const saveDir = document.getElementById("save-dir-input").value;
  const code = document.getElementById("code-editor").value;
  await executeCode({
    statusMessage: "Executing program before generation...",
    showResultMessage: false,
  });
  setProgramOutputBase("Submitting generation...");
  const result = await fetchJson("/api/tasks/submit", {
    method: "POST",
    body: JSON.stringify({ save_dir: saveDir, code }),
  });
  setProgramOutputBase(`Task submitted: ${result.task_id} (${result.status})`);
  renderAppState(result.app_state);
}

async function resetSession() {
  const result = await fetchJson("/api/reset", { method: "POST" });
  setProgramOutputBase(result.output || "Reset");
  renderAppState(result.app_state);
}

window.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("exec-btn").addEventListener("click", async () => {
    try {
      await executeCode();
    } catch (error) {
      manualBusyMessage = null;
      setBusyState(false);
      setProgramOutputBase(`Error: ${error.message}`);
    }
  });

  document.getElementById("run-btn").addEventListener("click", async () => {
    try {
      await runGeneration();
    } catch (error) {
      manualBusyMessage = null;
      setBusyState(false);
      setProgramOutputBase(`Error: ${error.message}`);
    }
  });

  document.getElementById("reset-btn").addEventListener("click", async () => {
    try {
      manualBusyMessage = null;
      await resetSession();
    } catch (error) {
      setProgramOutputBase(`Error: ${error.message}`);
    }
  });

  const prevBtn = document.getElementById("result-prev-btn");
  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      resultPageIndex = Math.max(0, resultPageIndex - 1);
      renderResultSvg(null);
    });
  }

  const nextBtn = document.getElementById("result-next-btn");
  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      resultPageIndex = Math.min(currentResultPages.length - 1, resultPageIndex + 1);
      renderResultSvg(null);
    });
  }

  const cdrPrevBtn = document.getElementById("cdr-prev-btn");
  if (cdrPrevBtn) {
    cdrPrevBtn.addEventListener("click", () => {
      cdrPageIndex = Math.max(0, cdrPageIndex - 1);
      renderCdrControlSvg();
    });
  }

  const cdrNextBtn = document.getElementById("cdr-next-btn");
  if (cdrNextBtn) {
    cdrNextBtn.addEventListener("click", () => {
      cdrPageIndex = Math.min(currentCdrPages.length - 1, cdrPageIndex + 1);
      renderCdrControlSvg();
    });
  }

  const taskSelect = document.getElementById("task-select");
  if (taskSelect) {
    // Freeze polling-driven re-render while the user is opening/interacting with the select.
    taskSelect.addEventListener("mousedown", () => {
      taskSelectFrozen = true;
    });
    taskSelect.addEventListener("focus", () => {
      taskSelectFrozen = true;
    });
    taskSelect.addEventListener("blur", () => {
      taskSelectFrozen = false;
    });
    taskSelect.addEventListener("change", async () => {
      try {
        const taskId = taskSelect.value;
        const result = await fetchJson("/api/tasks/select", {
          method: "POST",
          body: JSON.stringify({ task_id: taskId }),
        });
        // Unfreeze after selection so UI can update immediately.
        taskSelectFrozen = false;
        renderAppState(result.app_state);
      } catch (error) {
        taskSelectFrozen = false;
        setProgramOutputBase(`Error: ${error.message}`);
      }
    });
  }

  try {
    await refreshState();
  } catch (error) {
    setProgramOutputBase(`Error: ${error.message}`);
  }

  // Poll task state.
  setInterval(async () => {
    try {
      await refreshState();
    } catch (error) {
      // Keep silent for intermittent refresh errors.
    }
  }, 2000);
});

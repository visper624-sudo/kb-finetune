/* 知识库微调 Demo 控制台：零依赖前端 */
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const FORMAT_NAMES = {
  policy: "制度解释", product: "产品说明", steps: "操作步骤",
  troubleshoot: "故障处理", integration: "多片段整合",
  refusal: "资料不足拒答", refusal_conflict: "冲突拒答", partial: "部分可答",
};
const DECISION_NAMES = { answer: "作答", refuse: "拒答", conflict: "冲突", partial: "部分可答" };
const CONFIG_NAMES = {
  baseline: "基座模型", dpf: "微调 DPH", ablation_no_refusal: "消融-无拒答",
  ablation_no_citation: "消融-无引用", ablation_no_retrieval: "消融-无检索", lora: "Qwen LoRA",
};

function esc(s) {
  return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* ---------------- 轻量 Markdown：表格 / 编号 / 【标题】 / 引用 ---------------- */
function renderAnswer(text) {
  const lines = (text || "").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*\|/.test(line)) {
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(lines[i]); i++; }
      const cells = rows.map(r => r.trim().replace(/^\||\|$/g, "").split("|").map(x => x.trim()));
      const body = cells.filter(r => !r.every(c => /^-{2,}$/.test(c)));
      if (body.length) {
        out.push("<table><thead><tr>" + body[0].map(c => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>" +
          body.slice(1).map(r => "<tr>" + r.map(c => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") + "</tbody></table>");
      }
      continue;
    }
    if (/^\s*\d+[.、)]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+[.、)]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+[.、)]\s+/, "")); i++;
      }
      out.push("<ol>" + items.map(x => `<li>${inline(x)}</li>`).join("") + "</ol>");
      continue;
    }
    if (/^【[^】]+】/.test(line.trim())) {
      const m = line.trim().match(/^【([^】]+)】(.*)$/);
      out.push(`<h4>${esc(m[1])}</h4>` + (m[2].trim() ? `<p>${inline(m[2].trim())}</p>` : ""));
      i++; continue;
    }
    if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    i++;
  }
  return out.join("");
}

function inline(s) {
  let t = esc(s);
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/\[(\d+)\]/g, '<button class="cite" data-n="$1">[$1]</button>');
  return t;
}

/* ---------------- 问答 ---------------- */
async function ask(q) {
  if (!q || !q.trim()) return;
  $("#askBtn").disabled = true;
  $("#ftAnswer").innerHTML = '<p class="muted">推理中…</p>';
  $("#baselineAnswer").innerHTML = '<p class="muted">生成中…</p>';
  $("#answerGrid").classList.remove("hidden");
  try {
    const r = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "请求失败");
    renderAsk(d);
  } catch (e) {
    $("#ftAnswer").innerHTML = `<p style="color:var(--red)">${esc(e.message)}</p>`;
  } finally {
    $("#askBtn").disabled = false;
  }
}

function renderAsk(d) {
  const ft = d.finetuned, bl = d.baseline;
  $("#baselineAnswer").innerHTML = renderAnswer(bl.answer);
  $("#baselineMeta").innerHTML =
    `<span class="tag gray">无引用</span><span class="tag gray">无格式约束</span><span>${bl.latency_ms} ms</span>`;
  $("#ftAnswer").innerHTML = renderAnswer(ft.answer);
  const citeCount = (ft.citations || []).length;
  const decisionTag = ft.decision === "answer" ? "teal" : (ft.decision === "refuse" ? "amber" : "red");
  $("#ftMeta").innerHTML =
    `<span class="tag ${decisionTag}">${DECISION_NAMES[ft.decision] || ft.decision}</span>` +
    `<span class="tag blue">${FORMAT_NAMES[ft.format] || ft.format}</span>` +
    `<span class="tag teal">引用 ${citeCount}</span>` +
    `<span>gate=${(ft.gate_prob * 100).toFixed(1)}%</span>` +
    `<span>${ft.latency_ms} ms</span>`;
  $("#ftEvidence").textContent = (ft.evidence && ft.evidence.length)
    ? "证据句：" + ft.evidence.slice(0, 3).map(e => e.text.slice(0, 42)).join(" / ") : "";

  const ctxs = d.contexts || [];
  $("#contexts").innerHTML = ctxs.map(c => `
    <div class="ctx" id="ctx-${c.n}">
      <div class="ctx-head"><span class="n">${c.n}</span><span>《${esc(c.title)}》</span>
        <span>第${c.page_no}页</span><span>${esc(c.heading_path || "")}</span></div>
      <div class="text">${esc((c.text || "").replace(/^《[^》]*》\n?/, ""))}</div>
    </div>`).join("");
  $("#contextsWrap").classList.remove("hidden");
  const f = d.features || {};
  $("#features").innerHTML = [
    ["top1 相似度", f.top1_sim], ["最大词覆盖", f.max_cov],
    ["OOV 比例", f.oov_ratio], ["子问题数", f.n_parts], ["冲突标记", f.conflict],
    ["检索耗时", (d.timings || {}).retrieval_ms + " ms"],
  ].map(([k, v]) => `<span>${k}：${v ?? "-"}</span>`).join("");
}

/* ---------------- 数据集 ---------------- */
async function loadSamples(cap = "") {
  $("#sampleList").innerHTML = '<p class="muted">加载中…</p>';
  const r = await fetch("/api/samples?cap=" + encodeURIComponent(cap));
  const d = await r.json();
  $("#sampleList").innerHTML = (d.samples || []).slice(0, 18).map(s => `
    <div class="sample">
      <div class="q">${esc(s.question)}</div>
      <div class="tags">
        <span class="tag blue">${FORMAT_NAMES[s.format] || s.format}</span>
        <span class="tag gray">${s.split}</span>
        <span class="tag ${s.task === "refusal" ? "amber" : "teal"}">${s.task === "refusal" ? "拒答/部分" : "作答"}</span>
      </div>
      <div class="a">${esc(s.answer)}</div>
    </div>`).join("") || '<p class="muted">暂无样本</p>';
}

/* ---------------- 评测 ---------------- */
let STATS = null;
function metricRows(summary) {
  const cols = Object.keys(summary);
  const rows = [
    ["faithfulness", "忠实性", 3], ["hallucination_rate", "幻觉率", 3],
    ["citation_accuracy", "引用准确率", 3], ["citation_coverage", "引用覆盖率", 3],
    ["citation_gold", "引用 gold 命中", 3], ["key_point_coverage", "要点覆盖率", 3],
    ["format_compliance", "格式合规率", 3], ["term_accuracy", "术语准确率", 3],
    ["refusal_accuracy", "拒答准确率", 3], ["false_refusal_rate", "误拒率", 3],
    ["composite", "综合分", 3],
  ];
  let html = "<table class='metrics'><thead><tr><th>指标</th>" +
    cols.map(c => `<th>${CONFIG_NAMES[c] || c}</th>`).join("") + "</tr></thead><tbody>";
  for (const [key, name, digits] of rows) {
    const vals = cols.map(c => summary[c] ? summary[c][key] : null);
    const lowerBetter = key === "hallucination_rate" || key === "false_refusal_rate";
    const nums = vals.filter(v => typeof v === "number");
    const best = nums.length ? (lowerBetter ? Math.min(...nums) : Math.max(...nums)) : null;
    html += `<tr><td>${name}</td>` + vals.map(v => {
      if (typeof v !== "number") return "<td class='num'>—</td>";
      const isBest = v === best && nums.length > 1;
      return `<td class='num ${isBest ? "best" : ""}'>${v.toFixed(digits)}</td>`;
    }).join("") + "</tr>";
  }
  html += `<tr><td>样本量</td>` + cols.map(c => `<td class='num'>${summary[c] ? summary[c].n : "—"}</td>`).join("") + "</tr>";
  return html + "</tbody></table>";
}

function renderStats(d) {
  STATS = d;
  const ds = d.dataset || {}, ev = d.eval || {}, model = d.model || {};
  const dpf = ev.dpf || {};
  $("#datasetStats").innerHTML = [
    ["样本总数", ds.total, "train " + ((ds.by_split || {}).train || 0) + " / valid " + ((ds.by_split || {}).valid || 0) + " / test " + ((ds.by_split || {}).test || 0)],
    ["拒答类占比", ((ds.refusal_ratio || 0) * 100).toFixed(1) + "%", "建议 12%~15%"],
    ["知识文档", (d.corpus || {}).docs || 30, "切片 " + ((d.corpus || {}).chunks || 57)],
    ["主题规格", 140, "多片段组合 26 组"],
  ].map(([k, v, sub], i) => `<div class="stat ${i === 1 ? "amber" : i === 0 ? "teal" : ""}">
      <div class="k">${k}</div><div class="v">${v}<small>${sub || ""}</small></div></div>`).join("");
  $("#metricCards").innerHTML = [
    ["综合分（DPH）", (dpf.composite ?? 0).toFixed(3), "baseline " + ((ev.baseline || {}).composite ?? 0).toFixed(3), "teal"],
    ["拒答准确率", ((dpf.refusal_accuracy ?? 0) * 100).toFixed(1) + "%", "误拒 " + ((dpf.false_refusal_rate ?? 0) * 100).toFixed(1) + "%", "amber"],
    ["引用覆盖率", ((dpf.citation_coverage ?? 0) * 100).toFixed(1) + "%", "准确率 " + ((dpf.citation_accuracy ?? 0) * 100).toFixed(1) + "%", "blue"],
    ["格式合规率", ((dpf.format_compliance ?? 0) * 100).toFixed(1) + "%", "4 类格式", ""],
  ].map(([k, v, sub, cls]) => `<div class="stat ${cls}"><div class="k">${k}</div><div class="v">${v}<small>${sub}</small></div></div>`).join("");
  if (Object.keys(ev).length) {
    $("#metricTable").innerHTML = metricRows(ev);
    const entries = Object.entries(ev).filter(([k]) => k !== "lora");
    const max = Math.max(...entries.map(([, v]) => v.composite || 0), 0.001);
    $("#barChart").innerHTML = entries.map(([k, v]) => `
      <div class="bar-row"><span>${CONFIG_NAMES[k] || k}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${((v.composite || 0) / max * 100).toFixed(1)}%"></div></div>
        <span class="num">${(v.composite || 0).toFixed(3)}</span></div>`).join("");
    const pc = d.per_capability || {};
    let html = "<table class='metrics'><thead><tr><th>能力</th><th>n</th><th>综合分</th><th>拒答</th><th>引用</th><th>格式</th></tr></thead><tbody>";
    for (const [cap, m] of Object.entries(pc)) {
      html += `<tr><td>${FORMAT_NAMES[cap] || cap}</td><td class='num'>${m.n}</td>
        <td class='num'>${(m.composite || 0).toFixed(3)}</td>
        <td class='num'>${m.refusal_accuracy == null ? "—" : (m.refusal_accuracy * 100).toFixed(0) + "%"}</td>
        <td class='num'>${m.citation_coverage == null ? "—" : (m.citation_coverage * 100).toFixed(0) + "%"}</td>
        <td class='num'>${(m.format_compliance * 100).toFixed(0)}%</td></tr>`;
    }
    $("#capTable").innerHTML = html + "</tbody></table>";
  }
  $("#statusText").textContent = "DPH 已加载 · " + (model.version || "dph-1.0");
}

/* ---------------- 交互 ---------------- */
function switchTab(name) {
  $$(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "dataset" && !$("#sampleList").dataset.loaded) {
    loadSamples(); $("#sampleList").dataset.loaded = "1";
  }
}

document.addEventListener("click", (e) => {
  const tab = e.target.closest(".tab");
  if (tab) { switchTab(tab.dataset.tab); history.replaceState(null, "", "#" + tab.dataset.tab); }
  const cite = e.target.closest(".cite");
  if (cite) {
    const n = cite.dataset.n;
    $$(".ctx").forEach(c => c.classList.remove("active"));
    const ctx = document.getElementById("ctx-" + n);
    if (ctx) { ctx.classList.add("active"); ctx.scrollIntoView({ behavior: "smooth", block: "center" }); }
  }
  const chip = e.target.closest(".chip");
  if (chip) { $("#question").value = chip.textContent; ask(chip.textContent); }
  const filter = e.target.closest(".filter");
  if (filter) {
    $$(".filter").forEach(f => f.classList.remove("active"));
    filter.classList.add("active");
    loadSamples(filter.dataset.cap || "");
  }
});

$("#askBtn").addEventListener("click", () => ask($("#question").value));
$("#question").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") ask($("#question").value);
});

(async function init() {
  const hash = (location.hash || "#chat").slice(1);
  switchTab(["chat", "dataset", "metrics", "integration"].includes(hash) ? hash : "chat");
  try {
    const r = await fetch("/api/stats");
    renderStats(await r.json());
  } catch (e) { console.warn(e); }
  const params = new URLSearchParams(location.search);
  const q = params.get("q");
  if (q) { $("#question").value = q; ask(q); }
})();

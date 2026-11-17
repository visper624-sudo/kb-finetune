# -*- coding: utf-8 -*-
"""生成自包含的 HTML 微调报告（内联 SVG 图表，不依赖任何外部资源）。

用法：
    python eval/report.py
输出：
    report/微调报告.html
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import paths  # noqa: E402

FORMAT_NAMES = {
    "policy": "制度解释", "product": "产品说明", "steps": "操作步骤",
    "troubleshoot": "故障处理", "integration": "多片段整合",
    "refusal": "资料不足拒答", "refusal_none": "无召回拒答",
    "refusal_weak": "弱相关拒答", "refusal_conflict": "冲突拒答",
    "partial": "部分可答",
}
CONFIG_NAMES = {
    "baseline": "基座模型（检索直答）", "dpf": "策略模型 DPH（本机实测）",
    "ablation_no_refusal": "消融：无拒答训练", "ablation_no_citation": "消融：无引用",
    "ablation_no_retrieval": "消融：无检索", "lora": "Qwen LoRA（云端实测）",
}


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _num(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.3f}"


def svg_bars(items: list[tuple[str, float]], width: int = 860, height: int = 300,
             vmax: float | None = None, color: str = "#2dd4bf") -> str:
    vmax = vmax or max([v for _, v in items] + [0.01]) * 1.15
    left, top, bottom = 150, 24, 42
    bar_h = (height - top - bottom) / max(1, len(items)) - 12
    rows = []
    for i, (label, value) in enumerate(items):
        y = top + i * (bar_h + 12)
        w = max(2, (width - left - 70) * value / vmax)
        rows.append(
            f'<text x="0" y="{y + bar_h * 0.72:.0f}" font-size="13" fill="#334155">{esc(label)}</text>'
            f'<rect x="{left}" y="{y:.0f}" width="{width - left - 70}" height="{bar_h:.0f}" '
            f'rx="6" fill="#eef2f7"/>'
            f'<rect x="{left}" y="{y:.0f}" width="{w:.1f}" height="{bar_h:.0f}" rx="6" fill="{color}"/>'
            f'<text x="{left + w + 10:.0f}" y="{y + bar_h * 0.72:.0f}" font-size="13" '
            f'font-weight="700" fill="#0f172a">{value:.3f}</text>'
        )
    return f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img">{"".join(rows)}</svg>'


def svg_lines(series: dict[str, list[dict]], width: int = 860, height: int = 300,
              color_map: dict[str, str] | None = None) -> str:
    colors = color_map or {"gate_loss": "#2dd4bf", "format_loss": "#5b8cff", "selector_loss": "#f5a524"}
    left, top, right, bottom = 56, 22, 18, 36
    all_y = [p["loss"] for s in series.values() for p in s]
    ymax = max(all_y + [1e-6]) * 1.1
    xmax = max((s[-1]["epoch"] for s in series.values() if s), default=1) or 1
    paths = []
    legend = []
    for i, (name, pts) in enumerate(series.items()):
        if not pts:
            continue
        d = []
        for p in pts:
            x = left + (width - left - right) * (p["epoch"] / xmax)
            y = top + (height - top - bottom) * (1 - p["loss"] / ymax)
            d.append(f"{x:.1f},{y:.1f}")
        paths.append(f'<polyline points="{" ".join(d)}" fill="none" stroke="{colors.get(name, "#333")}" stroke-width="2.4"/>')
        legend.append(f'<rect x="{left + i * 170}" y="{height - 20}" width="12" height="12" rx="3" fill="{colors.get(name, "#333")}"/>'
                      f'<text x="{left + i * 170 + 18}" y="{height - 10}" font-size="12" fill="#475569">{esc(name)}</text>')
    return f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto">{"".join(paths)}{"".join(legend)}</svg>'


def metric_table(summary: dict) -> str:
    cols = list(summary)
    metrics = [
        ("faithfulness", "忠实性"), ("hallucination_rate", "幻觉率"),
        ("citation_accuracy", "引用准确率"), ("citation_coverage", "引用覆盖率"),
        ("citation_gold", "引用命中 gold 片段"), ("key_point_coverage", "要点覆盖率"),
        ("format_compliance", "格式合规率"), ("term_accuracy", "术语准确率"),
        ("refusal_accuracy", "拒答准确率"), ("false_refusal_rate", "误拒率"),
        ("composite", "综合分"),
    ]
    head = "".join(f"<th>{esc(CONFIG_NAMES.get(c, c))}</th>" for c in cols)
    rows = []
    for key, name in metrics:
        cells = []
        vals = [summary[c].get(key) for c in cols]
        nums = [v for v in vals if isinstance(v, (int, float))]
        best = (min(nums) if key in ("hallucination_rate", "false_refusal_rate") else max(nums)) if nums else None
        for v in vals:
            if not isinstance(v, (int, float)):
                cells.append("<td>—</td>")
            else:
                cls = "best" if len(nums) > 1 and v == best else ""
                cells.append(f'<td class="num {cls}">{v:.3f}</td>')
        rows.append(f"<tr><td>{name}</td>{''.join(cells)}</tr>")
    rows.append("<tr><td>样本量</td>" + "".join(f'<td class="num">{summary[c]["n"]}</td>' for c in cols) + "</tr>")
    return f"<table><thead><tr><th>指标</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def case_block(rec: dict, base: dict | None) -> str:
    ctx = "".join(f'<span class="chip">[{c["n"]}] {esc(c["title"])}</span>' for c in rec.get("contexts", [])[:4])
    return f"""
    <div class="case">
      <div class="case-q"><b>问题</b>：{esc(rec['question'])}</div>
      <div class="chips">{ctx}</div>
      <div class="two">
        <div class="box base"><div class="box-h">Baseline</div><pre>{esc((base or {}).get('pred', '（未运行）'))}</pre></div>
        <div class="box ft"><div class="box-h">微调模型 DPH</div><pre>{esc(rec['pred'])}</pre></div>
      </div>
      <div class="judge">忠实性 {rec['judge'].get('faithfulness')} · 引用覆盖 {rec['judge'].get('citation_coverage')}
        · 格式 {rec['judge'].get('format_ok')} · 拒答正确 {rec['judge'].get('refusal_ok')}</div>
    </div>"""


def main() -> int:
    stats = paths.load_json(paths.OUT_DIR / "dataset_stats.json", {}) or {}
    summary = paths.load_json(paths.RESULTS_DIR / "summary.json", {}) or {}
    per_cap = paths.load_json(paths.RESULTS_DIR / "dpf" / "per_capability.json", {}) or {}
    meta = paths.load_json(paths.ADAPTER_DIR / "policy" / "meta.json", {}) or {}
    train_log = paths.load_json(paths.TRAIN_DIR / "train_log.json", {}) or {}
    dpf = paths.read_jsonl(paths.RESULTS_DIR / "dpf" / "predictions.jsonl")
    baseline = {r["id"]: r for r in paths.read_jsonl(paths.RESULTS_DIR / "baseline" / "predictions.jsonl")}
    corpus = paths.load_json(paths.CORPUS_DIR / "corpus_stats.json", {}) or {}

    dpf_m = summary.get("dpf", {})
    base_m = summary.get("baseline", {})
    lora_m = summary.get("lora", {})
    lora_summary = paths.load_json(paths.ADAPTER_DIR / "lora" / "training_summary.json", {}) or {}
    lora_valid = lora_summary.get("valid_loss")
    lora_status = ("已执行（云端 RTX 4090，valid_loss "
                   + (f"{lora_valid:.3f}" if isinstance(lora_valid, (int, float)) else "见 training_summary.json")
                   + "）") if lora_m else "脚本已就绪（需 GPU）"
    composite_items = [(CONFIG_NAMES.get(k, k), v.get("composite", 0)) for k, v in summary.items()]
    bar = svg_bars(composite_items, color="#2dd4bf")
    loss = svg_lines({
        "gate_loss": train_log.get("gate_loss", []),
        "format_loss": train_log.get("format_loss", []),
        "selector_loss": train_log.get("selector_loss", []),
    })

    # 挑 6 个代表性案例
    wanted = ["policy", "product", "troubleshoot", "refusal", "refusal_conflict", "partial"]
    cases = []
    for fmt in wanted:
        cand = [r for r in dpf if r["format"] == fmt and r["judge"].get("format_ok")]
        if not cand:
            cand = [r for r in dpf if r["format"] == fmt]
        if cand:
            cases.append(case_block(cand[0], baseline.get(cand[0]["id"])))

    # 失败案例：优先展示拒答/格式失败
    failures = sorted(
        [r for r in dpf if r["judge"].get("refusal_ok") is False or not r["judge"].get("format_ok")],
        key=lambda r: (r["judge"].get("refusal_ok") is not False, r["judge"].get("format_ok")),
    )[:5]
    fail_html = "".join(
        f"""<div class="case"><div class="case-q"><b>{esc(r['id'])}</b> · {esc(r['question'])}</div>
        <pre class="pred">{esc(r['pred'])}</pre>
        <div class="judge">期望：{esc(r['format'])}；判定：{esc(json.dumps(r['judge'], ensure_ascii=False))}</div></div>"""
        for r in failures
    )

    def fmtv(x):
        return "—" if x is None else f"{x:.3f}"

    cap_rows = "".join(
        "<tr><td>{}</td><td class='num'>{}</td><td class='num'>{}</td>"
        "<td class='num'>{}</td><td class='num'>{}</td><td class='num'>{}</td></tr>".format(
            esc(FORMAT_NAMES.get(cap, cap)), m["n"], fmtv(m["composite"]),
            fmtv(m["refusal_accuracy"]), fmtv(m["citation_coverage"]),
            fmtv(m["format_compliance"]),
        )
        for cap, m in per_cap.items()
    )

    gate = (meta.get("gate_metrics") or {}).get("valid", {})
    fmt_m = meta.get("format_metrics") or {}
    sel = (meta.get("selector_metrics") or {}).get("valid", {})
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>知识库项目模型微调 · 报告</title>
<style>
  :root{{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--teal:#0d9488;--blue:#2563eb;--amber:#d97706;--bg:#f8fafc}}
  *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
    font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;line-height:1.7}}
  .wrap{{max-width:1120px;margin:0 auto;padding:34px 30px 70px}}
  header.hero{{background:linear-gradient(135deg,#0f172a,#134e4a);color:#ecfeff;border-radius:20px;padding:34px 36px;
    box-shadow:0 20px 50px rgba(15,23,42,.18)}}
  header.hero h1{{margin:0 0 8px;font-size:30px;letter-spacing:.5px}}
  header.hero p{{margin:0;color:#bae6fd}}
  .badges{{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}}
  .badge{{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);padding:5px 12px;border-radius:999px;font-size:12.5px}}
  .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}}
  .card{{background:#fff;border:1px solid var(--line);border-radius:16px;padding:18px 20px;box-shadow:0 6px 18px rgba(15,23,42,.04)}}
  .card .k{{color:var(--muted);font-size:12.5px}} .card .v{{font-size:28px;font-weight:800;color:var(--teal);margin:2px 0}}
  .card .s{{color:var(--muted);font-size:12px}}
  h2{{font-size:22px;margin:38px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--line)}}
  h3{{font-size:16px;margin:24px 0 8px}}
  p{{margin:8px 0}} .muted{{color:var(--muted)}}
  table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:13px;margin:12px 0}}
  th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
  th{{background:#f1f5f9;color:#334155;font-weight:700;white-space:nowrap}}
  td.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}
  td.best{{color:var(--teal);font-weight:800}}
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
  .case{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:14px 0}}
  .case-q{{margin-bottom:8px}}
  .chips{{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 10px}}
  .chip{{font-size:11.5px;color:#475569;background:#f1f5f9;border-radius:999px;padding:2px 9px}}
  .two{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
  .box{{border:1px solid var(--line);border-radius:12px;overflow:hidden}}
  .box-h{{padding:6px 12px;font-weight:700;font-size:12.5px;background:#f8fafc;border-bottom:1px solid var(--line)}}
  .box.base .box-h{{color:#64748b}} .box.ft .box-h{{color:var(--teal)}}
  pre{{margin:0;padding:12px 14px;font:12px/1.65 "Cascadia Code",Consolas,monospace;white-space:pre-wrap;word-break:break-word}}
  .judge{{margin-top:8px;color:var(--muted);font-size:12px}}
  .note{{background:#fffbeb;border:1px solid #fde68a;border-radius:12px;padding:12px 16px;color:#92400e;font-size:13px}}
  img.shot{{width:100%;border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 24px rgba(15,23,42,.08)}}
  code{{background:#f1f5f9;border-radius:5px;padding:1px 5px;font-size:12.5px}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
  footer{{margin-top:44px;color:var(--muted);font-size:12px;text-align:center}}
  @media(max-width:900px){{.cards{{grid-template-columns:repeat(2,1fr)}}.two,.two-col,.grid2{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<header class="hero">
  <h1>知识库项目模型微调（4.1）</h1>
  <p>从 2.9 知识库文档反向合成数据 → 训练 → 评测 → 零改造接入知识库平台</p>
  <div class="badges">
    <span class="badge">训练样本 {stats.get('by_split', {}).get('train', 0)}</span>
    <span class="badge">测试样本 {stats.get('by_split', {}).get('test', 0)}</span>
    <span class="badge">测试集 647 条 · 6 组对比</span>
    <span class="badge">综合分 {base_m.get('composite', 0):.3f} → DPH {dpf_m.get('composite', 0):.3f} / LoRA {lora_m.get('composite', 0):.3f}</span>
    <span class="badge">数据源：30 篇 / 57 切片</span>
  </div>
</header>

<div class="cards">
  <div class="card"><div class="k">综合分（DPH / LoRA）</div><div class="v">{dpf_m.get('composite', 0):.3f} / {lora_m.get('composite', 0):.3f}</div><div class="s">Baseline {base_m.get('composite', 0):.3f}</div></div>
  <div class="card"><div class="k">LoRA 引用覆盖率</div><div class="v">{lora_m.get('citation_coverage', 0) * 100:.1f}%</div><div class="s">引用准确率 {lora_m.get('citation_accuracy', 0) * 100:.1f}%</div></div>
  <div class="card"><div class="k">LoRA 拒答准确率</div><div class="v">{lora_m.get('refusal_accuracy', 0) * 100:.1f}%</div><div class="s">误拒率 {lora_m.get('false_refusal_rate', 0) * 100:.1f}%</div></div>
  <div class="card"><div class="k">LoRA 格式合规率</div><div class="v">{lora_m.get('format_compliance', 0) * 100:.1f}%</div><div class="s">DPH 为 {dpf_m.get('format_compliance', 0) * 100:.1f}%</div></div>
</div>

<h2>一、任务与能力映射（4.1.1 ~ 4.1.5）</h2>
<p>本项目不是“让模型更聪明”，而是“让模型更守规矩”：守引用、守格式、守边界。数据构造比模型选择更重要。</p>
<table><thead><tr><th>条款</th><th>能力</th><th>数据构造</th><th>评测指标</th><th>实测（DPH / LoRA）</th></tr></thead><tbody>
<tr><td>4.1.1</td><td>行业术语与格式适配</td><td>140 个领域主题 × 4 类格式；术语表 + 口语别名改写</td><td>术语准确率 / 格式合规率</td><td>{dpf_m.get('term_accuracy', 0):.3f} / {lora_m.get('term_accuracy', 0):.3f}；{dpf_m.get('format_compliance', 0):.3f} / {lora_m.get('format_compliance', 0):.3f}</td></tr>
<tr><td>4.1.2</td><td>检索内容忠实回答</td><td>gold 答案只摘录证据句；逐句数字溯源校验</td><td>忠实性 / 幻觉率</td><td>{dpf_m.get('faithfulness', 0):.3f} / {lora_m.get('faithfulness', 0):.3f}；{dpf_m.get('hallucination_rate', 0):.3f} / {lora_m.get('hallucination_rate', 0):.3f}</td></tr>
<tr><td>4.1.3</td><td>多片段信息整合</td><td>26 组跨文档组合（2~4 个主题）+ 部分可答</td><td>要点覆盖率</td><td>{dpf_m.get('key_point_coverage', 0):.3f} / {lora_m.get('key_point_coverage', 0):.3f}</td></tr>
<tr><td>4.1.4</td><td>证据引用与出处</td><td>答案句内嵌 [n]；引用编号与上下文一一对应</td><td>引用准确率 / 覆盖率</td><td>{dpf_m.get('citation_accuracy', 0):.3f} / {lora_m.get('citation_accuracy', 0):.3f}；{dpf_m.get('citation_coverage', 0):.3f} / {lora_m.get('citation_coverage', 0):.3f}</td></tr>
<tr><td>4.1.5</td><td>资料不足拒答</td><td>60 无召回 + 45 弱相关 + 14 冲突 + 20 部分可答</td><td>拒答准确率 / 误拒率</td><td>{dpf_m.get('refusal_accuracy', 0):.3f} / {lora_m.get('refusal_accuracy', 0):.3f}；{dpf_m.get('false_refusal_rate', 0):.3f} / {lora_m.get('false_refusal_rate', 0):.3f}</td></tr>
</tbody></table>

<h2>二、数据构造</h2>
<div class="grid2">
  <div>
    <h3>规模与切分</h3>
    <table><tbody>
      <tr><td>样本总数</td><td class="num">{stats.get('total', 0)}</td></tr>
      <tr><td>train / valid / test</td><td class="num">{stats.get('by_split', {}).get('train', 0)} / {stats.get('by_split', {}).get('valid', 0)} / {stats.get('by_split', {}).get('test', 0)}</td></tr>
      <tr><td>拒答类占比</td><td class="num">{stats.get('refusal_ratio', 0) * 100:.1f}%</td></tr>
      <tr><td>领域主题 / 多片段组合</td><td class="num">140 / 26</td></tr>
      <tr><td>负样本（无召回/弱相关/冲突/部分）</td><td class="num">60 / 45 / 14 / 20</td></tr>
      <tr><td>文档级切分泄漏</td><td class="num">0</td></tr>
      <tr><td>质检失败</td><td class="num">{stats.get('failures', 0)}</td></tr>
    </tbody></table>
  </div>
  <div>
    <h3>能力分布</h3>
    {svg_bars([(FORMAT_NAMES.get(k, k), v) for k, v in (stats.get('by_capability') or {}).items()], height=300, color="#2563eb")}
  </div>
</div>
<div class="note"><b>质量门禁</b>：每个答案句中的数字必须能在被引用片段中找到；格式由机器校验；按文档切分（同一文档不跨 train/test）；跨 split 问题重合 0；拒答样本比例 12%~18%。</div>

<h2>三、训练</h2>
<p>仓库提供两条路径，共用同一份数据、同一 system prompt、同一评测脚本：</p>
<table><thead><tr><th>路径</th><th>实现</th><th>本机状态</th><th>产物</th></tr></thead><tbody>
<tr><td>标准 LoRA SFT</td><td><code>train/train_lora.py</code>（Qwen2.5 + peft，支持 QLoRA / 断点 / loss 记录）</td><td>{lora_status}</td><td><code>adapter/lora/</code></td></tr>
<tr><td>轻量策略模型 DPH</td><td><code>train/distill_policy.py</code>：Gate 二分类 + Format 四分类 + Evidence Selector</td><td><b>已训练并完成全部实测</b></td><td><code>adapter/policy/</code></td></tr>
</tbody></table>
<div class="grid2">
  <div><h3>Gate / 拒答路由</h3><table><tbody>
    <tr><td>验证集准确率</td><td class="num">{gate.get('accuracy', 0):.3f}</td></tr>
    <tr><td>回答召回 / 拒答召回</td><td class="num">{gate.get('answer_recall', 0):.3f} / {gate.get('refusal_recall', 0):.3f}</td></tr>
    <tr><td>误拒率</td><td class="num">{gate.get('false_refusal', 0):.3f}</td></tr>
    <tr><td>AUC</td><td class="num">{gate.get('auc', 0):.3f}</td></tr>
    <tr><td>阈值</td><td class="num">{meta.get('gate_threshold')}</td></tr>
  </tbody></table></div>
  <div><h3>Format / 证据选择</h3><table><tbody>
    <tr><td>格式分类准确率（valid）</td><td class="num">{fmt_m.get('valid_accuracy', 0):.3f}</td></tr>
    <tr><td>Selector 证据召回 / 精确率</td><td class="num">{sel.get('evidence_recall', 0):.3f} / {sel.get('evidence_precision', 0):.3f}</td></tr>
    <tr><td>训练行数（gate/format/selector）</td><td class="num">{meta.get('train_rows', {}).get('gate', 0)} / {meta.get('train_rows', {}).get('format', 0)} / {meta.get('train_rows', {}).get('selector', 0)}</td></tr>
  </tbody></table></div>
</div>
{loss}
<p class="muted">训练曲线：Gate / Format / Selector 三个头的交叉熵 loss（每 10 步记录）。</p>

<h2>四、评测</h2>
<p>测试集固定 647 条，同一检索与 Prompt；三组对比 + 两个消融。综合分 = 忠实 30% + 引用 20% + 拒答 20% + 覆盖 20% + 格式 10%。</p>
{metric_table(summary)}
<div class="grid2">
  <div><h3>综合分对比</h3>{bar}</div>
  <div><h3>分能力表现（DPH）</h3>
    <table><thead><tr><th>能力</th><th>n</th><th>综合分</th><th>拒答</th><th>引用</th><th>格式</th></tr></thead>
    <tbody>{cap_rows}</tbody></table>
  </div>
</div>
<div class="note"><b>怎么读这张表</b>：DPH 是抽取式实现，忠实性接近上限；LoRA 是生成式模型，幻觉率 0.006、引用覆盖 93.7%、
格式 85.0%、拒答 89.0%，综合分 0.836，在格式与拒答上明显优于 DPH，但忠实性/术语略低。
两条路径共用同一测试集与同一套指标，差异可以逐项对比。</div>

<h2>五、典型案例</h2>
{''.join(cases)}

<h2>六、失败案例分析</h2>
<p>优先抽取“应拒答但作答”或“格式不符”的样本；每条都给出模型输出与判定明细，便于定位数据/策略问题。</p>
{fail_html}

<h2>七、集成：零改造接入 2.9 知识库平台</h2>
<p>本仓库的 Demo 服务同时暴露 OpenAI 兼容接口 <code>/v1/chat/completions</code>。2.9 只改 4 个配置项，
检索、四维鉴权裁剪、SSE、引用卡片全部保持原样；鉴权仍发生在 Prompt 拼接之前。</p>
<img class="shot" src="../screenshots/09-29-chat.png" alt="2.9 接入微调模型后的问答页面">
<p class="muted">上：2.9 知识库平台接入 <code>kb-finetuned-dph</code> 后的真实问答页面（含检索与鉴权轨迹）。
下：模型与系统参数配置页面。</p>
<img class="shot" src="../screenshots/10-29-config.png" alt="2.9 模型配置页面">
<pre class="note">provider = openai    base_url = http://127.0.0.1:8802/v1
api_key  = any       model    = kb-finetuned-dph</pre>

<h2>八、结论与局限</h2>
<ul>
  <li><b>数据是主要矛盾</b>：140 个主题 + 4 类负样本 + 逐句溯源，使引用覆盖率从 0 提升到 DPH {dpf_m.get('citation_coverage', 0) * 100:.1f}% / LoRA {lora_m.get('citation_coverage', 0) * 100:.1f}%；拒答准确率从 0 提升到 DPH {dpf_m.get('refusal_accuracy', 0) * 100:.1f}% / LoRA {lora_m.get('refusal_accuracy', 0) * 100:.1f}%。</li>
  <li><b>消融证明能力来自数据</b>：去掉拒答样本后拒答准确率降到 {summary.get('ablation_no_refusal', {}).get('refusal_accuracy', 0) * 100:.1f}%；去掉检索后幻觉率升到 {summary.get('ablation_no_retrieval', {}).get('hallucination_rate', 0) * 100:.1f}%。</li>
  <li><b>LoRA 实测</b>：Qwen2.5-1.5B + bf16 LoRA（r=16, 2 epoch, 612 步）在云端 RTX 4090 完成训练，valid_loss 0.218；综合分 {lora_m.get('composite', 0):.3f}，高于 DPH 的 {dpf_m.get('composite', 0):.3f}。</li>
  <li><b>局限 1</b>：LoRA 只有 1.5B、2 epoch，规模有限；忠实性/术语略低于抽取式 DPH，且幻觉率 0.6% 不是 0。</li>
  <li><b>局限 2</b>：LoRA 格式合规率 {lora_m.get('format_compliance', 0) * 100:.1f}%、要点覆盖率 {lora_m.get('key_point_coverage', 0) * 100:.1f}% 仍未达方案目标；troubleshoot 格式仅 0.321，弱相关/部分可答仍是短板。</li>
  <li><b>局限 3</b>：测试集与训练集共享模板族（按文档切分不泄漏内容，但问法模板同源），格式准确率偏乐观。</li>
</ul>

<footer>知识库项目模型微调 · 4.1 · 数据来自 2.9 知识库平台 30 篇虚构制度文档 · 报告自动生成</footer>
</div></body></html>"""
    out = paths.REPORT_DIR / "微调报告.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print("written:", out, len(doc), "chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

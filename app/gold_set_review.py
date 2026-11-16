"""Build a reusable local HTML reviewer for image-rule gold sets."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from app.gold_set import parse_bool
from app.image_rules import hamming_distance


REVIEW_CSV_FIELDS = (
    "rule_id",
    "episode_id",
    "frame_index",
    "timestamp_s",
    "evidence_path",
    "auto_hit",
    "human_label",
    "is_true_positive",
    "note",
)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def _metric_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("episode_id")), int(row.get("frame_index"))


def read_metric_rows(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def _review_id(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("rule_id", "")),
        str(row.get("episode_id", "")),
        str(_as_int(row.get("frame_index"))),
        str(bool(parse_bool(row.get("auto_hit")))),
    ]
    return "|".join(parts)


def _stable_key(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(REVIEW_CSV_FIELDS).encode("utf-8"))
    for row in rows:
        digest.update(_review_id(row).encode("utf-8"))
        digest.update(b"\0")
    return f"gold-set-review:{digest.hexdigest()}"


def _relative_or_file_uri(path: str | Path, base_dir: Path) -> str:
    path = Path(path).resolve()
    base_dir = base_dir.resolve()
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_uri()


def _copy_evidence(
    source_path: str | Path,
    output_dir: Path,
    episode_id: str,
    frame_index: int,
    role: str,
) -> str:
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"evidence frame does not exist: {source}")
    safe_episode_id = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        episode_id,
    ).strip("._-") or "episode"
    target_dir = output_dir / "evidence" / safe_episode_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{role}_{frame_index:08d}{source.suffix.lower()}"
    if not target.exists():
        shutil.copy2(source, target)
    return (Path("evidence") / safe_episode_id / target.name).as_posix()


def build_review_items(
    gold_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    hit_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    copy_evidence: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach current/previous sampled frames and rule evidence to each row."""
    output_dir = output_dir.resolve()
    metrics_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        metrics_by_episode.setdefault(str(row.get("episode_id")), []).append(
            deepcopy(row)
        )
    for rows in metrics_by_episode.values():
        rows.sort(key=_metric_key)

    metric_lookup = {
        _metric_key(row): row
        for rows in metrics_by_episode.values()
        for row in rows
    }
    hit_lookup = {
        (
            str(row.get("rule_id")),
            str(row.get("episode_id")),
            _as_int(row.get("frame_index")),
        ): row
        for row in hit_rows
    }

    review_items: list[dict[str, Any]] = []
    missing_current: list[str] = []
    missing_previous: list[str] = []
    evidence_references = 0
    evidence_sources: set[str] = set()

    for gold_row in gold_rows:
        rule_id = str(gold_row.get("rule_id", ""))
        episode_id = str(gold_row.get("episode_id", ""))
        frame_index = _as_int(gold_row.get("frame_index"))
        if frame_index is None:
            raise ValueError(f"gold-set row has invalid frame_index: {gold_row}")
        current = metric_lookup.get((episode_id, frame_index))
        if current is None:
            missing_current.append(_review_id(gold_row))
            continue

        episode_metrics = metrics_by_episode[episode_id]
        position = next(
            (
                index
                for index, row in enumerate(episode_metrics)
                if _as_int(row.get("frame_index")) == frame_index
            ),
            None,
        )
        if position is None:
            raise ValueError(f"metric position not found for {_review_id(gold_row)}")
        previous = episode_metrics[position - 1] if position > 0 else None
        if previous is None:
            missing_previous.append(_review_id(gold_row))

        current_source = str(current.get("evidence_path", ""))
        if copy_evidence:
            current_uri = str(_copy_evidence(
                current_source,
                output_dir,
                episode_id,
                frame_index,
                "current",
            ))
            evidence_references += 1
            evidence_sources.add(str(Path(current_source).resolve()))
            previous_uri = None
            if previous is not None:
                previous_frame = _as_int(previous.get("frame_index"))
                if previous_frame is not None:
                    previous_uri = str(_copy_evidence(
                        str(previous.get("evidence_path", "")),
                        output_dir,
                        episode_id,
                        previous_frame,
                        "previous",
                    ))
                    evidence_references += 1
                    evidence_sources.add(
                        str(Path(previous.get("evidence_path", "")).resolve())
                    )
        else:
            current_uri = _relative_or_file_uri(current_source, output_dir)
            evidence_references += 1
            evidence_sources.add(str(Path(current_source).resolve()))
            previous_uri = (
                _relative_or_file_uri(previous.get("evidence_path", ""), output_dir)
                if previous is not None
                else None
            )
            if previous is not None:
                evidence_references += 1
                evidence_sources.add(
                    str(Path(previous.get("evidence_path", "")).resolve())
                )

        hit = hit_lookup.get((rule_id, episode_id, frame_index), {})
        current_phash = str(current.get("phash", ""))
        previous_phash = str(previous.get("phash", "")) if previous else ""
        item = deepcopy(gold_row)
        item["review_id"] = _review_id(gold_row)
        item["auto_hit_bool"] = bool(parse_bool(gold_row.get("auto_hit")))
        item["current_uri"] = current_uri
        item["previous_uri"] = previous_uri
        item["previous_frame_index"] = (
            _as_int(previous.get("frame_index")) if previous else None
        )
        item["previous_timestamp_s"] = (
            _as_float(previous.get("timestamp_s")) if previous else None
        )
        item["current_brightness_mean"] = _as_float(current.get("brightness_mean"))
        item["current_laplacian_variance"] = _as_float(
            current.get("laplacian_variance")
        )
        item["sampled_hamming_distance"] = (
            hamming_distance(previous_phash, current_phash)
            if previous_phash and current_phash
            else None
        )
        item["auto_rule_value"] = _as_float(hit.get("value"))
        item["auto_rule_threshold"] = _as_float(hit.get("threshold"))
        item["auto_rule_message"] = hit.get("message", "")
        item["review_guidance"] = _guidance(rule_id)
        review_items.append(item)

    summary = {
        "input_rows": len(gold_rows),
        "review_items": len(review_items),
        "missing_current_frames": missing_current,
        "rows_without_previous_frame": missing_previous,
        "evidence_references": evidence_references,
        "unique_evidence_files": len(evidence_sources),
        "copy_evidence": copy_evidence,
    }
    if missing_current:
        raise FileNotFoundError(
            "gold-set rows missing current image metrics: "
            + ", ".join(missing_current)
        )
    return review_items, summary


def _guidance(rule_id: str) -> dict[str, str]:
    defaults = {
        "title": "判断自动规则描述的条件是否成立",
        "true": "规则成立",
        "false": "规则不成立",
        "detail": "根据当前证据判断自动规则的语义是否真实成立。",
    }
    if rule_id == "image_blur":
        return {
            "title": "Blur：看当前帧",
            "true": "当前帧确实明显模糊",
            "false": "当前帧足够清晰",
            "detail": "低纹理背景或空桌面不等于模糊；重点看任务关键区域是否不可辨认。",
        }
    if rule_id == "image_static_candidate":
        return {
            "title": "Static candidate：对比前一帧和当前帧",
            "true": "两帧确实视觉上几乎不变",
            "false": "两帧存在明显视觉变化",
            "detail": "该规则只判断视觉静止，不代表数据一定损坏；正常等待或准备动作也可能成立。",
        }
    if rule_id == "image_freeze":
        return {
            "title": "Freeze：对比前一帧和当前帧",
            "true": "图像几乎不变，且怀疑相机卡帧/时间错位",
            "false": "图像有变化，或 state-motion 证据不足以支持冻结",
            "detail": "重点看前后帧是否几乎一致，同时参考规则给出的 state-motion 阈值。",
        }
    return defaults


def _json_script_payload(items: list[dict[str, Any]], storage_key: str) -> str:
    payload = {
        "storageKey": storage_key,
        "fields": list(REVIEW_CSV_FIELDS),
        "items": items,
    }
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def write_review_html(
    items: list[dict[str, Any]],
    *,
    output_path: Path,
    summary: dict[str, Any],
) -> None:
    """Write a dependency-free HTML page that works from the local filesystem."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_script_payload(items, _stable_key(items))
    escaped_summary = html.escape(json.dumps(summary, ensure_ascii=False))
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Rule Gold Set Review</title>
<style>
:root {{ color-scheme: light; --line:#d9e2ec; --ink:#172033; --muted:#607083; --bg:#f5f7fa; --card:#fff; --accent:#175cd3; --ok:#087443; --bad:#b42318; --warn:#b54708; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Microsoft YaHei",Arial,sans-serif; }}
header {{ position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid var(--line); padding:14px 20px; }}
.header-row {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap; }}
h1 {{ font-size:18px; margin:0 0 4px; }}
.subtitle {{ color:var(--muted); font-size:12px; }}
.controls {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:10px; }}
select,input,button {{ font:inherit; }}
select,input {{ border:1px solid var(--line); background:#fff; border-radius:5px; padding:7px 8px; }}
button {{ border:1px solid var(--line); background:#fff; border-radius:5px; padding:7px 11px; cursor:pointer; }}
button.primary {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
button.good {{ border-color:#a6f4c5; background:#ecfdf3; color:var(--ok); }}
button.bad {{ border-color:#fecdca; background:#fef3f2; color:var(--bad); }}
button.warn {{ border-color:#fec84b; background:#fffaeb; color:var(--warn); }}
button[aria-pressed="true"] {{ outline:3px solid #b2ddff; }}
.progress {{ min-width:210px; height:8px; background:#e4e9f0; border-radius:99px; overflow:hidden; }}
.progress span {{ display:block; height:100%; background:var(--accent); width:0; }}
.progress-label {{ font-variant-numeric:tabular-nums; font-size:12px; color:var(--muted); }}
main {{ max-width:1500px; margin:18px auto 60px; padding:0 18px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px; margin-bottom:16px; }}
.card-top {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap; }}
.ids {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
.rule {{ font-weight:700; }}
.badge {{ display:inline-block; padding:3px 7px; border-radius:999px; background:#eef4ff; color:#1849a9; font-size:12px; }}
.badge.hit {{ background:#fff6ed; color:#b54708; }}
.badge.control {{ background:#eef4ff; color:#175cd3; }}
.badge.done {{ background:#ecfdf3; color:#087443; }}
.meta {{ color:var(--muted); font-size:12px; margin-top:5px; }}
.guidance {{ margin:10px 0; padding:10px; background:#f8fafc; border-left:3px solid var(--accent); color:#344054; font-size:13px; }}
.images {{ display:grid; grid-template-columns:repeat(2,minmax(220px,1fr)); gap:10px; }}
.frame {{ border:1px solid var(--line); border-radius:6px; overflow:hidden; background:#e9eef4; }}
.frame-header {{ padding:7px 8px; font-size:12px; color:#344054; background:#f8fafc; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:8px; }}
.frame img {{ display:block; width:100%; aspect-ratio:4/3; object-fit:contain; background:#111; }}
.empty {{ aspect-ratio:4/3; display:grid; place-items:center; color:#8595aa; font-size:13px; background:#f2f5f8; }}
.actions {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }}
.note {{ width:100%; margin-top:8px; border:1px solid var(--line); border-radius:5px; min-height:42px; padding:8px; font:inherit; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:7px; font-size:12px; }}
.metric {{ padding:7px; background:#f8fafc; border:1px solid #e5eaf1; border-radius:5px; }}
.metric strong {{ display:block; color:#344054; }}
.metric span {{ color:#475467; font-variant-numeric:tabular-nums; }}
@media (max-width:750px) {{ .images {{ grid-template-columns:1fr; }} header {{ padding:10px; }} }}
</style>
</head>
<body>
<header>
  <div class="header-row">
    <div>
      <h1>Image Rule Gold Set Review</h1>
      <div class="subtitle">Blur 看当前帧；Static / Freeze 必须对比前一帧和当前帧。</div>
    </div>
    <div class="controls">
      <label>规则 <select id="ruleFilter"><option value="">All</option></select></label>
      <label>状态 <select id="statusFilter">
        <option value="all">All</option>
        <option value="unreviewed">Unreviewed</option>
        <option value="reviewed">Reviewed</option>
        <option value="hit">Auto hit</option>
        <option value="control">Control</option>
      </select></label>
      <input id="search" type="search" placeholder="Search episode / frame / note">
      <button id="prevUnreviewed">上一未复核</button>
      <button id="nextUnreviewed">下一未复核</button>
      <button id="exportCsv" class="primary">导出 CSV</button>
      <label class="primary"><input type="file" id="importCsv" accept=".csv" style="display:none">导入已复核 CSV</label>
    </div>
  </div>
  <div class="controls">
    <div class="progress" aria-hidden="true"><span id="progressBar"></span></div>
    <span id="progressText" class="progress-label">0 / 0 reviewed</span>
    <span id="decisionText" class="progress-label"></span>
  </div>
</header>
<main><div id="cards"></div><div id="empty" hidden>没有匹配的复核项。</div></main>
<script type="application/json" id="review-data">{payload}</script>
<script>
const payload = JSON.parse(document.getElementById('review-data').textContent);
const fields = payload.fields;
let rows = payload.items.map(x => ({{...x, human_label:x.human_label||'', is_true_positive:x.is_true_positive||'', note:x.note||''}}));
let filtered = [...rows];
const cards = document.getElementById('cards');
const ruleFilter = document.getElementById('ruleFilter');
const statusFilter = document.getElementById('statusFilter');
const search = document.getElementById('search');
function value(row, key) {{ return row[key] ?? ''; }}
function reviewed(row) {{ return row.is_true_positive === '1' || row.is_true_positive === '0' || String(row.human_label).toLowerCase() === 'true' || String(row.human_label).toLowerCase() === 'false'; }}
function saved() {{ try {{ return JSON.parse(localStorage.getItem(payload.storageKey) || '{{}}'); }} catch {{ return {{}}; }} }}
function save() {{
  const state = {{}};
  rows.forEach(r => state[r.review_id] = {{human_label:r.human_label, is_true_positive:r.is_true_positive, note:r.note}});
  try {{ localStorage.setItem(payload.storageKey, JSON.stringify(state)); }} catch {{}}
}}
function restore() {{
  const state = saved();
  rows.forEach(r => {{ const s=state[r.review_id]; if(s) {{ r.human_label=s.human_label||''; r.is_true_positive=s.is_true_positive||''; r.note=s.note||''; }} }});
}}
function setDecision(id, decision) {{
  const row=rows.find(r=>r.review_id===id); if(!row) return;
  row.is_true_positive=decision===undefined?'':String(decision);
  row.human_label=decision===undefined?'':(decision?'true':'false');
  save(); render(); requestAnimationFrame(()=>{{ const card=document.querySelector(`[data-id="${{CSS.escape(id)}}"] .actions`); if(card) card.focus(); }});
}}
function currentVisibleId() {{
  const cards=[...document.querySelectorAll('.card')]; if(!cards.length) return undefined;
  const center=window.innerHeight/2; let best=cards[0], distance=Infinity;
  cards.forEach(card=>{{const box=card.getBoundingClientRect(); const current=Math.abs((box.top+box.bottom)/2-center); if(current<distance){{distance=current;best=card;}}}});
  return best.dataset.id;
}}
function moveUnreviewed(direction) {{
  const ids=filtered.map(r=>r.review_id); if(!ids.length) return;
  let index=ids.indexOf(currentVisibleId()); if(index<0) index=0;
  for(let step=0;step<ids.length;step++) {{
    index=(index+direction+ids.length)%ids.length; const row=rows.find(r=>r.review_id===ids[index]);
    if(row && !reviewed(row)) {{ location.hash=''; document.querySelector(`[data-id="${{CSS.escape(row.review_id)}}"]`).scrollIntoView({{behavior:'smooth',block:'center'}}); return; }}
  }}
}}
function csvEscape(v) {{ const s=String(v ?? ''); return /[",\\r\\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s; }}
function exportCsv() {{
  const lines=[fields.join(',')];
  rows.map(r=>fields.map(f=>csvEscape(value(r,f))).join(',')).forEach(x=>lines.push(x));
  const blob=new Blob(['\\ufeff'+lines.join('\\r\\n')],{{type:'text/csv;charset=utf-8'}});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='gold_set_reviewed.csv'; a.click(); URL.revokeObjectURL(a.href);
}}
function parseCsv(text) {{
  const rows=[]; let row=[], cell='',quoted=false;
  for(let i=0;i<text.length;i++) {{
    const c=text[i];
    if(quoted) {{ if(c==='\"') {{ if(text[i+1]==='\"'){{cell+='\"';i++;}} else quoted=false; }} else cell+=c; }}
    else if(c==='\"') quoted=true;
    else if(c===',') {{ row.push(cell); cell=''; }}
    else if(c==='\\n') {{ row.push(cell.replace(/\\r$/,'')); rows.push(row); row=[];cell=''; }}
    else cell+=c;
  }}
  if(cell||row.length){{row.push(cell.replace(/\\r$/,''));rows.push(row);}}
  return rows.filter(r=>r.length&&r.some(x=>x!==''));
}}
function importCsv(file) {{
  file.text().then(text=>{{
    const parsed=parseCsv(text); const head=parsed.shift(); const incoming=parsed.map(a=>Object.fromEntries(head.map((h,i)=>[h,a[i]??''])));
    const lookup=Object.fromEntries(incoming.map(r=>[`${{r.rule_id}}|${{r.episode_id}}|${{r.frame_index}}|${{String(r.auto_hit).toLowerCase()==='true'}}`,r]));
    rows.forEach(r=>{{ const inc=lookup[r.review_id]; if(inc){{r.human_label=inc.human_label||'';r.is_true_positive=inc.is_true_positive||'';r.note=inc.note||'';}} }});
    save();render();
  }});
}}
function frameHtml(row, uri, label, frame, ts) {{
  const body=uri?`<img src="${{uri}}" loading="lazy" alt="${{label}} frame">`:`<div class="empty">无前一采样帧</div>`;
  return `<section class="frame"><div class="frame-header"><strong>${{label}}</strong><span>frame ${{frame??'-'}} / ${{ts??'-'}}s</span></div>${{body}}</section>`;
}}
function metric(label,value) {{ return `<div class="metric"><strong>${{label}}</strong><span>${{value??'-'}}</span></div>`; }}
function render() {{
  const previousRule=ruleFilter.value;
  const rules=[...new Set(rows.map(r=>r.rule_id))].sort(); ruleFilter.innerHTML='<option value="">All</option>'+rules.map(r=>`<option>${{r}}</option>`).join('');
  ruleFilter.value=previousRule;
  const selectedRule=ruleFilter.value, status=statusFilter.value, q=search.value.trim().toLowerCase();
  filtered=rows.filter(r=>{{ if(selectedRule&&r.rule_id!==selectedRule)return false; if(status==='unreviewed'&&reviewed(r))return false; if(status==='reviewed'&&!reviewed(r))return false; if(status==='hit'&&!r.auto_hit_bool)return false; if(status==='control'&&r.auto_hit_bool)return false; if(q&&!`${{r.rule_id}} ${{r.episode_id}} ${{r.frame_index}} ${{r.note}}`.toLowerCase().includes(q))return false; return true; }});
  cards.innerHTML=filtered.map(r=>{{
    const done=reviewed(r), decision=r.is_true_positive==='1'||String(r.human_label).toLowerCase()==='true';
    const decisionText=done?(decision?'规则成立':'规则不成立'):'未复核';
    return `<article class="card" data-id="${{r.review_id}}" tabindex="-1">
      <div class="card-top"><div><div class="ids"><span class="rule">${{r.rule_id}}</span><span class="badge ${{r.auto_hit_bool?'hit':'control'}}">${{r.auto_hit_bool?'auto hit':'control'}}</span><span class="badge ${{done?'done':''}}">${{decisionText}}</span></div><div class="meta">${{r.episode_id}} · frame ${{r.frame_index}} · ${{r.timestamp_s}}s</div></div><div class="meta">review_id: ${{r.review_id}}</div></div>
      <div class="guidance"><strong>${{r.review_guidance.title}}</strong><br>${{r.review_guidance.detail}}<br><span>成立：${{r.review_guidance.true}}；不成立：${{r.review_guidance.false}}</span></div>
      <div class="images">${{frameHtml(r,r.previous_uri,'前一采样帧',r.previous_frame_index,r.previous_timestamp_s)}}${{frameHtml(r,r.current_uri,'当前帧',r.frame_index,r.timestamp_s)}}</div>
      <div class="metrics">${{metric('Sampled pHash Hamming',r.sampled_hamming_distance)}}${{metric('Auto rule value',r.auto_rule_value)}}${{metric('Auto rule threshold',r.auto_rule_threshold)}}${{metric('Laplacian variance',r.current_laplacian_variance)}}${{metric('Brightness mean',r.current_brightness_mean)}}</div>
      <div class="actions" tabindex="-1">
        <button class="good" data-decision="1" aria-pressed="${{done&&decision}}">1 / ${{r.review_guidance.true}}</button>
        <button class="bad" data-decision="0" aria-pressed="${{done&&!decision}}">2 / ${{r.review_guidance.false}}</button>
        <button class="warn" data-decision="">3 / 不确定</button>
      </div>
      <textarea class="note" data-note="1" placeholder="备注：例如任务等待、低纹理、图像变化点、疑似相机卡帧">${{r.note||''}}</textarea>
    </article>`;
  }}).join('');
  document.getElementById('empty').hidden=filtered.length>0;
  const reviewedCount=rows.filter(reviewed).length, trueCount=rows.filter(r=>reviewed(r)&&(r.is_true_positive==='1'||String(r.human_label).toLowerCase()==='true')).length, falseCount=reviewedCount-trueCount;
  document.getElementById('progressBar').style.width=rows.length?`${{reviewedCount/rows.length*100}}%`:'0%';
  document.getElementById('progressText').textContent=`${{reviewedCount}} / ${{rows.length}} reviewed`;
  document.getElementById('decisionText').textContent=`成立 ${{trueCount}} · 不成立 ${{falseCount}} · 待复核 ${{rows.length-reviewedCount}}`;
  cards.querySelectorAll('button[data-decision]').forEach(b=>b.onclick=()=>setDecision(b.closest('.card').dataset.id,b.dataset.decision===''?undefined:b.dataset.decision==='1'));
  cards.querySelectorAll('textarea[data-note]').forEach(t=>t.oninput=()=>{{const row=rows.find(r=>r.review_id===t.closest('.card').dataset.id);row.note=t.value;save();}});
}}
document.addEventListener('keydown',e=>{{
  if(e.target.matches('input,textarea,select'))return;
  if(e.key==='1')setDecision(currentVisibleId(),true);
  if(e.key==='2')setDecision(currentVisibleId(),false);
  if(e.key==='3')setDecision(currentVisibleId(),undefined);
  if(e.key==='n')moveUnreviewed(1);
  if(e.key==='p')moveUnreviewed(-1);
}});
ruleFilter.onchange=render;statusFilter.onchange=render;search.oninput=render;
document.getElementById('nextUnreviewed').onclick=()=>moveUnreviewed(1);
document.getElementById('prevUnreviewed').onclick=()=>moveUnreviewed(-1);
document.getElementById('exportCsv').onclick=exportCsv;
document.getElementById('importCsv').onchange=e=>{{if(e.target.files[0])importCsv(e.target.files[0]);}};
restore();render();
</script>
<noscript><p data-summary="{escaped_summary}">This reviewer requires JavaScript.</p></noscript>
</body>
</html>"""
    output_path.write_text(document, encoding="utf-8")

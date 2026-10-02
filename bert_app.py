"""
BERT: From Pretraining to Fine-tuning
SEAS 8525 - Computer Vision and Generative AI
Dr. Elbasheer

Run:  streamlit run bert_app.py
Theme: put config.toml in a folder named .streamlit next to this file.
"""

import re

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(
    page_title="BERT Walkthrough",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette (one color code for the whole app) ────────────────────────────────
NAVY   = "#002147"
GOLD   = "#FFC400"
BLUE   = "#1D4ED8"   # inputs and tokens (sentence A)
PURPLE = "#7C3AED"   # sentence B
RED    = "#E11D48"   # [MASK] and hidden states
GREEN  = "#0E9F6E"   # outputs and predictions
ORANGE = "#D97706"   # losses and training signal
MUTED  = "#475569"

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root{
  --navy:#002147; --gold:#FFC400; --bg:#F3F6FB; --border:#D9E1EC;
  --text:#0F172A; --muted:#475569;
  --blue:#1D4ED8; --purple:#7C3AED; --red:#E11D48; --green:#0E9F6E; --orange:#D97706;
}

/* ── App shell ── */
.stApp{ background:var(--bg); color:var(--text); }
[data-testid="stHeader"]{ background:transparent; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp div, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3{
  font-family:'Lexend', system-ui, sans-serif;
}
.stApp code, .stApp pre, .stApp pre *{ font-family:'JetBrains Mono', monospace !important; }
[data-testid="stMainBlockContainer"], .block-container{
  max-width:1180px; padding-top:2.2rem; padding-bottom:3rem;
}
[data-testid="stSidebar"]{ background:#FFFFFF; border-right:1px solid var(--border); }

/* ── Widgets ── */
.stApp label p{ font-size:1rem !important; font-weight:600; color:var(--navy); }
.stTextInput input, .stTextArea textarea{
  font-size:1.05rem !important; border-radius:10px !important;
}
.stButton > button{ border-radius:10px; font-weight:700; font-size:1rem; padding:.55rem 1.3rem; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--navy) !important; border:none !important; color:#fff !important;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{
  background:#0B3470 !important; color:var(--gold) !important;
}
[data-testid="stExpander"] details{
  background:#fff; border:1px solid var(--border) !important; border-radius:14px !important;
}
[data-testid="stExpander"] summary p{ font-weight:700; color:var(--navy); font-size:1.08rem; }
[data-testid="stVerticalBlockBorderWrapper"]{ border-radius:16px; }

/* Sidebar radio as a clean menu */
[data-testid="stSidebar"] [role="radiogroup"]{ gap:2px; }
[data-testid="stSidebar"] [role="radiogroup"] label{
  padding:8px 10px; border-radius:10px; width:100%;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{ background:#F3F6FB; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){ background:#EAF0FA; }
[data-testid="stSidebar"] [role="radiogroup"] label p{
  font-size:.95rem !important; font-weight:500; color:#1E293B;
}

/* ── Section header ── */
.sec-hdr{ margin:0 0 22px; }
.sec-pill{ display:inline-block; background:var(--gold); color:var(--navy); font-weight:800;
  font-size:.78rem; letter-spacing:.12em; text-transform:uppercase;
  padding:5px 13px; border-radius:999px; }
.sec-title{ font-size:2.55rem; font-weight:800; color:var(--navy); letter-spacing:-.02em;
  line-height:1.12; margin:14px 0 8px; }
.sec-sub{ font-size:1.18rem; color:var(--muted); line-height:1.5; }
.sec-rule{ height:5px; width:80px; background:var(--gold); border-radius:4px; margin-top:16px; }

/* ── Main idea ── */
.key{ background:var(--navy); border-left:9px solid var(--gold); border-radius:18px;
  padding:24px 30px; margin:4px 0 28px; box-shadow:0 10px 26px rgba(0,33,71,.16); }
.key-label{ color:var(--gold); font-size:.8rem; font-weight:800; letter-spacing:.16em;
  text-transform:uppercase; }
.key-text{ color:#fff; font-size:1.45rem; font-weight:600; line-height:1.45; margin-top:8px; }
.key-text b{ color:var(--gold); font-weight:700; }
.key.hero{ padding:32px 38px; border-left-width:12px; margin:8px 0 26px; }
.key.hero .key-text{ font-size:1.85rem; line-height:1.4; margin-top:10px; }
@media (max-width:900px){
  .key.hero{ padding:24px 26px; }
  .key.hero .key-text{ font-size:1.45rem; }
}

/* ── Body text ── */
.lead{ font-size:1.13rem; line-height:1.75; color:#1E293B; margin:0 0 22px; }
.lead b{ color:var(--navy); }
.hint{ font-size:1.02rem; color:var(--muted); line-height:1.6; margin:0 0 14px; }
.sub{ display:flex; align-items:center; gap:12px; font-size:1.55rem; font-weight:800;
  color:var(--navy); margin:38px 0 16px; letter-spacing:-.01em; }
.sub::before{ content:""; width:10px; height:28px; background:var(--gold); border-radius:3px; }

/* ── Grids and cards ── */
.grid{ display:grid; gap:18px; margin:4px 0 20px; }
.g2{ grid-template-columns:repeat(2,1fr); }
.g3{ grid-template-columns:repeat(3,1fr); }
@media (max-width:900px){ .g2,.g3{ grid-template-columns:1fr; } }
.card{ background:#fff; border:1px solid var(--border); border-top:6px solid var(--c);
  border-radius:16px; padding:18px 22px; }
.card-title{ font-size:1.18rem; font-weight:700; color:var(--c); margin-bottom:8px; }
.card-body{ font-size:1.02rem; line-height:1.65; color:#334155; }
.card-body b{ color:var(--text); }
.card code{ font-size:.9rem; background:#F1F5F9; padding:2px 6px; border-radius:6px; color:var(--navy); }

/* ── Tags ── */
.tag{ display:inline-block; padding:3px 11px; border-radius:999px; font-size:.84rem;
  font-weight:700; color:var(--c); background:color-mix(in srgb, var(--c) 13%, white);
  border:1px solid color-mix(in srgb, var(--c) 35%, white); margin:2px 4px 2px 0;
  white-space:nowrap; }

/* ── Class tip ── */
.tip{ background:#FFFBEA; border:1px solid #FDE68A; border-left:7px solid var(--gold);
  border-radius:14px; padding:16px 22px; margin:24px 0; font-size:1.04rem; line-height:1.7;
  color:#1E293B; }
.tip-label{ display:inline-block; background:var(--navy); color:var(--gold); font-size:.72rem;
  font-weight:800; letter-spacing:.12em; text-transform:uppercase; padding:3px 10px;
  border-radius:999px; margin-right:8px; }

/* ── Tables ── */
.tbl-wrap{ background:#fff; border:1px solid var(--border); border-radius:16px;
  overflow:hidden; overflow-x:auto; margin:4px 0 20px; box-shadow:0 2px 8px rgba(0,33,71,.05); }
.tbl{ width:100%; border-collapse:collapse; font-size:1.02rem; }
.tbl th{ background:var(--navy); color:#fff; font-weight:700; text-align:left;
  padding:14px 18px; font-size:.95rem; letter-spacing:.02em; }
.tbl td{ padding:14px 18px; border-top:1px solid #E8EDF4; vertical-align:top;
  line-height:1.55; color:#1E293B; }
.tbl tbody tr:nth-child(even) td{ background:#F7F9FC; }
.tbl td:first-child{ font-weight:700; color:var(--navy); }
.tbl tr.hl td{ background:#FFF6D1 !important; }
.tbl td.ncol{ font-family:'JetBrains Mono', monospace; font-weight:700; text-align:center; }
.tbl th.ncol{ text-align:center; }
.tbl .mono{ font-family:'JetBrains Mono', monospace; font-size:.92rem; color:var(--navy);
  white-space:nowrap; }

/* ── Token pills ── */
.tok-row{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:12px 0; }
.tok{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:1rem;
  padding:6px 12px; border-radius:10px; border:2px solid; line-height:1.3; }
.tok.cls { background:var(--gold); color:var(--navy); border-color:var(--gold); }
.tok.sep { background:#E2E8F0; color:var(--navy); border-color:#94A3B8; }
.tok.mask{ background:#FFE4EA; color:var(--red); border-color:var(--red); }
.tok.a   { background:#DBEAFE; color:var(--blue); border-color:#93C5FD; }
.tok.b   { background:#EDE9FE; color:var(--purple); border-color:#C4B5FD; }
.tok.out { background:#D1FAE5; color:var(--green); border-color:var(--green); }
.tok.wp  { border-style:dashed; }
.tok-legend{ display:flex; flex-wrap:wrap; gap:16px; font-size:.92rem; color:var(--muted);
  margin:6px 0 16px; }
.tok-legend span.sw{ display:inline-block; width:14px; height:14px; border-radius:4px;
  margin-right:6px; vertical-align:-2px; border:2px solid; }

/* ── Colored key-term bullets ── */
.blist{ margin:4px 0 18px; padding:0; list-style:none; }
.blist li{ position:relative; padding:9px 0 9px 28px; font-size:1.04rem; line-height:1.68;
  color:#334155; border-bottom:1px dashed var(--border); }
.blist li:last-child{ border-bottom:none; }
.blist li::before{ content:""; position:absolute; left:7px; top:17px; width:9px; height:9px;
  border-radius:50%; background:var(--c); }
.blist .k{ font-weight:800; color:var(--c); }
.blist b{ color:var(--navy); }
.blist code{ font-family:'JetBrains Mono', monospace; font-size:.89rem; background:#F1F5F9;
  padding:2px 6px; border-radius:6px; color:var(--navy); }
.blist .chain{ display:block; font-family:'JetBrains Mono', monospace; font-size:.9rem;
  color:var(--navy); background:#F7F9FC; border-radius:8px; padding:7px 11px; margin:7px 0 0; }

/* ── Numbered steps ── */
.steps{ background:#fff; border:1px solid var(--border); border-radius:16px; padding:6px 22px; }
.step{ display:flex; gap:16px; align-items:flex-start; padding:14px 0; }
.step + .step{ border-top:1px dashed var(--border); }
.num{ flex:0 0 34px; height:34px; border-radius:50%; background:var(--gold); color:var(--navy);
  font-weight:800; display:flex; align-items:center; justify-content:center; font-size:1rem; }
.step-text{ font-size:1.05rem; line-height:1.6; color:#1E293B; padding-top:4px; }
.step-text b{ color:var(--navy); }

/* ── Flow (Section 1) ── */
.flow{ display:flex; align-items:stretch; gap:14px; margin:10px 0 32px; flex-wrap:wrap; }
.flow-box{ flex:1 1 265px; background:#fff; border:1px solid var(--border);
  border-top:8px solid var(--c); border-radius:18px; padding:24px 26px 22px;
  box-shadow:0 4px 16px rgba(0,33,71,.07); }
.flow-kicker{ font-size:.82rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase;
  color:var(--c); }
.flow-title{ font-size:1.5rem; font-weight:800; color:var(--navy); margin:8px 0 14px;
  letter-spacing:-.01em; line-height:1.2; }
.flow-box ul{ margin:0; padding-left:20px; font-size:1.08rem; line-height:1.95; color:#334155; }
.flow-box ul b{ color:var(--navy); }
.flow-op{ display:flex; align-items:center; font-size:2.9rem; font-weight:800; color:var(--navy);
  padding:0 4px; }
@media (max-width:900px){ .flow-op{ justify-content:center; font-size:2.2rem; } }

/* ── Equation boxes (Section 2) ── */
.eq{ display:flex; align-items:stretch; gap:10px; flex-wrap:wrap; margin:6px 0 10px; }
.eq-box{ flex:1 1 170px; border-radius:14px; padding:16px 16px; text-align:center;
  background:color-mix(in srgb, var(--c) 10%, white); border:2px solid var(--c); }
.eq-name{ font-weight:800; font-size:1.1rem; color:var(--c); }
.eq-desc{ font-size:.95rem; color:#334155; margin-top:6px; line-height:1.45; }
.eq-op{ display:flex; align-items:center; font-size:2rem; font-weight:800; color:var(--navy); }
.eq-note{ text-align:center; font-size:1rem; color:var(--muted); margin:6px 0 4px; }

/* ── Worked embedding example (Section 2) ── */
.eg-wrap{ background:#fff; border:1px solid var(--border); border-radius:16px; padding:18px 18px;
  overflow-x:auto; margin:4px 0 8px; }
.eg{ display:grid; gap:8px 6px; align-items:center; min-width:820px; }
.eg-label{ font-weight:800; font-size:.98rem; text-align:right; padding-right:10px; }
.eg-col{ display:flex; justify-content:center; }
.eg-col .tok{ font-size:.85rem; padding:5px 8px; }
.eg-op{ font-size:1.3rem; font-weight:800; color:var(--navy); text-align:right; padding-right:14px;
  line-height:.8; }
.chip{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:.85rem; text-align:center;
  padding:7px 2px; border-radius:9px; color:var(--c); border:2px solid var(--c);
  background:color-mix(in srgb, var(--c) 10%, white); }

/* ── Joint pretraining banner (Sections 3 and 4) ── */
.dual{ background:#fff; border:1px solid var(--border); border-radius:16px;
  padding:14px 18px 15px; margin:2px 0 26px; }
.dual-label{ font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase;
  color:#64748B; margin-bottom:11px; }
.dual-row{ display:flex; align-items:stretch; gap:12px; flex-wrap:wrap; }
.dual-box{ flex:1 1 235px; border-radius:12px; padding:11px 15px;
  border:2px solid #E6EBF2; background:#F8FAFC; }
.dual-box.on{ border-color:var(--c); background:color-mix(in srgb, var(--c) 9%, white); }
.dual-n{ font-size:.71rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase;
  color:#A3AFBF; }
.dual-box.on .dual-n{ color:var(--c); }
.dual-t{ font-size:1.06rem; font-weight:800; color:#A3AFBF; margin-top:2px; }
.dual-box.on .dual-t{ color:var(--navy); }
.dual-d{ font-size:.95rem; color:#A3AFBF; line-height:1.5; margin-top:3px; }
.dual-box.on .dual-d{ color:#475569; }
.dual-op{ display:flex; align-items:center; font-size:1.7rem; font-weight:800;
  color:var(--navy); }
@media (max-width:900px){ .dual-op{ justify-content:center; } }
.dual-foot{ font-family:'JetBrains Mono', monospace; font-size:.93rem; color:var(--navy);
  text-align:center; margin-top:12px; background:#F1F5F9; border-radius:9px; padding:8px 10px; }
.dual-foot b{ color:var(--orange); }

/* ── Vertical pipeline (Section 4) ── */
.pipe{ margin:8px 0 26px; }
.pipe-row{ display:flex; gap:16px; align-items:flex-start; }
.pipe-num{ flex:0 0 42px; height:42px; border-radius:50%; background:var(--c); color:#fff;
  font-weight:800; font-size:1.1rem; display:flex; align-items:center; justify-content:center;
  box-shadow:0 3px 10px rgba(0,33,71,.14); }
.pipe-body{ flex:1; min-width:0; background:#fff; border:1px solid var(--border);
  border-left:6px solid var(--c); border-radius:14px; padding:15px 22px; }
.pipe-what{ font-size:1.22rem; font-weight:800; color:var(--navy); letter-spacing:-.01em; }
.pipe-why{ font-size:1.04rem; line-height:1.7; color:#334155; margin-top:6px; }
.pipe-why b{ color:var(--navy); }
.pipe-why code{ font-family:'JetBrains Mono', monospace; font-size:.9rem; background:#F1F5F9;
  padding:2px 7px; border-radius:6px; color:var(--navy); }
.pipe-io{ font-family:'JetBrains Mono', monospace; font-size:.98rem; line-height:1.9;
  color:var(--navy); background:#F7F9FC; border-radius:10px; padding:10px 14px;
  margin:4px 0 10px; }
.pipe-io b{ color:var(--blue); }
.pipe-link{ width:42px; text-align:center; font-size:1.5rem; color:#94A3B8;
  line-height:1.5; font-weight:700; }

/* ── Context comparison (Section 3) ── */
.cmp{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:6px 0 22px; }
@media (max-width:900px){ .cmp{ grid-template-columns:1fr; } }
.cmp-box{ background:#fff; border:1px solid var(--border); border-top:6px solid var(--c);
  border-radius:16px; padding:16px 20px; }
.cmp-kicker{ font-size:.76rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
  color:var(--c); margin-bottom:10px; }
.cmp-sent{ font-family:'JetBrains Mono', monospace; font-size:1.02rem; line-height:1.9;
  color:var(--navy); }
.cmp-blank{ background:#FFE4EA; color:var(--red); border:2px solid var(--red);
  border-radius:8px; padding:2px 8px; font-weight:700; }
.cmp-cut{ text-decoration:line-through; color:#94A3B8; }

/* ── Split bar (80/10/10) ── */
.split{ display:flex; height:54px; border-radius:12px; overflow:hidden; margin:10px 0 6px;
  border:1px solid var(--border); }
.split div{ display:flex; align-items:center; justify-content:center; color:#fff;
  font-weight:700; font-size:.95rem; text-align:center; padding:0 6px; line-height:1.2; }

/* ── FAQ ── */
.qa{ background:#fff; border:1px solid var(--border); border-radius:14px; margin:0 0 12px;
  overflow:hidden; }
.qa-q{ padding:13px 18px; font-weight:700; font-size:1.03rem; color:var(--navy);
  background:#F7F9FC; }
.qa-q span{ background:var(--gold); color:var(--navy); border-radius:6px; padding:1px 8px;
  font-size:.8rem; margin-right:8px; }
.qa-a{ padding:13px 18px; font-size:1.02rem; line-height:1.7; color:#1E293B; }

/* ── Sidebar blocks ── */
.brand{ background:var(--navy); border-radius:16px; padding:18px 18px 16px; margin:4px 0 18px; }
.brand-pill{ display:inline-block; background:var(--gold); color:var(--navy); font-weight:800;
  font-size:.72rem; letter-spacing:.12em; padding:3px 10px; border-radius:999px; }
.brand-title{ color:#fff; font-size:1.35rem; font-weight:800; margin:10px 0 4px; }
.brand-course{ color:#CBD5E1; font-size:.9rem; }
.brand-inst{ color:var(--gold); font-size:.88rem; font-weight:600; margin-top:6px; }
.side-label{ font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
  color:#64748B; margin:18px 0 8px; }
.prog-text{ font-size:.9rem; font-weight:600; color:var(--navy); margin-bottom:6px; }
.prog{ height:8px; background:#E2E8F0; border-radius:99px; overflow:hidden; }
.prog div{ height:100%; background:var(--gold); border-radius:99px; }
.legend-item{ display:flex; align-items:center; gap:10px; font-size:.9rem; color:#1E293B;
  margin:6px 0; }
.legend-item i{ width:14px; height:14px; border-radius:4px; background:var(--c);
  display:inline-block; }
.setup-cmd{ display:block; font-family:'JetBrains Mono', monospace; font-size:.78rem;
  color:var(--navy); background:#F3F6FB; border:1px solid var(--border); border-radius:8px;
  padding:6px 10px; margin-top:6px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── HTML helpers ──────────────────────────────────────────────────────────────
def html(s):
    """Render raw HTML. Leading spaces and blank lines are removed so the
    markdown parser never turns indented HTML into a code block."""
    clean = "\n".join(line.strip() for line in s.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)

def section_header(num, total, title, subtitle):
    html(f"""
    <div class="sec-hdr">
      <span class="sec-pill">Section {num} of {total}</span>
      <div class="sec-title">{title}</div>
      <div class="sec-sub">{subtitle}</div>
      <div class="sec-rule"></div>
    </div>""")

def key_idea(text, hero=False):
    html(f'<div class="key{" hero" if hero else ""}">'
         f'<div class="key-label">Main idea</div>'
         f'<div class="key-text">{text}</div></div>')

def lead(text):
    html(f'<p class="lead">{text}</p>')

def hint(text):
    html(f'<p class="hint">{text}</p>')

def sub(text):
    html(f'<div class="sub">{text}</div>')

def tip(text):
    html(f'<div class="tip"><span class="tip-label">Class tip</span>{text}</div>')

def tag(text, color):
    return f'<span class="tag" style="--c:{color};">{text}</span>'

def card(title, body, color):
    return (f'<div class="card" style="--c:{color};">'
            f'<div class="card-title">{title}</div>'
            f'<div class="card-body">{body}</div></div>')

def grid(cards, cols=2):
    html(f'<div class="grid g{cols}">{"".join(cards)}</div>')

def steps(items):
    rows = "".join(
        f'<div class="step"><div class="num">{i}</div><div class="step-text">{t}</div></div>'
        for i, t in enumerate(items, 1))
    html(f'<div class="steps">{rows}</div>')

def table(headers, rows, num_cols=(), highlight_rows=()):
    """headers: list of str. rows: list of lists (cells may contain HTML)."""
    th = "".join(f'<th class="{"ncol" if i in num_cols else ""}">{h}</th>'
                 for i, h in enumerate(headers))
    body = ""
    for r, row in enumerate(rows):
        tds = "".join(f'<td class="{"ncol" if i in num_cols else ""}">{c}</td>'
                      for i, c in enumerate(row))
        body += f'<tr class="{"hl" if r in highlight_rows else ""}">{tds}</tr>'
    html(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{th}</tr></thead>'
         f'<tbody>{body}</tbody></table></div>')

def tok(text, kind):
    return f'<span class="tok {kind}">{text}</span>'

def token_kind(token, segment):
    if token == "[CLS]":
        return "cls"
    if token == "[SEP]":
        return "sep"
    if token == "[MASK]":
        return "mask"
    base = "a" if segment == 0 else "b"
    return base + (" wp" if token.startswith("##") else "")

def token_row(tokens, segments):
    pills = "".join(tok(t, token_kind(t, s)) for t, s in zip(tokens, segments))
    html(f'<div class="tok-row">{pills}</div>')

def token_legend(show_b=True):
    items = [("#FFC400", "#FFC400", "[CLS] summary token"),
             ("#E2E8F0", "#94A3B8", "[SEP] boundary"),
             ("#DBEAFE", "#93C5FD", "Sentence A")]
    if show_b:
        items.append(("#EDE9FE", "#C4B5FD", "Sentence B"))
    items.append(("#FFFFFF", "#64748B", "##piece (dashed) = part of a word"))
    spans = "".join(
        f'<span><span class="sw" style="background:{bg};border-color:{bd};'
        f'{"border-style:dashed;" if "dashed" in lab else ""}"></span>{lab}</span>'
        for bg, bd, lab in items)
    html(f'<div class="tok-legend">{spans}</div>')

def pretrain_banner(active):
    """The two pretraining objectives, with the current one lit. Shown in BOTH Section 3
    and Section 4 so it stays obvious that they run together, not one after the other."""
    objs = [(1, RED,   "Objective 1", "Masked Language Modeling",
             "Predict words hidden behind [MASK]"),
            (2, GREEN, "Objective 2", "Next Sentence Prediction",
             "Decide whether sentence B follows A")]
    boxes = []
    for n, c, kicker, title, desc in objs:
        boxes.append(f'<div class="dual-box{" on" if n == active else ""}" style="--c:{c};">'
                     f'<div class="dual-n">{kicker}</div>'
                     f'<div class="dual-t">{title}</div>'
                     f'<div class="dual-d">{desc}</div></div>')
    html(f"""
    <div class="dual">
      <div class="dual-label">One pretraining run, both objectives at the same time</div>
      <div class="dual-row">{boxes[0]}<div class="dual-op">+</div>{boxes[1]}</div>
      <div class="dual-foot">total loss = <b>MLM loss</b> + <b>NSP loss</b>,
        back-propagated together</div>
    </div>""")

def keyrows(items):
    """Bulleted rows with a bold, colored lead-in term.
    items: list of (color, term, text). Pass term="" for a plain bullet."""
    lis = ""
    for color, term, text in items:
        lead_in = f'<span class="k">{term}</span> ' if term else ""
        lis += f'<li style="--c:{color};">{lead_in}{text}</li>'
    html(f'<ul class="blist">{lis}</ul>')

def faq(items):
    blocks = "".join(
        f'<div class="qa"><div class="qa-q"><span>Q{i}</span>{q}</div>'
        f'<div class="qa-a">{a}</div></div>' for i, (q, a) in enumerate(items, 1))
    html(blocks)


# ── Chart style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": "#D9E1EC",
    "axes.labelcolor": MUTED,
    "axes.titleweight": "bold",
    "axes.titlecolor": NAVY,
    "axes.titlesize": 13,
    "xtick.color": MUTED,
    "ytick.color": "#1E293B",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ── Model loading (imports are lazy so the app opens instantly) ───────────────
def _quiet_transformers():
    """transformers v5 prints a key-by-key load report for every checkpoint.
    The 'UNEXPECTED' rows are normal here (each head ignores the other's weights)."""
    import transformers
    transformers.logging.set_verbosity_error()

@st.cache_resource(show_spinner="Loading BERT tokenizer...")
def load_tokenizer():
    _quiet_transformers()
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained("bert-base-uncased")

@st.cache_resource(show_spinner="Loading BERT model (about 30 s the first time)...")
def load_bert():
    _quiet_transformers()
    from transformers import BertModel
    return BertModel.from_pretrained("bert-base-uncased")

@st.cache_resource(show_spinner="Loading BERT's next-sentence head...")
def load_nsp():
    _quiet_transformers()
    from transformers import BertForNextSentencePrediction
    return BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

@st.cache_resource(show_spinner="Loading fill-mask pipeline...")
def load_fill_mask():
    _quiet_transformers()
    from transformers import pipeline
    return pipeline("fill-mask", model="bert-base-uncased")

@st.cache_resource(show_spinner="Loading sentiment pipeline...")
def load_sentiment():
    _quiet_transformers()
    from transformers import pipeline
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")


# ── Navigation ────────────────────────────────────────────────────────────────
SECTIONS = [
    "The Bridge: Transformer to BERT",
    "BERT's Input: Three Embeddings",
    "Pretraining 1: Masked LM",
    "Pretraining 2: Next Sentence",
    "The [CLS] Token",
    "Fine-tuning for Tasks",
    "Where BERT Fits Real Problems",
    "BERT Variants",
]
LABELS = [f"{i}.  {s}" for i, s in enumerate(SECTIONS, 1)]
TOTAL = len(SECTIONS)

if "sec" not in st.session_state:
    st.session_state.sec = LABELS[0]

def go(delta):
    i = LABELS.index(st.session_state.sec)
    st.session_state.sec = LABELS[max(0, min(TOTAL - 1, i + delta))]

with st.sidebar:
    html("""
    <div class="brand">
      <span class="brand-pill">SEAS 8525</span>
      <div class="brand-title">BERT Walkthrough</div>
      <div class="brand-course">Computer Vision &amp; Generative AI</div>
      <div class="brand-inst">Dr. Elbasheer</div>
    </div>""")

    st.radio("Sections", LABELS, key="sec", label_visibility="collapsed")
    idx = LABELS.index(st.session_state.sec)

    html(f"""
    <div class="side-label">Progress</div>
    <div class="prog-text">Section {idx + 1} of {TOTAL}</div>
    <div class="prog"><div style="width:{(idx + 1) / TOTAL * 100:.0f}%;"></div></div>
    <div class="side-label">Quick setup</div>
    <span class="setup-cmd">pip install -r requirements.txt</span>
    <span class="setup-cmd">streamlit run bert_app.py</span>
    """)

num = idx + 1


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: THE BRIDGE
# ══════════════════════════════════════════════════════════════════════════════
if num == 1:
    section_header(1, TOTAL, "The Bridge: Transformer to BERT",
                   "You already built the encoder. BERT is a stack of them, trained in a new way.")

    key_idea("BERT is the <b>Transformer encoder</b> you already know, stacked "
             "<b>12 times</b> and pretrained to fill in hidden words using context "
             "from <b>both sides</b>.", hero=True)

    lead("In class you traced <b>\"the cat sat on the mat\"</b> through a Transformer encoder. "
         "Every word attended to every other word, so the output for <b>\"sat\"</b> knew about "
         "\"cat\" and \"mat\" at the same time. That two-way understanding is the foundation of BERT.")

    html(f"""
    <div class="flow">
      <div class="flow-box" style="--c:{BLUE};">
        <div class="flow-kicker">What you already know</div>
        <div class="flow-title">Transformer encoder</div>
        <ul><li><b>Multi-head self-attention</b></li><li>Feed-forward network</li>
        <li>LayerNorm and residuals</li><li>One rich vector per word</li></ul>
      </div>
      <div class="flow-op">+</div>
      <div class="flow-box" style="--c:{ORANGE};">
        <div class="flow-kicker">What BERT adds</div>
        <div class="flow-title">Stacking and pretraining</div>
        <ul><li><b>12 encoder blocks</b> (24 in Large)</li><li><b>768-dim</b> hidden size</li>
        <li>Special tokens [CLS] [SEP] [MASK]</li><li><b>Self-supervised</b> pretraining</li></ul>
      </div>
      <div class="flow-op">=</div>
      <div class="flow-box" style="--c:{GREEN};">
        <div class="flow-kicker">The result</div>
        <div class="flow-title">A reusable language model</div>
        <ul><li><b>110M</b> pretrained parameters</li><li>Fine-tune for almost any task</li>
        <li>Reads context in <b>both directions</b></li><li>State of the art in 2018</li></ul>
      </div>
    </div>""")

    tip("BERT comes from the 2018 paper <i>BERT: Pre-training of Deep Bidirectional "
        "Transformers for Language Understanding</i> by Devlin et al. at Google. "
        "The key word is <b>bidirectional</b>. GPT reads left to right only; "
        "BERT reads the whole sentence at once.")

    with st.expander("Frequently asked questions"):
        faq([
            ("Is BERT an encoder or a decoder?",
             "Encoder only. It produces a rich vector for each input token but does not "
             "generate new text by itself. Generating text is a decoder's job, which is "
             "why BERT is paired with a task head or a separate decoder model rather than "
             "writing anything on its own."),
            ("How is BERT different from GPT?",
             "GPT is decoder-only and trained to predict the next token, left to right. "
             "BERT is encoder-only and trained to predict hidden tokens using both sides. "
             "GPT generates text; BERT understands it."),
            ("Why pretrain? Can we just train BERT on our task?",
             "BERT-Base has 110 million parameters. Training that from scratch needs billions "
             "of words and days of compute. Pretraining on Wikipedia and BooksCorpus gives "
             "general language understanding once; fine-tuning on your task then takes hours."),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BERT'S INPUT
# ══════════════════════════════════════════════════════════════════════════════
elif num == 2:
    section_header(2, TOTAL, "BERT's Input: Three Embeddings",
                   "What each token looks like before the first encoder layer runs")

    key_idea("Every token enters BERT as the <b>sum of three vectors</b>: "
             "<b>what</b> the word is, <b>which sentence</b> it belongs to, "
             "and <b>where</b> it sits.")

    lead("Unlike the original Transformer, BERT uses <b>three embeddings</b> for every "
         "token: a <b>token</b> embedding, a <b>segment</b> embedding and a "
         "<b>position</b> embedding. Each one is a learned lookup table, and the three "
         "vectors are added together.")

    table(
        ["Model", "Embeddings used"],
        [
            ["Original Transformer",
             f"{tag('Token', BLUE)} + {tag('Position (fixed sine waves)', ORANGE)}"],
            ["BERT",
             f"{tag('Token', BLUE)} + {tag('Segment', PURPLE)} + "
             f"{tag('Position (learned)', ORANGE)}"],
        ],
        highlight_rows=(1,),
    )

    sub("The three embeddings")
    grid([
        card("1. Token embedding",
             f"{tag('Table: 30,522 rows x 768', BLUE)}<br>"
             "<b>Tells BERT what the word is.</b> Think of a dictionary: each token ID picks one row of the table. "
             "The same word always gets the same row, wherever it appears.<br><br>"
             "<b>Without it:</b> BERT would not know which words it is reading.",
             BLUE),
        card("2. Segment embedding",
             f"{tag('Table: 2 rows x 768', PURPLE)}<br>"
             "<b>Tells BERT which sentence the token is in.</b> Think of team jerseys: every token in sentence A wears jersey A; "
             "every token in sentence B wears jersey B.<br><br>"
             "<b>Without it:</b> in <i>question [SEP] passage</i>, BERT could not tell "
             "which words belong to the question.",
             PURPLE),
        card("3. Position embedding",
             f"{tag('Table: 512 rows x 768', ORANGE)}<br>"
             "<b>Tells BERT where the token sits.</b> Think of seat numbers: position 0 gets row 0, position 1 gets row 1, "
             "and so on up to 511.<br><br>"
             "<b>Without it:</b> self-attention ignores word order, so "
             "<i>dog bites man</i> and <i>man bites dog</i> would look identical.",
             ORANGE),
    ], cols=3)

    # ── Worked example, column by column ─────────────────────────────────────
    sub("Watch it happen on one input")
    hint("Two sentences packed into one input: <b>the cat sat on the mat</b> and "
         "<b>it slept</b>. Read each column top to bottom.")

    ex_tokens = ["[CLS]", "the", "cat", "sat", "on", "the", "mat", "[SEP]", "it", "slept", "[SEP]"]
    ex_segs   = [0] * 8 + [1] * 3
    same      = {1, 5}   # the two "the" tokens

    def chip(text, color, extra=""):
        return f'<div class="chip" style="--c:{color};{extra}">{text}</div>'

    def emb_name(t):
        return {"[CLS]": "CLS", "[SEP]": "SEP"}.get(t, t)

    cols = len(ex_tokens)
    cells = '<div class="eg-label"></div>'
    for i, (t, s) in enumerate(zip(ex_tokens, ex_segs)):
        cls = "eg-col same" if i in same else "eg-col"
        cells += f'<div class="{cls}">{tok(t, token_kind(t, s))}</div>'
    rows = [
        ("Token", BLUE,   lambda i, t, s: f"E<sub>{emb_name(t)}</sub>"),
        ("Segment", PURPLE, lambda i, t, s: f"E<sub>{'A' if s == 0 else 'B'}</sub>"),
        ("Position", ORANGE, lambda i, t, s: f"E<sub>{i}</sub>"),
    ]
    for r, (name, color, fn) in enumerate(rows):
        if r > 0:
            cells += f'<div class="eg-op">+</div>' + '<div></div>' * cols
        cells += f'<div class="eg-label" style="color:{color};">{name}</div>'
        for i, (t, s) in enumerate(zip(ex_tokens, ex_segs)):
            ring = "box-shadow:0 0 0 3px #FFC400;" if (name == "Token" and i in same) else ""
            cells += chip(fn(i, t, s), color, ring)
    cells += '<div class="eg-op">=</div>' + '<div></div>' * cols
    cells += f'<div class="eg-label" style="color:{NAVY};">Input vector</div>'
    for i in range(cols):
        cells += chip(f"x<sub>{i}</sub>", NAVY, "background:#002147;color:#fff;")

    html(f'<div class="eg-wrap"><div class="eg" style="grid-template-columns:'
         f'120px repeat({cols}, minmax(58px, 1fr));">{cells}</div></div>')

    tip("Look at the two <b>the</b> tokens (gold ring). They get the <b>exact same</b> token "
        "embedding, E<sub>the</sub>. Only their position embeddings differ "
        "(E<sub>1</sub> and E<sub>5</sub>), so they enter the encoder as two "
        "<b>different</b> vectors. Notice too that the first [SEP] belongs to sentence A.")

    # ── Why add ──────────────────────────────────────────────────────────────
    sub("Why add them instead of stacking them side by side?")
    hint("A tiny made-up example with 4 numbers per vector instead of 768:")
    toy = [("Token: cat", BLUE,     [0.2, -0.5,  0.8,  0.1]),
           ("Segment: A", PURPLE,   [0.1,  0.1, -0.1,  0.0]),
           ("Position: 2", ORANGE,  [0.0,  0.3,  0.1, -0.2]),
           ("Sum = input", NAVY,    [0.3, -0.1,  0.8, -0.1])]
    table(["Vector", "dim 1", "dim 2", "dim 3", "dim 4"],
          [[tag(n, c)] + [f"{v:+.1f}" for v in vals] for n, c, vals in toy],
          num_cols=(1, 2, 3, 4), highlight_rows=(3,))
    steps([
        "<b>Same width everywhere.</b> Adding keeps each token at 768 numbers, the width "
        "every encoder layer expects. Stacking would make it 2,304.",
        "<b>Nothing important is lost.</b> The three tables are learned together, so "
        "training spreads word, sentence and position information across the 768 numbers "
        "in ways the layers can still pull apart.",
        "<b>One more step.</b> BERT applies <b>LayerNorm</b> (and dropout) to the sum. "
        "The result is what encoder layer 1 actually receives.",
    ])

    html(f"""
    <div class="key" style="background:#fff;border:1px solid #D9E1EC;border-left:9px solid {GOLD};
         box-shadow:none;text-align:center;">
      <div class="key-label" style="color:{NAVY};">The full recipe</div>
      <div class="key-text" style="color:{NAVY};font-family:'JetBrains Mono',monospace;font-size:1.15rem;">
        x<sub>i</sub> = LayerNorm(
        <span style="color:{BLUE};">E<sub>token</sub></span> +
        <span style="color:{PURPLE};">E<sub>segment</sub></span> +
        <span style="color:{ORANGE};">E<sub>position</sub></span> )
      </div>
      <div class="hint" style="margin:10px 0 0;">
        The three tables hold about <b>23.8 million</b> learned numbers, roughly a fifth of
        BERT-Base's 110 million. The original Transformer used fixed position waves and
        had no segment table.</div>
    </div>""")

    # ── Special tokens ───────────────────────────────────────────────────────
    sub("The three special tokens")
    grid([
        card("[CLS] the summary token",
             "Always the <b>first token</b>. After 12 layers its output vector is used as "
             "a summary of the whole input for sentence-level tasks. Think of it as the "
             "note-taker for the sentence.", NAVY),
        card("[SEP] the separator",
             "Marks the <b>end of a sentence</b>. With two sentences (question and context, "
             "for example) one [SEP] follows each, so BERT can tell them apart.", "#475569"),
        card("[MASK] the blank",
             "Used <b>only in pretraining</b> to hide a word BERT must predict. It never "
             "appears during fine-tuning. The 80/10/10 rule in Section 3 exists to soften "
             "that mismatch.", RED),
    ], cols=3)

    # ── Live demo ────────────────────────────────────────────────────────────
    sub("Try it: tokenize, then look inside the real vectors")
    hint("Type a sentence, or two, and press Tokenize. Try a long or unusual word such as "
         "<b>embeddings</b> to see WordPiece split it into pieces marked with ##.")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            sent1 = st.text_input("Sentence A", value="The cat sat on the mat.")
        with c2:
            sent2 = st.text_input("Sentence B (optional)", value="")
        if st.button("Tokenize", type="primary"):
            st.session_state.tok_inputs = (sent1, sent2)

    # Results stay visible while the student picks tokens below
    if st.session_state.get("tok_inputs") == (sent1, sent2):
        tokenizer = load_tokenizer()
        enc = (tokenizer(sent1, sent2, return_tensors="pt") if sent2.strip()
               else tokenizer(sent1, return_tensors="pt"))
        tokens  = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        seg_ids = enc["token_type_ids"][0].tolist()
        ids     = enc["input_ids"][0].tolist()

        sub("Result")
        token_row(tokens, seg_ids)
        token_legend(show_b=bool(sent2.strip()))

        table(
            ["Position", "Token", "Token ID", "Segment"],
            [[str(i), f'<span class="mono">{t}</span>', str(tid),
              tag("A", BLUE) if s == 0 else tag("B", PURPLE)]
             for i, (t, tid, s) in enumerate(zip(tokens, ids, seg_ids))],
            num_cols=(0, 2),
        )

        sub("Look inside: the three vectors for one token")
        hint("These are BERT's real learned numbers, first 8 of 768. With the default "
             "sentence, pick each <b>the</b> in turn: the token row stays the same while "
             "the position row changes.")

        pick = st.selectbox("Token to inspect", list(range(len(tokens))),
                            index=min(1, len(tokens) - 1),
                            format_func=lambda i: f"position {i}:   {tokens[i]}")

        import torch
        emb = load_bert().embeddings
        with torch.no_grad():
            tv = emb.word_embeddings.weight[ids[pick]]
            sv = emb.token_type_embeddings.weight[seg_ids[pick]]
            pv = emb.position_embeddings.weight[pick]
            total  = tv + sv + pv
            normed = emb.LayerNorm(total)

        seg_name = "A" if seg_ids[pick] == 0 else "B"
        vec_rows = [
            (f"Token: {tokens[pick]}", BLUE,   tv),
            (f"Segment: {seg_name}",   PURPLE, sv),
            (f"Position: {pick}",      ORANGE, pv),
            ("Sum",                    NAVY,   total),
            ("After LayerNorm",        GREEN,  normed),
        ]
        table(["Vector"] + [f"d{j}" for j in range(8)],
              [[tag(n, c)] + [f"{v:+.3f}" for v in vec.detach().numpy()[:8]]
               for n, c, vec in vec_rows],
              num_cols=tuple(range(1, 9)), highlight_rows=(4,))
        hint("Check any column: token + segment + position equals the Sum row. "
             "LayerNorm then rescales the sum so its 768 numbers have mean 0 and "
             "standard deviation 1 before learned scaling.")

    with st.expander("Show the tokenization and embedding code"):
        st.code("""
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased")

enc = tokenizer("The cat sat on the mat.", return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
# ['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', '.', '[SEP]']

emb = model.embeddings                        # the three lookup tables
print(emb.word_embeddings.weight.shape)       # (30522, 768)  token table
print(emb.token_type_embeddings.weight.shape) # (2, 768)      segment table
print(emb.position_embeddings.weight.shape)   # (512, 768)    position table

i   = 1                                       # the first "the"
tid = enc["input_ids"][0, i]
seg = enc["token_type_ids"][0, i]
with torch.no_grad():
    x = (emb.word_embeddings.weight[tid]
         + emb.token_type_embeddings.weight[seg]
         + emb.position_embeddings.weight[i])
    x = emb.LayerNorm(x)                      # this is what layer 1 receives
print(x.shape)                                # (768,)
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MASKED LANGUAGE MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif num == 3:
    section_header(3, TOTAL, "Pretraining Task 1: Masked Language Model",
                   "How BERT learns from plain text with no human labels")

    pretrain_banner(1)

    key_idea("Hide <b>15% of the words</b> and make BERT guess them. To guess well, "
             "it has to read the words on <b>both sides</b> of the blank.")

    html(f"""
    <div class="tok-row" style="margin:6px 0 4px;">
      {tok('[CLS]','cls')}{tok('the','a')}{tok('cat','a')}{tok('[MASK]','mask')}
      {tok('on','a')}{tok('the','a')}{tok('mat','a')}{tok('[SEP]','sep')}
      <span style="font-size:1.8rem;font-weight:800;color:{NAVY};margin:0 6px;">&#8594;</span>
      {tok('sat','out')}
    </div>
    <p class="hint">Left context <b>"the cat"</b> and right context <b>"on the mat"</b>
    together make <b style="color:{GREEN};">"sat"</b> the obvious answer.</p>
    """)

    sub("The 15% rule")
    html(f"""
    <div class="split">
      <div style="width:80%;background:{RED};">80%</div>
      <div style="width:10%;background:{ORANGE};">10%</div>
      <div style="width:10%;background:#64748B;">10%</div>
    </div>
    <div class="tok-row" style="gap:6px;">
      {tag('80% replaced with [MASK]: "the cat [MASK] on the mat"', RED)}
      {tag('10% random word: "the cat apple on the mat"', ORANGE)}
      {tag('10% left unchanged: "the cat sat on the mat"', '#64748B')}
    </div>""")

    steps([
        "<b>Pick 15% of the tokens</b> at random. In a six-word sentence, about one word.",
        "<b>Corrupt each picked token</b> using the 80/10/10 split above.",
        "<b>Predict the original word</b> at those positions only. The loss ignores "
        "every other position.",
    ])

    tip("<b>Why not always use [MASK]?</b> Because [MASK] never appears during fine-tuning. "
        "The random and unchanged cases force BERT to build a usable vector for "
        "<b>every</b> token, not just the blanks.")

    sub("Try it: hide a word, then take away the right side")
    hint("Press <b>Load sentence</b>, then <b>click any word</b> to hide it. BERT guesses it "
         "twice: once from the <b>whole sentence</b>, once with every word after the blank "
         "<b>cut away</b>, which is all a left-to-right model like GPT would ever see. The "
         "stop stays on both runs, so BERT is always asked for a <b>word</b>, never "
         "for punctuation.")

    with st.container(border=True):
        mlm_sentence = st.text_input("Sentence", value="The cat sat on the mat.",
                                     key="mlm_sent")
        if st.button("Load sentence", type="primary"):
            st.session_state.mlm_loaded = mlm_sentence
            st.session_state.mlm_pick = None

    # The sentence-final punctuation is held back and re-attached to BOTH runs. Without
    # it, "The cat [MASK]" asks BERT to end a sentence, and it answers "." at 80%, which
    # says nothing about context. With it, both runs must name a word.
    body, tail = re.match(r"^(.*?)([^\w\s]*)\s*$", mlm_sentence.strip(), re.S).groups()
    words = body.strip().split()
    tail  = tail or "."

    if st.session_state.get("mlm_loaded") == mlm_sentence and words:
        if st.session_state.get("mlm_pick") is None:
            st.session_state.mlm_pick = min(2, len(words) - 1)

        sub("Click a word to hide it")
        PER_ROW = 8
        for start in range(0, len(words), PER_ROW):
            row = list(range(start, min(start + PER_ROW, len(words))))
            cols = st.columns(PER_ROW)
            for col, i in zip(cols, row):
                with col:
                    if st.button(words[i], key=f"mlm_w{i}", use_container_width=True,
                                 type="primary" if st.session_state.mlm_pick == i
                                 else "secondary"):
                        st.session_state.mlm_pick = i

        pick      = min(st.session_state.mlm_pick, len(words) - 1)
        true_word = re.sub(r"^\W+|\W+$", "", words[pick]).lower()
        both      = " ".join(words[:pick] + ["[MASK]"] + words[pick + 1:]) + tail
        left      = " ".join(words[:pick] + ["[MASK]"]) + tail
        cut       = " ".join(words[pick + 1:])
        same      = (pick == len(words) - 1)

        blank = '<span class="cmp-blank">[MASK]</span>'
        html(f"""
        <div class="cmp">
          <div class="cmp-box" style="--c:{GREEN};">
            <div class="cmp-kicker">Both sides: what BERT gets</div>
            <div class="cmp-sent">{both.replace('[MASK]', blank)}</div>
          </div>
          <div class="cmp-box" style="--c:{MUTED};">
            <div class="cmp-kicker">Left side only: GPT-style</div>
            <div class="cmp-sent">{left[:-len(tail)].replace('[MASK]', blank)}
              <span class="cmp-cut">{cut}</span>{tail}</div>
          </div>
        </div>""")

        with st.spinner("Running BERT fill-mask..."):
            fm       = load_fill_mask()
            res_both = fm(both, top_k=5)
            res_left = res_both if same else fm(left, top_k=5)

        wb = [r["token_str"].strip() for r in res_both]
        sb = [r["score"] * 100 for r in res_both]
        wl = [r["token_str"].strip() for r in res_left]
        sl = [r["score"] * 100 for r in res_left]

        def prob_of(ws, ss, word):
            lower = [w.lower() for w in ws]
            return ss[lower.index(word)] if word in lower else None

        pb, pl = prob_of(wb, sb, true_word), prob_of(wl, sl, true_word)

        if same:
            note = ("You hid the <b>last</b> word, so there was no right context to take "
                    "away and both runs are identical. Pick a word nearer the start.")
        elif pb is not None and pl is None:
            note = (f"Only the two-sided run recovered the hidden word "
                    f"<b>{true_word}</b> ({pb:.1f}%). Dropping the right side pushed it out "
                    f"of the top 5 entirely. <b>That is what bidirectional buys you.</b>")
        elif pb is not None and pl is not None:
            note = (f"Both runs found <b>{true_word}</b>, but look at the confidence: "
                    f"<b>{pb:.1f}%</b> with the full sentence against <b>{pl:.1f}%</b> from "
                    f"the left side alone.")
        else:
            note = (f"Neither run recovered <b>{true_word}</b> in its top 5. Try a word the "
                    f"rest of the sentence pins down harder, such as the verb in "
                    f"\"the cat sat on the mat\".")

        sub("What each run guessed")
        html(f"""
        <div class="key" style="border-left-color:{GREEN};">
          <div class="key-label" style="color:#6EE7B7;">Top guess, side by side</div>
          <div class="key-text">
            Both sides &#8594; <b style="color:#6EE7B7;">{wb[0]}</b>
            <span style="font-size:1rem;color:#CBD5E1;">({sb[0]:.1f}%)</span>
            &nbsp;&nbsp;&#183;&nbsp;&nbsp;
            Left only &#8594; <b style="color:#FDA4AF;">{wl[0]}</b>
            <span style="font-size:1rem;color:#CBD5E1;">({sl[0]:.1f}%)</span>
          </div>
          <div style="color:#E2E8F0;font-size:1.04rem;line-height:1.65;margin-top:12px;">
            {note}</div>
        </div>""")

        panels = [(wb, sb, GREEN, "#A7E3CB", "Both sides: BERT")]
        if not same:
            panels.append((wl, sl, ORANGE, "#F6DFB5", "Left side only: GPT-style"))

        hi = max(max(sb), max(sl)) * 1.20
        fig, axes = plt.subplots(1, len(panels), figsize=(11.5 if len(panels) == 2 else 7, 3.8),
                                 squeeze=False)
        for ax, (ws, ss, strong, weak, title) in zip(axes[0], panels):
            face = [strong if w.lower() == true_word else weak for w in ws]
            bars = ax.barh(ws[::-1], ss[::-1], color=face[::-1], height=0.62)
            for bar, sc, w in zip(bars, ss[::-1], ws[::-1]):
                if w.lower() == true_word:
                    bar.set_edgecolor(GOLD)
                    bar.set_linewidth(2.5)
                ax.text(bar.get_width() + hi * 0.012,
                        bar.get_y() + bar.get_height() / 2,
                        f"{sc:.1f}%", va="center", fontsize=10.5, color="#1E293B")
            ax.set_xlim(0, hi)
            ax.set_xlabel("Probability (%)")
            ax.tick_params(axis="y", labelsize=12)
            ax.set_title(title, loc="left")
            style_axes(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        hint("Both panels share one probability scale, so a shorter bar really is a less "
             "confident guess. The <b>gold outline</b> marks the word that was actually hidden.")

    with st.expander("Show the fill-mask code"):
        st.code("""
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-uncased")

sentence = "The cat sat on the mat"      # note: no full stop yet
i        = 2                             # hide "sat"
words    = sentence.split()

# Re-attach the full stop to BOTH runs. Skip it and the left-only run becomes
# "The cat [MASK]", where the most likely token is the end of the sentence:
#     .  80.3%   ;  12.8%   !  5.0%
# That is BERT answering a punctuation question, not a vocabulary one.
both = " ".join(words[:i] + ["[MASK]"] + words[i + 1:]) + "."
left = " ".join(words[:i] + ["[MASK]"]) + "."

for name, text in [("both sides", both), ("left only", left)]:
    print(f"{name:11s} {text}")
    for r in fill_mask(text, top_k=5):
        print(f"    {r['token_str']:11s} {r['score']*100:5.1f}%")

# both sides  The cat [MASK] on the mat.
#     sat          16.8%      <-- recovered, and ranked first
#     lay           8.3%
#     was           6.2%
# left only   The cat [MASK].
#     asked         5.3%      <-- "sat" is gone from the top 5 entirely
#     said          5.1%
#     barked        4.2%
#
# Same model, same blank, same 12 layers. The only thing that changed is whether
# BERT could read "on the mat". That is what the word bidirectional is doing.
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: NEXT SENTENCE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif num == 4:
    section_header(4, TOTAL, "Pretraining Task 2: Next Sentence Prediction",
                   "How BERT learns relationships between two sentences")

    pretrain_banner(2)

    key_idea("Show BERT two sentences and ask <b>one yes-or-no question</b>: "
             "does B actually come next? Half of its training pairs really do follow, "
             "half are random, and telling them apart is what teaches BERT how "
             "sentences connect.")

    lead("Masked LM works <b>inside</b> a single sentence. But plenty of real tasks come as "
         "a <b>pair</b>: a question and a passage, a premise and a hypothesis, two sentences "
         "that might mean the same thing. NSP is the task that teaches BERT to relate two "
         "sentences at once.")

    # ── One spine, walked once ───────────────────────────────────────────────
    sub("The whole flow, start to finish")
    s2_pills = (tok('[CLS]', 'cls') + tok('the', 'a') + tok('cat', 'a') + tok('sat', 'a')
                + tok('on', 'a') + tok('the', 'a') + tok('mat', 'a') + tok('[SEP]', 'sep')
                + tok('it', 'b') + tok('fell', 'b') + tok('asleep', 'b') + tok('[SEP]', 'sep'))

    stages = [
        (BLUE, "Two sentences go in",
         '<div class="pipe-io"><b>A</b>&nbsp; The cat sat on the mat.<br>'
         '<b>B</b>&nbsp; It fell asleep in the sun.</div>'
         "BERT builds these pairs out of raw text by itself. Nobody labels them."),
        (PURPLE, "Pack both into one sequence",
         f'<div class="tok-row" style="margin:6px 0 10px;">{s2_pills}</div>'
         "One <b>[SEP]</b> closes each sentence, and a <b>segment ID</b> stamps every token "
         "as belonging to A or B. Now it is a single input, not two."),
        (ORANGE, "Run all 12 encoder layers",
         "Every token attends to every other token, straight <b>across</b> the [SEP] "
         "boundary, so words in B can look back at words in A. By the top layer the slot at "
         "<b>position 0</b> has read both sentences, which makes it the one place a summary "
         "of the <b>whole pair</b> can live. That slot is the token written <b>[CLS]</b>."),
        (GREEN, "Read position 0 through a 2-way classifier",
         "Take the 768 numbers sitting at position 0, push them through "
         "<code>Linear(768 &#8594; 2)</code>, then softmax. Out come two probabilities that "
         "add up to 100%: " + tag("IsNext", GREEN) + tag("NotNext", RED)),
    ]

    pipe = ""
    for i, (color, what, why) in enumerate(stages, 1):
        if i > 1:
            pipe += '<div class="pipe-link"><span>&#8595;</span></div>'
        pipe += (f'<div class="pipe-row">'
                 f'<div class="pipe-num" style="--c:{color};">{i}</div>'
                 f'<div class="pipe-body" style="--c:{color};">'
                 f'<div class="pipe-what">{what}</div>'
                 f'<div class="pipe-why">{why}</div>'
                 f'</div></div>')
    html(f'<div class="pipe">{pipe}</div>')

    # ── Dataset construction, kept separate from the forward pass ────────────
    sub("Where the right answer comes from")
    hint("Stages 1 to 4 are the <b>forward pass</b>. The <b>label</b> is a separate story: "
         "BERT invents it while reading the corpus, which is why NSP needs no annotators.")
    grid([
        card("50% of pairs &#8594; IsNext &#10003;",
             "B is <b>the sentence that really came next</b> in the document.<br><br>"
             '<span class="mono">A: The cat sat on the mat.</span><br>'
             '<span class="mono">B: It fell asleep in the sun.</span>', GREEN),
        card("50% of pairs &#8594; NotNext &#10007;",
             "B is a <b>random sentence</b> pulled from somewhere else in the corpus."
             "<br><br>"
             '<span class="mono">A: The cat sat on the mat.</span><br>'
             '<span class="mono">B: Stocks fell sharply in Tokyo.</span>', RED),
    ])

    # ── Demo: the same four stages, with real numbers ────────────────────────
    sub("Try it: send your own pair through stages 1 to 4")
    hint("This runs <b>BertForNextSentencePrediction</b>, the very head BERT was pretrained "
         "with. Pick a preset or type your own pair.")

    NSP_PRESETS = {
        "Coherent follow-on":
            ("The cat sat on the mat.", "It fell asleep in the sun."),
        "Unrelated sentence":
            ("The cat sat on the mat.", "Stocks fell sharply in Tokyo today."),
        "Same topic, but not a follow-on":
            ("The cat sat on the mat.",
             "Cats have been kept as household pets for thousands of years."),
    }

    st.session_state.setdefault("nsp_a", "The cat sat on the mat.")
    st.session_state.setdefault("nsp_b", "It fell asleep in the sun.")

    with st.container(border=True):
        pcols = st.columns(len(NSP_PRESETS))
        for col, (pname, (pa, pb)) in zip(pcols, NSP_PRESETS.items()):
            with col:
                if st.button(pname, key=f"nsp_preset_{pname}", use_container_width=True):
                    st.session_state.nsp_a = pa
                    st.session_state.nsp_b = pb
                    st.session_state.nsp_pair = (pa, pb)

        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Sentence A", key="nsp_a")
        with c2:
            st.text_input("Sentence B", key="nsp_b")
        if st.button("Score this pair", type="primary"):
            st.session_state.nsp_pair = (st.session_state.nsp_a, st.session_state.nsp_b)

    nsp_a, nsp_b = st.session_state.nsp_a, st.session_state.nsp_b

    if st.session_state.get("nsp_pair") == (nsp_a, nsp_b):
        if not nsp_a.strip() or not nsp_b.strip():
            st.error("Please fill in both sentences.")
        else:
            import torch
            with st.spinner("Running BERT and its next-sentence classifier..."):
                tokenizer = load_tokenizer()
                enc = tokenizer(nsp_a, nsp_b, return_tensors="pt")
                with torch.no_grad():
                    logits = load_nsp()(**enc).logits[0]
                probs = logits.softmax(-1)
            p_is, p_not = probs[0].item() * 100, probs[1].item() * 100

            tokens  = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            seg_ids = enc["token_type_ids"][0].tolist()
            a_n, b_n = seg_ids.count(0), seg_ids.count(1)

            # Stage 2, for real
            sub("Stage 2: the sequence BERT actually read")
            token_row(tokens, seg_ids)
            token_legend()
            html(f"""<div class="tok-row">
              {tag(f'Segment A: {a_n} tokens (with [CLS] and first [SEP])', BLUE)}
              {tag(f'Segment B: {b_n} tokens (with last [SEP])', PURPLE)}
              {tag(f'Total: {len(tokens)} tokens', NAVY)}</div>""")

            # Stage 4, for real
            label  = "IsNext" if p_is >= p_not else "NotNext"
            colr   = GREEN if label == "IsNext" else RED
            accent = "#6EE7B7" if label == "IsNext" else "#FDA4AF"

            sub("Stage 4: what came out of position 0")
            html(f"""
            <div class="key" style="border-left-color:{colr};">
              <div class="key-label" style="color:{accent};">Verdict</div>
              <div class="key-text">{label}
                <span style="font-size:1.05rem;color:#CBD5E1;">
                &nbsp;{max(p_is, p_not):.1f}% confident</span>
              </div>
            </div>""")

            def nsp_seg(pct, color, name):
                txt = f"{name} {pct:.1f}%" if pct >= 14 else ""
                return f'<div style="width:{pct:.2f}%;background:{color};">{txt}</div>'

            html(f'<div class="split">{nsp_seg(p_is, GREEN, "IsNext")}'
                 f'{nsp_seg(p_not, RED, "NotNext")}</div>')

    tip("BERT was originally pretrained on <b>two objectives at the same time</b>: "
        "<b>MLM</b> and <b>NSP</b>. <b>RoBERTa</b> (2019) later found NSP was <b>not "
        "necessary</b>, and could even be <b>unhelpful</b> depending on how the training "
        "data was constructed, so RoBERTa <b>dropped NSP entirely</b> and pretrained on "
        "masked-language modeling alone. Most later variants followed (Section 8). For a "
        "hint at why, run the <b>same topic</b> preset: B plainly does not follow A, yet "
        "BERT still answers IsNext, because NSP leans heavily on simple topic overlap.")

    sub("Pretraining versus fine-tuning")
    table(
        ["Phase", "Objective", "Data it needs"],
        [
            ["Original BERT pretraining",
             f"{tag('MLM', RED)} + {tag('NSP', GREEN)} jointly, in one run",
             "Raw text only. Self-supervised, so <b>no human labels</b>."],
            ["BERT fine-tuning",
             f"{tag('One task-specific objective', BLUE)}",
             "Your <b>labeled</b> examples for that one task (Section 6)."],
        ],
        highlight_rows=(0,),
    )

    with st.expander("Show the NSP code"):
        st.code("""
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

def nsp_score(a, b):
    enc = tokenizer(a, b, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits          # shape (1, 2)
    probs = logits.softmax(-1)[0]
    return {"IsNext": probs[0].item(), "NotNext": probs[1].item()}

print(nsp_score("The cat sat on the mat.", "It fell asleep in the sun."))
# {'IsNext': 0.99, 'NotNext': 0.01}   a real follow-on

print(nsp_score("The cat sat on the mat.", "Stocks fell sharply today."))
# {'IsNext': 0.01, 'NotNext': 0.99}   a random sentence

print(nsp_score("The cat sat on the mat.",
                "Cats have been kept as household pets for thousands of years."))
# {'IsNext': 0.98, 'NotNext': 0.02}   same TOPIC, but it does not follow.
# NSP is fooled, because topic overlap is most of what it learned to detect.
# RoBERTa dropped NSP for exactly this reason.
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: THE [CLS] TOKEN
# ══════════════════════════════════════════════════════════════════════════════
elif num == 5:
    section_header(5, TOTAL, "The [CLS] Token",
                   "Which of BERT's outputs you use depends on the question you ask")

    key_idea("BERT outputs <b>one 768-number vector per token</b>. For a whole-sentence "
             "answer, read the vector at <b>[CLS]</b>. For word-level answers, read "
             "<b>every token's</b> vector.")

    lead("[CLS] attends to every other token in all 12 layers. Pretraining (NSP) teaches it "
         "to carry sentence-level information, and fine-tuning shapes it for your task.")

    sub("Which output goes where")
    table(
        ["Task type", "Output used", "Shape", "Examples"],
        [
            ["Sentence classification", tag("[CLS] only", NAVY),
             '<span class="mono">(B, 768)</span>', "Sentiment, spam detection"],
            ["Token classification", tag("Every token", GREEN),
             '<span class="mono">(B, seq, 768)</span>', "Named entities, part-of-speech tags"],
            ["Question answering", tag("Every context token", GREEN),
             '<span class="mono">(B, seq, 768)</span>', "Predict the start and end of the answer span"],
            ["Sentence-pair tasks", tag("[CLS] of the pair", NAVY),
             '<span class="mono">(B, 768)</span>', "Paraphrase detection, inference, similarity"],
        ],
    )

    sub("Try it: run BERT and inspect its outputs")
    hint("Runs bert-base-uncased on your sentence and compares the output vectors of "
         "every pair of tokens.")

    with st.container(border=True):
        embed_sent = st.text_input("Sentence", value="The cat sat on the mat.", key="embed_sent")
        center = st.checkbox(
            "Remove the shared direction first (subtract the mean vector)", value=True,
            help="BERT's last-layer vectors all point in a similar direction, so raw "
                 "cosine similarities are high for every pair. Subtracting the mean "
                 "makes the real differences visible.")
        run = st.button("Run BERT", type="primary")

    if run:
        import torch
        with st.spinner("Running a forward pass through 12 layers..."):
            tokenizer  = load_tokenizer()
            model_bert = load_bert()
            enc = tokenizer(embed_sent, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            with torch.no_grad():
                hidden = model_bert(**enc).last_hidden_state[0]   # (seq_len, 768)

        html(f"""<div class="tok-row">
          {tag(f'Output shape: {tuple(hidden.shape)}', GREEN)}
          {tag(f'{len(tokens)} tokens x 768 numbers', NAVY)}</div>""")

        cls_vec = hidden[0].numpy()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("[CLS] mean", f"{cls_vec.mean():.3f}")
        c2.metric("[CLS] std",  f"{cls_vec.std():.3f}")
        c3.metric("[CLS] min",  f"{cls_vec.min():.2f}")
        c4.metric("[CLS] max",  f"{cls_vec.max():.2f}")

        h = hidden.numpy()
        if center:
            h = h - h.mean(axis=0, keepdims=True)
        h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        sim = h @ h.T

        off = sim[~np.eye(len(tokens), dtype=bool)]
        vmin, vmax = float(off.min()), float(off.max())
        cmap = LinearSegmentedColormap.from_list("gwu", ["#FFFFFF", "#93C5FD", BLUE, NAVY])

        n = len(tokens)
        fig, ax = plt.subplots(figsize=(max(6.5, n * 0.75), max(5.5, n * 0.65)))
        im = ax.imshow(sim, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(tokens, fontsize=12)
        if n <= 16:
            for i in range(n):
                for j in range(n):
                    v = sim[i, j]
                    frac = (v - vmin) / (vmax - vmin + 1e-8)
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if frac > 0.55 else "#1E293B")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Cosine similarity", color=MUTED)
        ax.set_title("How similar is each pair of output vectors?", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        tip("The color scale is stretched to the range in <b>this</b> sentence so differences "
            "stand out. Look for which words pair up, and where [CLS] and [SEP] sit compared "
            "with the ordinary words. Untick the checkbox to see how similar the raw vectors "
            "all are.")

    with st.expander("Show the embedding extraction code"):
        st.code("""
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased")

enc = tokenizer("The cat sat on the mat.", return_tensors="pt")
with torch.no_grad():
    out = model(**enc)

# out.last_hidden_state : (batch, seq_len, 768)  last-layer output for every token
# out.pooler_output     : (batch, 768)           [CLS] passed through linear + tanh

hidden          = out.last_hidden_state[0]   # (seq_len, 768)
cls_embedding   = hidden[0]                  # (768,)  the [CLS] token
word_embeddings = hidden[1:-1]               # without [CLS] and [SEP]
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════
elif num == 6:
    section_header(6, TOTAL, "Fine-tuning for Downstream Tasks",
                   "Same pretrained BERT, a different small head for each task")

    key_idea("Keep the pretrained BERT, add <b>one small layer</b> on top, and train "
             "<b>both together</b> on your labeled data for a few epochs.")

    lead("Because BERT already understands language, fine-tuning usually needs only "
         "<b>2 to 4 epochs</b> on a few thousand labeled examples.")

    sub("Four fine-tuning patterns")
    grid([
        card("A. Sentence classification",
             f"{tag('reads [CLS]', NAVY)}<br>One sentence in, one label out.<br>"
             "<b>Tasks:</b> sentiment, spam, topic.<br>"
             "<b>Head:</b> <code>nn.Linear(768, num_classes)</code>", BLUE),
        card("B. Token classification",
             f"{tag('reads every token', GREEN)}<br>One label per token.<br>"
             "<b>Tasks:</b> named entities, part-of-speech tags.<br>"
             "<b>Head:</b> <code>nn.Linear(768, num_labels)</code> on each token", GREEN),
        card("C. Question answering",
             f"{tag('reads every context token', GREEN)}<br>"
             "Input: [CLS] question [SEP] passage [SEP]. Two scores per token: "
             "answer <b>start</b> and answer <b>end</b>.<br>"
             "<b>Head:</b> <code>nn.Linear(768, 2)</code> on each token", ORANGE),
        card("D. Sentence-pair tasks",
             f"{tag('reads [CLS]', NAVY)}<br>"
             "Input: [CLS] A [SEP] B [SEP]. One label for the relationship.<br>"
             "<b>Tasks:</b> inference, paraphrase, similarity.<br>"
             "<b>Head:</b> <code>nn.Linear(768, num_classes)</code>", PURPLE),
    ])

    tip("In all four patterns the <b>whole model is trained</b>, not just the head. "
        "A small learning rate (2e-5 to 5e-5) keeps the pretrained weights from "
        "changing too much.")

    sub("Try it: sentiment analysis with a fine-tuned model")
    hint("This demo uses DistilBERT (a smaller BERT, see Section 8) fine-tuned on SST-2, "
         "the Stanford Sentiment Treebank. It is pattern A in action.")

    with st.container(border=True):
        sent_input = st.text_area("Text to analyze",
                                  value="The cat sat on the mat and looked very content.",
                                  height=90)
        run = st.button("Analyze sentiment", type="primary")

    if run:
        with st.spinner("Running sentiment analysis..."):
            result = load_sentiment()(sent_input)[0]

        label, score = result["label"], result["score"]
        color = GREEN if label == "POSITIVE" else RED
        html(f"""
        <div class="key" style="border-left-color:{color};">
          <div class="key-label" style="color:{'#6EE7B7' if label == 'POSITIVE' else '#FDA4AF'};">Prediction</div>
          <div class="key-text">{label.title()}
          <span style="font-size:1.05rem;color:#CBD5E1;">&nbsp;{score * 100:.1f}% confident</span></div>
        </div>""")

        pos = score if label == "POSITIVE" else 1 - score
        fig, ax = plt.subplots(figsize=(9, 1.4))
        ax.barh([0], [pos], color=GREEN, height=0.6)
        ax.barh([0], [1 - pos], left=[pos], color=RED, height=0.6)
        if pos > 0.08:
            ax.text(pos / 2, 0, f"Positive {pos * 100:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold")
        if 1 - pos > 0.08:
            ax.text(pos + (1 - pos) / 2, 0, f"Negative {(1 - pos) * 100:.0f}%",
                    ha="center", va="center", color="white", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.expander("Show the fine-tuning code"):
        st.code("""
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)      # positive / negative

texts  = ["I love this movie!", "This film was terrible."]
labels = torch.tensor([1, 0])

enc = tokenizer(texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt")

outputs = model(**enc, labels=labels)
loss    = outputs.loss      # cross-entropy on the [CLS] head
logits  = outputs.logits    # (batch, 2)

# Small learning rate: the pretrained weights should move only a little
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss.backward()
optimizer.step()
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: ENCODER TO DECODER
# ══════════════════════════════════════════════════════════════════════════════
elif num == 7:
    section_header(7, TOTAL, "Where BERT Fits Real Problems",
                   "Pointers for applied Praxis research: sectors, problems, "
                   "and what has been tried already")

    key_idea("Pick a problem where an organisation is <b>buried in text</b> and someone "
             "still has to <b>make a decision</b>. Then work out whether BERT belongs "
             "anywhere in the answer.")

    lead("A <b>menu, not a manual</b>. Every model and dataset named below is a "
         "<b>starting point to go and research</b>, not a recommendation. Open the panels "
         "that apply to you.")

    # ══ 1. How BERT shows up ═════════════════════════════════════════════════
    with st.expander("1.  How BERT shows up in a real system", expanded=True):
        lead("Sometimes BERT <b>is</b> the model. More often it is one component and "
             "something else makes the final call. Four shapes cover most applied work.")
        keyrows([
            (GREEN, "BERT is the model.",
             "Fine-tune it, add a classification or tagging head, ship it. Entirely "
             "legitimate, and the normal answer for high-volume routing and extraction."),
            (BLUE, "BERT reranks.",
             "Keyword or vector search returns 100 candidates, BERT reorders just those."
             '<span class="chain">search &#8594; 100 candidates &#8594; BERT reranks '
             "&#8594; the 10 a human reads</span>"),
            (PURPLE, "BERT retrieves.",
             "Encode every document once, then search by meaning instead of keywords. "
             "This is the retrieval half of a RAG system, where an LLM writes the answer."
             '<span class="chain">documents &#8594; embeddings + index &#8594; retrieve '
             "&#8594; LLM answers</span>"),
            (ORANGE, "BERT turns text into variables.",
             "Pull out labels and fields, join them to the tabular data you already have, "
             "feed the model you already had."
             '<span class="chain">notes &#8594; BERT extracts &#8594; join to labs, '
             "prices, demographics &#8594; existing model</span>"),
        ])
        tip("That last one is the most common shape in applied research, and it reframes "
            "your contribution usefully: the result is <b>&quot;adding text improved the "
            "existing model by this much, for these cases&quot;</b>, which is far stronger "
            "than an accuracy number on its own.")

    # ══ 2. Encoder or LLM ════════════════════════════════════════════════════
    with st.expander("2.  Encoder or LLM? You will be asked"):
        lead("They are good at <b>different things</b>, and most mature systems use both.")
        table(
            ["", f"{tag('Fine-tuned encoder', BLUE)}", f"{tag('Prompted LLM', PURPLE)}"],
            [
                ["Cost at volume", "Flat, on hardware you own.",
                 "Per call. Can dominate a budget."],
                ["Speed", "Milliseconds, easy to batch.", "Hundreds of ms to seconds."],
                ["Where data goes", "Stays in-house. Can run offline.",
                 "Usually leaves your network."],
                ["Labels needed", "Hundreds to thousands.", "Often none. Its real edge."],
                ["Reproducible?", "Fixed weights you can archive.",
                 "Providers update models, outputs drift."],
                ["Strongest at", "High-volume classification, extraction, retrieval.",
                 "Generation, summarising, the long tail."],
            ],
        )
        keyrows([
            (GREEN, "The design that often wins:",
             "have the <b>LLM label a few thousand examples</b>, check a sample by hand, "
             "then fine-tune a small encoder on the result. You get the LLM's coverage at "
             "the encoder's cost and speed."),
            (ORANGE, "A Praxis in itself:",
             "a careful head-to-head of the two on <b>your</b> task, reporting accuracy "
             "<b>and</b> cost <b>and</b> latency. Few papers report all three."),
        ])

    # ══ 3. Which encoder ═════════════════════════════════════════════════════
    with st.expander("3.  Which encoder for which job"):
        hint("Chosen by <b>role</b>, not by fame. Names to search for. Section 8 covers "
             "the family in more detail.")
        table(
            ["Its job", "Worth researching"],
            [
                ["Plain classification accuracy",
                 "<b>DeBERTa-v3</b>, <b>RoBERTa</b>. Often beat original BERT for free."],
                ["Documents past 512 tokens",
                 "<b>ModernBERT</b>, <b>Longformer</b>, <b>BigBird</b>. Essential for "
                 "legal, clinical and filings text."],
                ["Embeddings for search",
                 "<b>Sentence-BERT</b>, and the modern families (<b>E5</b>, <b>BGE</b>, "
                 "<b>GTE</b>). Plain BERT is a poor sentence encoder out of the box."],
                ["Reranking a shortlist",
                 "A <b>cross-encoder</b>: query and candidate scored together. Slower per "
                 "item, much more accurate."],
                ["Tight compute or on-premises",
                 "<b>DistilBERT</b>, <b>MobileBERT</b>, <b>ALBERT</b>."],
                ["Your specific domain",
                 "The models in panel 5. Usually the highest-value thing to try first."],
            ],
        )

    # ══ 4. Task shapes ═══════════════════════════════════════════════════════
    with st.expander("4.  The four task shapes"):
        hint("You met these as fine-tuning patterns in Section 6. They are also the "
             "quickest test of whether a messy real problem is a text problem at all.")
        keyrows([
            (BLUE, "Classification.",
             "One label per document. Is this complaint about a mortgage? "
             "<b>Most Praxis problems are this one.</b>"),
            (GREEN, "Token classification.",
             "One label per word. Which words are patient names, drug doses, "
             "contract parties?"),
            (PURPLE, "Sentence pair.",
             "A judgement about two texts together. Does this patient meet this "
             "criterion? Does this control satisfy this regulation?"),
            (ORANGE, "Similarity.",
             "Ranking by meaning. The duplicate ticket, the nearest prior case, even "
             "with no words in common."),
        ])

    # ══ 5. Sectors ═══════════════════════════════════════════════════════════
    with st.expander("5.  Pick your sector: problems, models, data"):
        t_cls  = tag("Classification", BLUE)
        t_tok  = tag("Token classification", GREEN)
        t_pair = tag("Sentence pair", PURPLE)
        t_sim  = tag("Similarity", ORANGE)

        def sector(rows, models, data, finding):
            table(["A problem you could work on", "Task shape"], rows)
            keyrows([
                (NAVY,   "Models to research:", models),
                (ORANGE, "Public data:",        data),
                (GREEN,  "What research suggests:", finding),
            ])

        tabs = st.tabs(["Health", "Finance", "Economics", "IT &amp; Security",
                        "Legal", "Education"])

        with tabs[0]:
            sector(
                [["Flag readmission risk from a discharge summary", t_cls],
                 ["Strip identifiers out of notes so they can be shared", t_tok],
                 ["Detect adverse drug reactions in patient forum posts", t_tok],
                 ["Assign billing or diagnosis codes to a note", t_cls],
                 ["Screen thousands of papers for a systematic review", t_cls],
                 ["Check a patient record against trial eligibility criteria", t_pair]],
                "<b>BioBERT</b>, <b>PubMedBERT</b>, <b>BlueBERT</b>, <b>SciBERT</b> "
                "(literature). <b>Bio+Clinical BERT</b>, <b>GatorTron</b> (hospital "
                "notes). <b>Clinical-Longformer</b> for long notes. Benchmarks: "
                "<b>BLURB</b>, <b>BLUE</b>.",
                "<b>MIMIC-III / MIMIC-IV</b> notes (free but credentialed, with required "
                "training, so start the paperwork early). <b>n2c2 / i2b2</b> corpora. "
                "<b>PubMed</b> abstracts.",
                "the sector where <b>domain pretraining pays off most</b>, since clinical "
                "vocabulary barely overlaps with Wikipedia. The practical gain usually "
                "shows up when text is added to an <b>existing structured risk model</b>. "
                "De-identification may be infrastructure you need before anything else is "
                "permitted.")

        with tabs[1]:
            sector(
                [["Score earnings-call tone against returns or volatility", t_cls],
                 ["Route consumer complaints by product and issue", t_cls],
                 ["Track how risk-factor sections shift year over year", t_cls],
                 ["Screen news for adverse media on a counterparty", t_cls],
                 ["Flag ESG claims the filing's own evidence contradicts", t_pair],
                 ["Extract parties, amounts and dates from loan documents", t_tok]],
                "<b>FinBERT</b> (more than one model carries this name, so check which you "
                "cite). <b>SEC-BERT</b> for filings. Benchmark: <b>FLUE</b>.",
                "<b>CFPB consumer complaints</b> (large, public, labelled by product, an "
                "unusually good Praxis dataset). <b>SEC EDGAR</b> filings in bulk. "
                "<b>Financial PhraseBank</b>.",
                "general sentiment models <b>misread financial language</b>, where "
                "&quot;liability&quot; and &quot;volatility&quot; are neutral technical terms. That is "
                "why FinBERT exists, and it makes an easy honest comparison. Treat the "
                "return-prediction literature carefully: results are sensitive to "
                "<b>time period and transaction costs</b>, so split by time, never "
                "at random.")

        with tabs[2]:
            sector(
                [["Measure policy uncertainty from news over time", t_cls],
                 ["Classify central bank statements as hawkish or dovish", t_cls],
                 ["Build a news sentiment index to nowcast activity", t_cls],
                 ["Code job postings to standard occupation categories", t_cls],
                 ["Track which skills employers demand, and how that shifts", t_tok],
                 ["Match postings to candidate profiles or training", t_sim]],
                "no dominant domain model. General <b>RoBERTa</b> or <b>DeBERTa-v3</b> "
                "fine-tuned on your own labelled sample is the normal route. "
                "<b>EconBERTa</b> is worth a look for economics entity extraction.",
                "<b>FOMC</b> statements, minutes and transcripts. <b>O*NET</b> occupation "
                "and skill taxonomies. Large job-posting collections. Legislative text.",
                "the influential uncertainty and sentiment indices were built by "
                "<b>counting keywords</b>. Rebuilding one with a contextual model and "
                "testing whether it tracks real outcomes better is a well-scoped "
                "contribution. Read the economics <b>text as data</b> methodology "
                "literature first, since it shapes how you defend your measurement.")

        with tabs[3]:
            sector(
                [["Route incoming incident tickets to the right team", t_cls],
                 ["Find the duplicate of a new bug report", t_sim],
                 ["Predict severity from a vulnerability description", t_cls],
                 ["Map a threat report to known attack techniques", t_cls],
                 ["Detect phishing and social engineering in email", t_cls],
                 ["Mine app-store reviews for defects and requests", t_cls]],
                "<b>CodeBERT</b>, <b>GraphCodeBERT</b>, <b>UniXcoder</b> (code). "
                "<b>SecureBERT</b>, <b>CySecBERT</b>, <b>SecBERT</b> (security text). "
                "<b>LogBERT</b> for logs. Benchmark: <b>CodeXGLUE</b>.",
                "<b>NVD / CVE</b> with published severity scores (public, labelled, "
                "large). <b>MITRE ATT&amp;CK</b>. GitHub issues. Stack Overflow.",
                "severity prediction is popular because NVD hands you labels at scale, but "
                "those labels are <b>assigned by people under process pressure</b>, so "
                "audit a sample before trusting them. Deduplication is the quieter, often "
                "more valuable target: a <b>similarity</b> problem, not classification, "
                "which is exactly why keyword search always handled it badly.")

        with tabs[4]:
            sector(
                [["Pull specific clause types out of a stack of contracts", t_tok],
                 ["Check whether an internal control satisfies a regulation", t_pair],
                 ["Rank documents by relevance for discovery or audit", t_sim],
                 ["Triage incoming public records requests", t_cls],
                 ["Classify solicitations and procurement notices", t_cls],
                 ["Find the nearest prior case to a new matter", t_sim]],
                "<b>Legal-BERT</b>, <b>CaseLaw-BERT</b>, and jurisdiction-specific "
                "variants. <b>Longformer</b> or <b>ModernBERT</b> for long documents. "
                "Benchmarks: <b>LexGLUE</b>, <b>LegalBench</b>, <b>CUAD</b>.",
                "<b>CUAD</b> (expert-annotated contract clauses). <b>LexGLUE</b>. "
                "<b>EUR-Lex</b> legislation. US court opinions via CourtListener.",
                "legal text breaks general models in a specific way: documents run "
                "<b>far past 512 tokens</b> and the decisive clause can sit anywhere. How "
                "you split a long document and recombine the pieces often matters more "
                "than which model you chose, and that choice is itself a publishable "
                "question. <b>Recall</b> usually matters more than precision, since a "
                "missed clause costs far more than one extra item to glance at.")

        with tabs[5]:
            sector(
                [["Score short answers or essays against a rubric", t_cls],
                 ["Summarise themes across thousands of course evaluations", t_cls],
                 ["Flag struggling students from forum posts", t_cls],
                 ["Match a student question to existing answers", t_sim],
                 ["Align curriculum text to standards or objectives", t_pair]],
                "no strong domain model. General <b>DeBERTa-v3</b> or <b>RoBERTa</b> "
                "fine-tuned on your institution's own graded sample.",
                "the <b>ASAP</b> essay scoring sets. Public MOOC forum corpora. Your own "
                "institution's anonymised data, often the most defensible choice.",
                "automated scoring can match the agreement between two human raters on "
                "some prompts, which shifts the real question from accuracy to "
                "<b>fairness and gaming</b>. Whether a scorer behaves equally across "
                "student groups, and whether length or vocabulary tricks fool it, is the "
                "harder and more valuable contribution.")

    # ══ 6. Traps ═════════════════════════════════════════════════════════════
    with st.expander("6.  Four traps specific to BERT"):
        hint("General methodology (splits, metrics, subgroups, drift) belongs to your "
             "research methods training and applies to any model. <b>These four come from "
             "BERT itself.</b>")
        keyrows([
            (BLUE, "TF-IDF is a real competitor, not a straw man.",
             "On keyword-driven classification it often matches a fine-tuned BERT in "
             "minutes of work, and on <b>long documents</b> it can win outright, because "
             "it reads the whole document while BERT sees only the first 512 tokens. "
             "BERT's edge appears when labels are few, texts are short, and meaning turns "
             "on word order or negation: <i>no evidence of fracture</i> and <i>evidence of "
             "fracture</i> are near-identical bags of words and opposite findings."),
            (ORANGE, "Two runs give two answers.",
             "Below a few thousand examples, BERT fine-tuning is strikingly sensitive to "
             "the random seed. Scores move several points between runs, and a minority of "
             "runs <b>collapse into predicting the majority class</b> and never recover. "
             "This is documented in the literature, not your mistake. Report a mean and "
             "spread across several seeds so you are not quoting your luckiest run."),
            (RED, "512 tokens silently redefines what you measured.",
             "Longer input is truncated without warning, so with long documents you "
             "evaluated their <b>opening few hundred words</b>, not your documents. Check "
             "the length distribution <b>before</b> reporting anything. A long tail is a "
             "finding that drives model choice, not a limitations footnote."),
            (GREEN, "A domain model needs general BERT beside it.",
             "<b>&quot;BioBERT reached 0.89&quot;</b> tells a reader nothing. <b>&quot;BioBERT 0.89 "
             "against BERT 0.81 on our data&quot;</b> answers a real question: does domain "
             "pretraining transfer to <b>this</b> problem? Run both. The code is identical "
             "apart from the model name, and the comparison turns a routine step into "
             "a contribution."),
        ])

    tip("A caution on every name above. These are <b>search terms, not citations</b>: "
        "availability, licences and access terms move quickly, and some names cover more "
        "than one model. Find the paper, confirm what it was trained on, and check you are "
        "permitted to use it. That verification is part of your literature review.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: BERT VARIANTS
# ══════════════════════════════════════════════════════════════════════════════
elif num == 8:
    section_header(8, TOTAL, "BERT Variants",
                   "How the community built on BERT after 2018")

    key_idea("Most variants keep BERT's architecture and change <b>how it is trained</b>: "
             "more data, different tasks, fewer parameters, or a specific domain.")

    sub("The family at a glance")
    table(
        ["Model", "Params", "Layers", "Hidden", "Heads", "What changed from BERT-Base"],
        [
            ["BERT-Base", "110M", "12", "768", "12",
             f"{tag('The original', NAVY)} Masked LM + NSP on Wikipedia and BooksCorpus "
             "(3.3B words)"],
            ["BERT-Large", "340M", "24", "1024", "16",
             f"{tag('Bigger', BLUE)} Twice as deep and wider; about 3x the parameters"],
            ["RoBERTa", "125M", "12", "768", "12",
             f"{tag('Better training', ORANGE)} No NSP, 10x more data, a new mask each "
             "epoch, much larger batches"],
            ["DistilBERT", "66M", "6", "768", "12",
             f"{tag('Smaller', GREEN)} Distilled from BERT-Base: 40% smaller, 60% faster, "
             "keeps about 97% of its accuracy"],
            ["ALBERT-base", "12M", "12", "768", "12",
             f"{tag('Shared weights', GREEN)} One set of layer weights reused 12 times; "
             "sentence order prediction replaces NSP"],
            ["BioBERT", "110M", "12", "768", "12",
             f"{tag('Biomedical', PURPLE)} BERT-Base further pretrained on PubMed and PMC "
             "articles"],
            ["SciBERT", "110M", "12", "768", "12",
             f"{tag('Scientific', PURPLE)} Trained from scratch on 1.14M papers with its own "
             "science vocabulary"],
        ],
        num_cols=(1, 2, 3, 4),
        highlight_rows=(0,),
    )

    sub("Which one should you use?")
    grid([
        card("General NLP tasks",
             "Start with <b>RoBERTa-base</b>. Same size as BERT-Base and usually better. "
             "Need something smaller? Use <b>DistilBERT</b>.", GREEN),
        card("Domain-specific text",
             "Biomedical: <b>BioBERT</b> or <b>PubMedBERT</b>. Legal: <b>LegalBERT</b>. "
             "Science: <b>SciBERT</b>. Domain pretraining often gives a clear boost "
             "on specialized tasks.", PURPLE),
        card("Production or limited hardware",
             "<b>DistilBERT</b> or <b>ALBERT-base</b> for speed and memory. Both run "
             "comfortably on a CPU.", BLUE),
        card("Maximum accuracy",
             "<b>RoBERTa-large</b> or <b>DeBERTa-large</b>. DeBERTa's disentangled attention "
             "beats RoBERTa on GLUE and SuperGLUE. Needs a large GPU to fine-tune.", ORANGE),
    ])

    with st.expander("Show how to load any variant from Hugging Face"):
        st.code("""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Change only this line to switch variants
MODEL = "roberta-base"    # or "distilbert-base-uncased", "allenai/scibert_scivocab_uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

# AutoTokenizer and AutoModel detect the architecture from the model card,
# so you never have to pick BertTokenizer vs RobertaTokenizer yourself.
enc = tokenizer("The cat sat on the mat.", return_tensors="pt",
                truncation=True, max_length=512)
print(model(**enc).logits.shape)   # (1, 2)
""", language="python")


# ── Previous / Next ───────────────────────────────────────────────────────────
st.divider()
p, _, n = st.columns([1, 2, 1])
with p:
    st.button("←  Previous", on_click=go, args=(-1,), disabled=num == 1,
              use_container_width=True, key="prev_btn")
with n:
    st.button("Next  →", on_click=go, args=(1,), disabled=num == TOTAL,
              type="primary", use_container_width=True, key="next_btn")

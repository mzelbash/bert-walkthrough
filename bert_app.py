"""
BERT: From Pretraining to Fine-tuning
SEAS 8525 - Computer Vision and Generative AI
Dr. Elbasheer
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

st.set_page_config(
    page_title="BERT Walkthrough",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #0f172a;
}

/* ── Sidebar ── */
.sidebar-brand   { padding:16px 0 20px; border-bottom:1px solid #e2e8f0; margin-bottom:14px; }
.sidebar-title   { font-size:1.15rem; font-weight:800; color:#1e1b4b; letter-spacing:-0.02em; }
.sidebar-course  { font-size:0.90rem; font-weight:700; color:#3730a3; margin-top:4px; }
.sidebar-course-name { font-size:0.83rem; font-weight:600; color:#4f46e5; }
.sidebar-instructor  { font-size:0.77rem; color:#6366f1; font-style:italic; }

.sidebar-setup { margin-top:16px; padding-top:14px; border-top:1px solid #e2e8f0; }
.sidebar-setup-label { font-size:0.70rem; font-weight:700; text-transform:uppercase;
                       letter-spacing:0.08em; color:#94a3b8; margin-bottom:8px; }
.setup-block  { margin-bottom:11px; }
.setup-tag    { display:inline-block; font-size:0.63rem; font-weight:700;
                text-transform:uppercase; color:#fff; padding:2px 9px; border-radius:3px; }
.setup-tag.conda { background:#4338ca; }
.setup-tag.pip   { background:#7c3aed; }
.setup-cmd  { display:block; font-family:'JetBrains Mono',monospace; font-size:0.72rem;
              color:#1e1b4b; background:#ffffff; border:1px solid #c7d2fe;
              border-radius:5px; padding:5px 10px; margin-top:4px; }

/* ── Section header ── */
.section-hdr { padding:22px 28px 18px; border-radius:12px;
               background:linear-gradient(135deg,#eef2ff 0%,#f8fafc 100%);
               border-left:5px solid #4f46e5; margin-bottom:24px; }
.section-num  { font-size:0.72rem; font-weight:700; text-transform:uppercase;
                letter-spacing:0.1em; color:#6366f1; margin-bottom:4px; }
.section-title { font-size:1.55rem; font-weight:800; color:#1e1b4b;
                 letter-spacing:-0.02em; line-height:1.2; }
.section-sub   { font-size:0.90rem; color:#475569; margin-top:5px; font-weight:400; }

/* ── Info / warn / success banners ── */
.info-box    { background:#eef2ff; border-left:4px solid #6366f1; border-radius:6px;
               padding:12px 16px; margin:12px 0; font-size:0.875rem; color:#1e1b4b; }
.warn-box    { background:#fffbeb; border-left:4px solid #f59e0b; border-radius:6px;
               padding:12px 16px; margin:12px 0; font-size:0.875rem; color:#1e1b4b; }
.success-box { background:#f0fdf4; border-left:4px solid #22c55e; border-radius:6px;
               padding:12px 16px; margin:12px 0; font-size:0.875rem; color:#1e1b4b; }
.class-tip   { background:#fdf4ff; border-left:4px solid #a855f7; border-radius:6px;
               padding:12px 16px; margin:12px 0; font-size:0.875rem; color:#1e1b4b; }
.class-tip strong { color:#7e22ce; }

/* ── Cards ── */
.enc-cards { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:14px 0 20px; }
.enc-card  { border-radius:9px; padding:14px 16px; }
.enc-card-title { font-size:0.88rem; font-weight:700; margin-bottom:7px; }
.enc-card-body  { font-size:0.82rem; color:#374151; line-height:1.68; }

/* ── Comparison table ── */
.comp-table { width:100%; border-collapse:collapse; font-family:'Inter',sans-serif;
              font-size:0.875rem; margin:0.5rem 0 1.2rem; }
.comp-table thead th { background:#f8fafc; color:#374151; font-weight:700;
                       font-size:0.70rem; letter-spacing:0.07em; text-transform:uppercase;
                       padding:11px 16px; text-align:left; border-bottom:2px solid #e2e8f0; }
.comp-table tbody td { padding:10px 16px; color:#1e1b4b; border-bottom:1px solid #f1f5f9;
                       vertical-align:top; line-height:1.55; }
.comp-table tbody tr:last-child td { border-bottom:none; }
.comp-table tbody tr:hover td { background:#fafbff; }

/* ── Token pills ── */
.token-row  { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
.token-pill { border-radius:6px; padding:5px 10px; font-family:'JetBrains Mono',monospace;
              font-size:0.80rem; font-weight:600; border:1px solid transparent; }
.token-cls  { background:#eef2ff; color:#3730a3; border-color:#c7d2fe; }
.token-sep  { background:#fef3c7; color:#92400e; border-color:#fde68a; }
.token-mask { background:#fee2e2; color:#991b1b; border-color:#fca5a5; }
.token-seg-a { background:#dbeafe; color:#1d4ed8; border-color:#93c5fd; }
.token-seg-b { background:#d1fae5; color:#065f46; border-color:#6ee7b7; }
.token-word  { background:#f8fafc; color:#1e293b; border-color:#e2e8f0; }
.token-wp    { background:#fdf4ff; color:#7e22ce; border-color:#d8b4fe; }

/* ── Math box ── */
.math-box  { background:#f8fafc; border:1px solid #e2e8f0; border-left:4px solid #4f46e5;
             border-radius:6px; padding:14px 18px; margin:10px 0; }
.math-step { display:flex; gap:10px; margin-bottom:8px; align-items:flex-start; }
.math-step-num { background:#4f46e5; color:#fff; border-radius:50%; width:22px; height:22px;
                 font-size:0.70rem; display:flex; align-items:center; justify-content:center;
                 flex-shrink:0; }

/* ── Pipeline ── */
.pipeline   { display:flex; align-items:flex-start; gap:4px; flex-wrap:wrap; }
.pipe-step  { display:flex; flex-direction:column; align-items:center; gap:5px; }
.pipe-box   { background:#f8fafc; border:2px solid #c7d2fe; border-radius:8px;
              padding:9px 12px; text-align:center; font-size:0.82rem; font-weight:700;
              color:#1e1b4b; min-width:96px; }
.pipe-box small { font-size:0.69rem; font-weight:400; color:#6366f1; }
.pipe-arrow { font-size:1.4rem; color:#4f46e5; font-weight:700; margin-top:22px; }

/* ── Q&A ── */
.qa-item { margin:0 0 14px; border:1px solid #e2e8f0; border-radius:8px; overflow:hidden; }
.qa-q { background:#f8fafc; padding:11px 16px; font-weight:600; font-size:0.875rem; }
.qa-a { background:#ffffff; padding:11px 16px; line-height:1.75; font-size:0.875rem; }
.qa-num { background:#4f46e5; color:#fff; border-radius:4px; padding:1px 7px;
          font-size:0.72rem; margin-right:6px; }

/* ── Attention heatmap ── */
.attn-grid { display:grid; gap:3px; margin:10px 0; }
.attn-cell { border-radius:3px; height:28px; display:flex; align-items:center;
             justify-content:center; font-size:0.72rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def section_header(num, title, subtitle):
    st.markdown(f"""
    <div class="section-hdr">
      <div class="section-num">Section {num}</div>
      <div class="section-title">{title}</div>
      <div class="section-sub">{subtitle}</div>
    </div>""", unsafe_allow_html=True)

def info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def success(text):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def tip(text):
    st.markdown(f'<div class="class-tip"><strong>Class Tip:</strong> {text}</div>',
                unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BERT tokenizer...")
def load_tokenizer():
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained("bert-base-uncased")

@st.cache_resource(show_spinner="Loading BERT model (this takes ~30 s the first time)...")
def load_bert():
    from transformers import BertModel
    return BertModel.from_pretrained("bert-base-uncased")

@st.cache_resource(show_spinner="Loading fill-mask pipeline...")
def load_fill_mask():
    from transformers import pipeline
    return pipeline("fill-mask", model="bert-base-uncased")

@st.cache_resource(show_spinner="Loading sentiment pipeline...")
def load_sentiment():
    from transformers import pipeline
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-title">BERT Walkthrough</div>
      <div class="sidebar-course">SEAS 8525</div>
      <div class="sidebar-course-name">Computer Vision &amp; Generative AI</div>
      <div class="sidebar-instructor">Dr. Elbasheer</div>
    </div>""", unsafe_allow_html=True)

    sections = [
        "1. The Bridge: Transformer to BERT",
        "2. BERT's Input: Three Embeddings",
        "3. Pretraining Task 1: Masked LM",
        "4. Pretraining Task 2: Next Sentence",
        "5. The [CLS] Token",
        "6. Fine-tuning for Downstream Tasks",
        "7. Encoder Output to Decoder",
        "8. BERT Variants",
    ]
    section = st.radio("Sections", sections, label_visibility="collapsed")

    st.markdown("""
    <div class="sidebar-setup">
      <div class="sidebar-setup-label">Quick setup</div>
      <div class="setup-block">
        <span class="setup-tag pip">pip</span>
        <div class="setup-cmd">pip install -r requirements.txt</div>
        <div class="setup-cmd">streamlit run bert_app.py</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: THE BRIDGE
# ══════════════════════════════════════════════════════════════════════════════
if section.startswith("1"):
    section_header("1", "The Bridge: Transformer to BERT",
                   "You already built the encoder. BERT is 12 of them, trained in a new way.")

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:16px;'>
    In class you traced "the cat sat on the mat" through a Transformer encoder and watched
    each word token produce a rich contextual embedding of size 512. Every word attended to
    every other word; the output for "sat" knew about "cat" and "mat" at the same time.
    That bidirectional understanding <em>is</em> BERT's foundation.
    </p>
    """, unsafe_allow_html=True)

    # Visual bridge
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0;margin:18px 0 24px;overflow-x:auto;">
      <div style="background:#f0f9ff;border:2px solid #0284c7;border-radius:10px;
                  padding:16px 20px;min-width:200px;">
        <div style="font-size:0.72rem;font-weight:700;color:#0284c7;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:8px;">What you already know</div>
        <div style="font-weight:700;color:#0c4a6e;font-size:0.95rem;">Transformer Encoder</div>
        <div style="font-size:0.78rem;color:#374151;margin-top:6px;line-height:1.6;">
          Self-attention<br>Multi-head attention<br>Feed-forward (FFN)<br>
          LayerNorm + residuals<br>Output: 1 rich vector per word
        </div>
      </div>
      <div style="font-size:2rem;color:#4f46e5;font-weight:700;padding:0 16px;">&#8594;</div>
      <div style="background:#eef2ff;border:2px solid #4f46e5;border-radius:10px;
                  padding:16px 20px;min-width:200px;">
        <div style="font-size:0.72rem;font-weight:700;color:#4f46e5;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:8px;">BERT adds</div>
        <div style="font-weight:700;color:#1e1b4b;font-size:0.95rem;">12 (or 24) Encoders</div>
        <div style="font-size:0.78rem;color:#374151;margin-top:6px;line-height:1.6;">
          Stacked encoder blocks<br>768-dim hidden size<br>Learned positional encoding<br>
          Special tokens [CLS] [SEP] [MASK]<br>Self-supervised pretraining
        </div>
      </div>
      <div style="font-size:2rem;color:#4f46e5;font-weight:700;padding:0 16px;">=</div>
      <div style="background:#f0fdf4;border:2px solid #16a34a;border-radius:10px;
                  padding:16px 20px;min-width:200px;">
        <div style="font-size:0.72rem;font-weight:700;color:#16a34a;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:8px;">Result</div>
        <div style="font-weight:700;color:#14532d;font-size:0.95rem;">Universal Language Model</div>
        <div style="font-size:0.78rem;color:#374151;margin-top:6px;line-height:1.6;">
          110M pretrained parameters<br>Fine-tune on any NLP task<br>
          State-of-the-art (2018)<br>Understands context in both directions
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What changes, what stays the same")
    st.markdown("""
    <table class="comp-table">
      <thead><tr><th>Aspect</th><th>Original Transformer (encoder)</th><th>BERT</th></tr></thead>
      <tbody>
        <tr>
          <td><strong>Architecture</strong></td>
          <td>One encoder stack</td>
          <td>Same encoder block, stacked 12 (Base) or 24 (Large) times</td>
        </tr>
        <tr>
          <td><strong>Attention direction</strong></td>
          <td>Bidirectional (encoder) or causal (decoder)</td>
          <td>Always fully bidirectional: every token attends to every other token</td>
        </tr>
        <tr>
          <td><strong>Hidden size</strong></td>
          <td>512 in the paper</td>
          <td>768 (Base) or 1024 (Large)</td>
        </tr>
        <tr>
          <td><strong>Positional encoding</strong></td>
          <td>Fixed sinusoidal</td>
          <td>Learned positional embeddings (like ViT)</td>
        </tr>
        <tr>
          <td><strong>Input tokens</strong></td>
          <td>Word / subword tokens</td>
          <td>WordPiece subword tokens + special [CLS], [SEP], [MASK]</td>
        </tr>
        <tr>
          <td><strong>Training objective</strong></td>
          <td>Next-token prediction (autoregressive) or supervised translation</td>
          <td>Self-supervised: Masked LM + Next Sentence Prediction</td>
        </tr>
        <tr>
          <td><strong>Output used for</strong></td>
          <td>Translation target tokens</td>
          <td>[CLS] for classification, all tokens for span/token tasks</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    tip("""BERT was introduced in the 2018 paper <em>"BERT: Pre-training of Deep Bidirectional
    Transformers for Language Understanding"</em> by Devlin et al. at Google.
    The key word is <strong>bidirectional</strong>: previous models like GPT read left-to-right only.
    BERT reads the entire sentence at once, which gives it much richer context for each word.""")

    with st.expander("Frequently asked questions", expanded=False):
        st.markdown("""
        <div class="qa-item">
          <div class="qa-q"><span class="qa-num">Q1</span>
            Is BERT an encoder or a decoder?
          </div>
          <div class="qa-a">Encoder only. It produces rich embeddings for each input token
          but does not generate new text on its own. To generate text you need to add a decoder
          (covered in Section 7) or a language model head on top.</div>
        </div>
        <div class="qa-item">
          <div class="qa-q"><span class="qa-num">Q2</span>
            How is BERT different from GPT?
          </div>
          <div class="qa-a">GPT is a decoder-only model trained to predict the next token
          (left-to-right). BERT is an encoder-only model trained with masked prediction
          (bidirectional). GPT generates text; BERT understands it.</div>
        </div>
        <div class="qa-item">
          <div class="qa-q"><span class="qa-num">Q3</span>
            Why does BERT need pretraining? Can we just train it on our task?
          </div>
          <div class="qa-a">BERT-Base has 110 million parameters. Training that from scratch
          requires billions of words and weeks of compute. Pretraining on a large corpus
          (Wikipedia + BooksCorpus) gives the model general language understanding.
          You then fine-tune the pretrained model on your specific task in hours.</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BERT'S INPUT
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("2"):
    section_header("2", "BERT's Input: Three Embeddings",
                   "Token + Segment + Position, all summed before the first encoder layer")

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:16px;'>
    Before a single encoder layer runs, BERT constructs the input representation by summing
    three separate learned embeddings for each token. This is different from the Transformer
    you studied, which only had token + positional. The third embedding tells BERT
    which sentence each token belongs to.
    </p>
    """, unsafe_allow_html=True)

    # Three embedding cards
    st.markdown("""
    <div class="enc-cards" style="grid-template-columns:1fr 1fr 1fr;">
      <div class="enc-card" style="background:#dbeafe;border-left:4px solid #2563eb;">
        <div class="enc-card-title" style="color:#1d4ed8;">1. Token Embedding</div>
        <div class="enc-card-body">
          Lookup table mapping each WordPiece token to a 768-dim vector.
          Vocabulary size: 30,522 tokens.<br><br>
          <strong>Example:</strong> "cat" maps to a 768-dim vector learned during pretraining.
          Rare words like "unaffable" get split: ["un", "##aff", "##able"].
        </div>
      </div>
      <div class="enc-card" style="background:#d1fae5;border-left:4px solid #059669;">
        <div class="enc-card-title" style="color:#065f46;">2. Segment Embedding</div>
        <div class="enc-card-body">
          Tells BERT which sentence a token belongs to.
          All tokens in sentence A get embedding E<sub>A</sub>;
          all tokens in sentence B get E<sub>B</sub>.<br><br>
          <strong>Why:</strong> BERT is often given two sentences (e.g., question + context).
          Segment embeddings let it distinguish them.
        </div>
      </div>
      <div class="enc-card" style="background:#ede9fe;border-left:4px solid #7c3aed;">
        <div class="enc-card-title" style="color:#5b21b6;">3. Position Embedding</div>
        <div class="enc-card-body">
          Learned vector for each position 0 to 511.
          Unlike the original Transformer (sinusoidal), BERT's positional embeddings
          are trained from scratch alongside the model.<br><br>
          <strong>Max sequence length:</strong> 512 tokens.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="math-box">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:#1e1b4b;
                  text-align:center;padding:8px 0;">
        Input<sub>i</sub> = TokenEmb(token<sub>i</sub>) + SegmentEmb(seg<sub>i</sub>)
        + PositionEmb(i)
      </div>
      <div style="font-size:0.80rem;color:#475569;text-align:center;margin-top:4px;">
        All three are 768-dim vectors. They are <strong>added element-wise</strong> (not concatenated).
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Special tokens
    st.markdown("### Special tokens")
    st.markdown("""
    <div class="enc-cards">
      <div class="enc-card" style="background:#eef2ff;border-left:4px solid #4f46e5;">
        <div class="enc-card-title" style="color:#3730a3;">[CLS] Classification token</div>
        <div class="enc-card-body">
          Always the <strong>first token</strong> of every input. After passing through all
          12 encoder layers, its output vector summarizes the entire input sequence.
          Used as input to the classification head for sentence-level tasks.
          Think of it as the designated "note-taker" for the whole sentence.
        </div>
      </div>
      <div class="enc-card" style="background:#fef3c7;border-left:4px solid #d97706;">
        <div class="enc-card-title" style="color:#92400e;">[SEP] Separator token</div>
        <div class="enc-card-body">
          Marks the <strong>boundary between sentences</strong>. For a single sentence,
          it goes at the end. For two sentences (e.g., QA), it goes after each.
          Helps BERT's attention mechanism recognize where one sentence ends
          and the next begins.
        </div>
      </div>
      <div class="enc-card" style="background:#fee2e2;border-left:4px solid #dc2626;">
        <div class="enc-card-title" style="color:#991b1b;">[MASK] Mask token</div>
        <div class="enc-card-body">
          Used <strong>only during pretraining</strong>. Replaces a token that the model
          must predict. The model never sees [MASK] during fine-tuning, which creates
          a small train/test mismatch. RoBERTa later addressed this with dynamic masking.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Interactive tokenization demo
    st.markdown("### Interactive tokenization demo")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:12px;'>Type any sentence (or use the class example) and see "
        "exactly how BERT tokenizes it, adds special tokens, and assigns segment IDs.</p>",
        unsafe_allow_html=True
    )

    col_input, col_input2 = st.columns([3, 1])
    with col_input:
        sent1 = st.text_input("Sentence A", value="The cat sat on the mat.")
    with col_input2:
        sent2 = st.text_input("Sentence B (optional)", value="")

    if st.button("Tokenize", type="primary"):
        tokenizer = load_tokenizer()

        if sent2.strip():
            enc = tokenizer(sent1, sent2, return_tensors="pt")
        else:
            enc = tokenizer(sent1, return_tensors="pt")

        tokens   = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        seg_ids  = enc["token_type_ids"][0].tolist()
        pos_ids  = list(range(len(tokens)))

        # Token pills
        st.markdown("**Tokens after WordPiece + special tokens:**")
        pills = ""
        for tok, seg in zip(tokens, seg_ids):
            if tok == "[CLS]":
                cls = "token-cls"
            elif tok == "[SEP]":
                cls = "token-sep"
            elif tok == "[MASK]":
                cls = "token-mask"
            elif tok.startswith("##"):
                cls = "token-wp"
            elif seg == 0:
                cls = "token-seg-a"
            else:
                cls = "token-seg-b"
            pills += f'<span class="token-pill {cls}">{tok}</span>'
        st.markdown(f'<div class="token-row">{pills}</div>', unsafe_allow_html=True)

        # Legend
        st.markdown("""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 16px;font-size:0.75rem;">
          <span class="token-pill token-cls">[CLS]</span>
          <span class="token-pill token-sep">[SEP]</span>
          <span class="token-pill token-seg-a">Sentence A</span>
          <span class="token-pill token-seg-b">Sentence B</span>
          <span class="token-pill token-wp">##subword</span>
        </div>
        """, unsafe_allow_html=True)

        # Table of IDs
        df = pd.DataFrame({
            "Position": pos_ids,
            "Token":    tokens,
            "Token ID": enc["input_ids"][0].tolist(),
            "Segment":  ["A" if s == 0 else "B" for s in seg_ids],
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.markdown(f"""
        <p style='font-family:Inter,sans-serif;font-size:0.82rem;color:#475569;margin-top:8px;'>
        Sequence length: <strong>{len(tokens)}</strong> tokens
        (including [CLS] and [SEP]).
        Each token will become a 768-dim vector after the three embeddings are summed.
        </p>
        """, unsafe_allow_html=True)
    else:
        info("Click Tokenize to see how BERT processes the input before any encoding.")

    with st.expander("Show the tokenization code", expanded=False):
        st.code("""
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Single sentence
enc = tokenizer("The cat sat on the mat.", return_tensors="pt")
# enc["input_ids"]        -- token IDs, shape (1, seq_len)
# enc["token_type_ids"]   -- segment IDs (0 = sentence A)
# enc["attention_mask"]   -- 1 for real tokens, 0 for padding

tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
# ['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', '.', '[SEP]']

# Two sentences (e.g., for QA or NSP)
enc2 = tokenizer("The cat sat on the mat.", "Where did the cat sit?",
                 return_tensors="pt")
# token_type_ids: 0 for sentence A tokens, 1 for sentence B tokens
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MASKED LANGUAGE MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("3"):
    section_header("3", "Pretraining Task 1: Masked Language Model",
                   "How BERT learns bidirectional language understanding from unlabeled text")

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:16px;'>
    BERT is trained on Wikipedia and BooksCorpus without any human labels.
    The trick: randomly hide some words and ask the model to predict them.
    Because BERT sees the full sentence (both left and right context),
    it must use bidirectional understanding to fill in the blank.
    </p>
    """, unsafe_allow_html=True)

    # MLM mechanics
    st.markdown("### The 15% masking rule")
    st.markdown("""
    <div class="math-box">
      <div class="math-step">
        <div class="math-step-num">1</div>
        <div>
          <strong>Select 15% of tokens</strong> in each sequence as candidates for masking.
          For a 9-token sentence like "the cat sat on the mat", that is about 1-2 tokens.
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">2</div>
        <div>
          Of those selected tokens, apply one of three treatments:
          <strong>80%</strong> replaced with [MASK],
          <strong>10%</strong> replaced with a random word,
          <strong>10%</strong> left unchanged.
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">3</div>
        <div>
          The model predicts the original token at each masked position.
          Loss is computed <em>only</em> at masked positions, not across the whole sequence.
        </div>
      </div>
      <div class="math-step" style="margin-bottom:0;">
        <div class="math-step-num">4</div>
        <div>
          The 10% random / 10% unchanged trick prevents the model from learning to
          "always assume [MASK] means something is hidden." It must represent every
          token as if it might need to be predicted.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tip("""Compare this to the Transformer you learned: that model predicted the <em>next</em> word
    (left-to-right only). BERT predicts a <em>hidden</em> word using <em>both sides</em>.
    This is why "sat" in "the cat [MASK] on the mat" is easy for BERT: it has
    "cat" on the left AND "on the mat" on the right to work with.""")

    st.divider()

    # Live MLM demo
    st.markdown("### Live demo: mask a word and let BERT predict it")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:12px;'>Replace any word in the sentence with [MASK] "
        "and BERT will predict the top candidates. Use the class example "
        "or type your own.</p>",
        unsafe_allow_html=True
    )

    mlm_sentence = st.text_input(
        "Sentence with [MASK]",
        value="The cat [MASK] on the mat.",
        help="Put [MASK] where you want BERT to predict. Only one [MASK] at a time."
    )

    if st.button("Predict masked word", type="primary"):
        if "[MASK]" not in mlm_sentence:
            st.error("Please include exactly one [MASK] token in the sentence.")
        else:
            with st.spinner("Running BERT fill-mask..."):
                fill = load_fill_mask()
                results = fill(mlm_sentence, top_k=8)

            st.markdown("**BERT's top predictions:**")

            # Bar chart
            words  = [r["token_str"].strip() for r in results]
            scores = [r["score"] for r in results]
            colors = ["#4f46e5" if i == 0 else "#c7d2fe" for i in range(len(words))]

            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            bars = ax.barh(words[::-1], scores[::-1], color=colors[::-1],
                           edgecolor="#f1f5f9", linewidth=0.5)
            for bar, sc in zip(bars, scores[::-1]):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{sc*100:.1f}%", va="center", fontsize=9, color="#374151")
            ax.set_xlabel("Confidence", color="#475569", fontsize=9)
            ax.set_xlim(0, max(scores) * 1.25)
            ax.tick_params(colors="#64748b", labelsize=9)
            ax.spines[:].set_color("#e2e8f0")
            ax.set_title(f'Predictions for: "{mlm_sentence}"',
                         color="#0f172a", fontsize=10, pad=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Show top result in context
            top_word = results[0]["token_str"].strip()
            filled   = mlm_sentence.replace("[MASK]", f"**{top_word}**")
            success(f"Top prediction: \"{filled}\" ({results[0]['score']*100:.1f}% confidence)")

    else:
        info("Click the button to run BERT's fill-mask on the sentence above.")

    with st.expander("Show the MLM prediction code", expanded=False):
        st.code("""
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-uncased")

results = fill_mask("The cat [MASK] on the mat.", top_k=5)
for r in results:
    print(f"{r['token_str']:12s}  {r['score']*100:.1f}%")
# sat          98.2%
# sat          ...
# sleeping     ...

# BERT uses both left context ("the cat")
# AND right context ("on the mat") to predict the masked word.
# A left-to-right model like GPT only has "the cat" to work with.
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: NEXT SENTENCE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("4"):
    section_header("4", "Pretraining Task 2: Next Sentence Prediction",
                   "How BERT learns relationships between sentences using [CLS]")

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:16px;'>
    Many real NLP tasks involve two sentences: question answering ("Is this context relevant
    to this question?"), natural language inference ("Does this hypothesis follow from this
    premise?"), and semantic similarity. MLM alone does not teach sentence-level relationships.
    NSP is the second pretraining task that fills this gap.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("### How NSP works")
    st.markdown("""
    <div class="math-box">
      <div class="math-step">
        <div class="math-step-num">1</div>
        <div>
          BERT receives two sentences separated by [SEP]:
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;">
          [CLS] Sentence A [SEP] Sentence B [SEP]
          </span>
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">2</div>
        <div>
          50% of training pairs are <strong>IsNext</strong>: sentence B actually follows
          sentence A in the original text.<br>
          50% are <strong>NotNext</strong>: sentence B is a random sentence from a
          different document.
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">3</div>
        <div>
          After all 12 encoder layers, the <strong>[CLS] token output</strong>
          is passed to a binary classifier: IsNext or NotNext.
          This is the first time [CLS] is used as a sentence-level summary.
        </div>
      </div>
      <div class="math-step" style="margin-bottom:0;">
        <div class="math-step-num">4</div>
        <div>
          Both MLM loss and NSP loss are added together and optimized jointly
          during pretraining.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Visual example
    st.markdown("### Concrete example")
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("""
        <div class="enc-card" style="background:#f0fdf4;border-left:4px solid #16a34a;">
          <div class="enc-card-title" style="color:#14532d;">IsNext (positive pair)</div>
          <div class="enc-card-body">
            <div class="token-row" style="flex-wrap:wrap;">
              <span class="token-pill token-cls">[CLS]</span>
              <span class="token-pill token-seg-a">the</span>
              <span class="token-pill token-seg-a">cat</span>
              <span class="token-pill token-seg-a">sat</span>
              <span class="token-pill token-seg-a">on</span>
              <span class="token-pill token-seg-a">the</span>
              <span class="token-pill token-seg-a">mat</span>
              <span class="token-pill token-sep">[SEP]</span>
              <span class="token-pill token-seg-b">it</span>
              <span class="token-pill token-seg-b">fell</span>
              <span class="token-pill token-seg-b">asleep</span>
              <span class="token-pill token-sep">[SEP]</span>
            </div>
            <div style="margin-top:12px;font-size:0.82rem;">
              [CLS] output &#8594; classifier &#8594;
              <strong style="color:#14532d;">IsNext (1)</strong>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_neg:
        st.markdown("""
        <div class="enc-card" style="background:#fef2f2;border-left:4px solid #dc2626;">
          <div class="enc-card-title" style="color:#991b1b;">NotNext (negative pair)</div>
          <div class="enc-card-body">
            <div class="token-row" style="flex-wrap:wrap;">
              <span class="token-pill token-cls">[CLS]</span>
              <span class="token-pill token-seg-a">the</span>
              <span class="token-pill token-seg-a">cat</span>
              <span class="token-pill token-seg-a">sat</span>
              <span class="token-pill token-seg-a">on</span>
              <span class="token-pill token-seg-a">the</span>
              <span class="token-pill token-seg-a">mat</span>
              <span class="token-pill token-sep">[SEP]</span>
              <span class="token-pill token-seg-b">stocks</span>
              <span class="token-pill token-seg-b">fell</span>
              <span class="token-pill token-seg-b">today</span>
              <span class="token-pill token-sep">[SEP]</span>
            </div>
            <div style="margin-top:12px;font-size:0.82rem;">
              [CLS] output &#8594; classifier &#8594;
              <strong style="color:#991b1b;">NotNext (0)</strong>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Interactive demo
    st.markdown("### Try it: build an NSP input")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:12px;'>Enter two sentences and see exactly how BERT formats "
        "the NSP input with token type IDs.</p>",
        unsafe_allow_html=True
    )

    nsp_a = st.text_input("Sentence A", value="The cat sat on the mat.")
    nsp_b = st.text_input("Sentence B", value="It fell asleep in the sun.")

    if st.button("Build NSP input", type="primary"):
        tokenizer = load_tokenizer()
        enc = tokenizer(nsp_a, nsp_b, return_tensors="pt")
        tokens  = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        seg_ids = enc["token_type_ids"][0].tolist()

        pills = ""
        for tok, seg in zip(tokens, seg_ids):
            if tok == "[CLS]":
                cls = "token-cls"
            elif tok == "[SEP]":
                cls = "token-sep"
            elif tok.startswith("##"):
                cls = "token-wp"
            elif seg == 0:
                cls = "token-seg-a"
            else:
                cls = "token-seg-b"
            pills += f'<span class="token-pill {cls}">{tok}</span>'

        st.markdown("**NSP input sequence:**")
        st.markdown(f'<div class="token-row">{pills}</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0;font-size:0.75rem;">
          <span class="token-pill token-cls">[CLS] = sentence summary</span>
          <span class="token-pill token-seg-a">Segment A (type ID = 0)</span>
          <span class="token-pill token-seg-b">Segment B (type ID = 1)</span>
          <span class="token-pill token-sep">[SEP] = boundary</span>
        </div>
        """, unsafe_allow_html=True)

        seg_a_count = seg_ids.count(0)
        seg_b_count = seg_ids.count(1)
        st.markdown(f"""
        <p style='font-family:Inter,sans-serif;font-size:0.82rem;color:#475569;margin-top:8px;'>
        Sentence A: <strong>{seg_a_count} tokens</strong> (including [CLS] and first [SEP])
        &nbsp;|&nbsp;
        Sentence B: <strong>{seg_b_count} tokens</strong> (including final [SEP])
        &nbsp;|&nbsp;
        Total: <strong>{len(tokens)} tokens</strong>
        </p>
        """, unsafe_allow_html=True)

        info("""After passing this full sequence through all 12 BERT encoder layers,
        the [CLS] token at position 0 will hold a vector that encodes the relationship
        between both sentences. That vector is fed to a linear layer with 2 outputs
        (IsNext / NotNext) during pretraining.""")
    else:
        info("Click the button to see how BERT formats a two-sentence NSP input.")

    with st.expander("Show the NSP setup code", expanded=False):
        st.code("""
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

sent_a = "The cat sat on the mat."
sent_b = "It fell asleep in the sun."        # IsNext
sent_c = "Stocks fell sharply today."        # NotNext

def nsp_score(a, b):
    enc    = tokenizer(a, b, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits         # shape (1, 2)
    probs  = logits.softmax(-1)[0]
    return {"IsNext": probs[0].item(), "NotNext": probs[1].item()}

print(nsp_score(sent_a, sent_b))   # IsNext score high
print(nsp_score(sent_a, sent_c))   # NotNext score high
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: THE [CLS] TOKEN
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("5"):
    section_header("5", "The [CLS] Token",
                   "How a single vector summarizes an entire sentence after 12 encoder layers")

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:16px;'>
    After pretraining, BERT produces a 768-dim output vector for every input token.
    The choice of which output to use depends entirely on the downstream task.
    The [CLS] token is special: it attends to every other token in all 12 layers
    and ends up encoding the overall meaning of the full input.
    </p>
    """, unsafe_allow_html=True)

    # Which output to use diagram
    st.markdown("### Which output goes where")
    st.markdown("""
    <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-family:Inter,sans-serif;font-size:0.82rem;">
      <thead>
        <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
          <th style="padding:10px 14px;color:#374151;font-weight:700;">Task type</th>
          <th style="padding:10px 14px;color:#374151;font-weight:700;">Output used</th>
          <th style="padding:10px 14px;color:#374151;font-weight:700;">Shape</th>
          <th style="padding:10px 14px;color:#374151;font-weight:700;">Example</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid #f1f5f9;background:#eef2ff;">
          <td style="padding:10px 14px;font-weight:600;color:#3730a3;">Sentence classification</td>
          <td style="padding:10px 14px;font-family:monospace;">[CLS] token (pos 0)</td>
          <td style="padding:10px 14px;font-family:monospace;">(B, 768)</td>
          <td style="padding:10px 14px;">Sentiment analysis, spam detection</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 14px;font-weight:600;color:#065f46;">Token classification</td>
          <td style="padding:10px 14px;font-family:monospace;">Every token output</td>
          <td style="padding:10px 14px;font-family:monospace;">(B, seq, 768)</td>
          <td style="padding:10px 14px;">Named entity recognition, POS tagging</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;background:#fffbeb;">
          <td style="padding:10px 14px;font-weight:600;color:#92400e;">Question answering</td>
          <td style="padding:10px 14px;font-family:monospace;">All context tokens</td>
          <td style="padding:10px 14px;font-family:monospace;">(B, seq, 768)</td>
          <td style="padding:10px 14px;">Predict start/end span of the answer</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 14px;font-weight:600;color:#7e22ce;">Sentence similarity</td>
          <td style="padding:10px 14px;font-family:monospace;">[CLS] of each sentence</td>
          <td style="padding:10px 14px;font-family:monospace;">(B, 768)</td>
          <td style="padding:10px 14px;">Semantic textual similarity, NLI</td>
        </tr>
      </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Live embedding demo
    st.markdown("### Run BERT and inspect the output embeddings")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:12px;'>Runs bert-base-uncased on our sentence and shows the "
        "output shape, the [CLS] embedding, and a heatmap comparing embeddings "
        "across tokens.</p>",
        unsafe_allow_html=True
    )

    embed_sent = st.text_input("Sentence", value="The cat sat on the mat.",
                               key="embed_sent")

    if st.button("Run BERT and show embeddings", type="primary"):
        with st.spinner("Loading BERT and running forward pass..."):
            tokenizer = load_tokenizer()
            model_bert = load_bert()
            enc = tokenizer(embed_sent, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            with torch.no_grad():
                out = model_bert(**enc)
            hidden = out.last_hidden_state[0]  # (seq_len, 768)

        st.success(f"Output shape: {tuple(hidden.shape)}  "
                   f"({len(tokens)} tokens x 768 dims)")

        # CLS vector stats
        cls_vec = hidden[0].numpy()
        st.markdown("**[CLS] token statistics:**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Mean", f"{cls_vec.mean():.4f}")
        sc2.metric("Std", f"{cls_vec.std():.4f}")
        sc3.metric("Min", f"{cls_vec.min():.4f}")
        sc4.metric("Max", f"{cls_vec.max():.4f}")

        # Token similarity heatmap
        st.markdown("**Cosine similarity between all token embeddings:**")
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:0.80rem;color:#475569;"
            "margin-bottom:8px;'>Warm = similar embeddings; cool = different. "
            "Tokens with similar meanings or roles cluster together.</p>",
            unsafe_allow_html=True
        )
        norms = hidden / (hidden.norm(dim=-1, keepdim=True) + 1e-8)
        sim   = (norms @ norms.T).numpy()

        fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.6), max(5, len(tokens) * 0.55)))
        fig.patch.set_facecolor("white")
        im = ax.imshow(sim, cmap="RdYlBu_r", vmin=-1, vmax=1,
                       interpolation="nearest", aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9, color="#374151")
        ax.set_yticklabels(tokens, fontsize=9, color="#374151")
        ax.set_facecolor("white")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Cosine similarity", fontsize=8, color="#475569")
        cb.ax.tick_params(labelsize=8, colors="#64748b")
        ax.set_title("Token embedding cosine similarity (BERT last layer)",
                     color="#0f172a", fontsize=10, pad=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        tip("""Notice how [CLS] often has low similarity to all other tokens.
        That is expected: it is not a regular word, so its embedding is in a
        different part of the vector space, encoding sentence-level semantics
        rather than word-level meaning.""")

    else:
        info("Click the button to run the full BERT forward pass and inspect the output.")

    with st.expander("Show the embedding extraction code", expanded=False):
        st.code("""
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased")

enc = tokenizer("The cat sat on the mat.", return_tensors="pt")

with torch.no_grad():
    out = model(**enc)

# out.last_hidden_state: (batch, seq_len, 768) -- final layer output for all tokens
# out.pooler_output:     (batch, 768)          -- [CLS] passed through a linear+tanh layer

hidden = out.last_hidden_state[0]   # (seq_len, 768)
cls_embedding = hidden[0]           # (768,) -- the [CLS] token
word_embeddings = hidden[1:-1]      # (seq_len-2, 768) -- without [CLS] and [SEP]
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("6"):
    section_header("6", "Fine-tuning for Downstream Tasks",
                   "Same pretrained model, different task head attached to the output")

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:16px;'>
    Fine-tuning means taking the pretrained BERT weights and training the entire model
    (plus a small task-specific head) on labeled data for a specific task.
    Because BERT already understands language deeply, this typically takes 2-4 epochs
    on a few thousand labeled examples to achieve strong results.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("### Four fine-tuning patterns")
    st.markdown("""
    <div class="enc-cards" style="grid-template-columns:1fr 1fr;">
      <div class="enc-card" style="background:#eef2ff;border-left:4px solid #4f46e5;">
        <div class="enc-card-title" style="color:#3730a3;">
          A. Sentence Classification
        </div>
        <div class="enc-card-body">
          Input: one sentence.<br>
          Output: the <strong>[CLS] vector (768)</strong> fed to a linear layer
          producing class logits.<br><br>
          <strong>Tasks:</strong> sentiment analysis, spam detection, topic classification.<br>
          <strong>Head:</strong> <code>nn.Linear(768, num_classes)</code>
        </div>
      </div>
      <div class="enc-card" style="background:#f0fdf4;border-left:4px solid #16a34a;">
        <div class="enc-card-title" style="color:#14532d;">
          B. Token Classification (NER)
        </div>
        <div class="enc-card-body">
          Input: one sentence.<br>
          Output: <strong>every token's vector (seq, 768)</strong> fed to a linear layer
          producing a label per token.<br><br>
          <strong>Tasks:</strong> named entity recognition, POS tagging.<br>
          <strong>Head:</strong> <code>nn.Linear(768, num_labels)</code> applied to each token
        </div>
      </div>
      <div class="enc-card" style="background:#fffbeb;border-left:4px solid #d97706;">
        <div class="enc-card-title" style="color:#92400e;">
          C. Question Answering
        </div>
        <div class="enc-card-body">
          Input: [CLS] question [SEP] context [SEP].<br>
          Output: two scores per context token: probability it is the
          <strong>start</strong> or <strong>end</strong> of the answer span.<br><br>
          <strong>Tasks:</strong> SQuAD-style extractive QA.<br>
          <strong>Head:</strong> <code>nn.Linear(768, 2)</code> applied to each token
        </div>
      </div>
      <div class="enc-card" style="background:#fdf4ff;border-left:4px solid #a855f7;">
        <div class="enc-card-title" style="color:#7e22ce;">
          D. Sentence Pair Tasks
        </div>
        <div class="enc-card-body">
          Input: [CLS] sentence A [SEP] sentence B [SEP].<br>
          Output: <strong>[CLS] vector</strong> used for relationship label.<br><br>
          <strong>Tasks:</strong> natural language inference (entails/contradicts/neutral),
          semantic similarity, paraphrase detection.<br>
          <strong>Head:</strong> <code>nn.Linear(768, num_classes)</code>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tip("""In all four patterns, <strong>the BERT encoder weights are shared and updated
    during fine-tuning</strong>. You are not freezing BERT and only training the head.
    The entire model is trained end-to-end, but with a small learning rate (2e-5 to 5e-5)
    so the pretrained weights do not change too drastically.""")

    st.divider()

    # Live sentiment demo
    st.markdown("### Live demo: sentiment analysis (fine-tuned BERT)")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:12px;'>Uses DistilBERT fine-tuned on SST-2 (Stanford Sentiment Treebank). "
        "Enter any sentence and see the prediction with confidence score.</p>",
        unsafe_allow_html=True
    )

    sent_input = st.text_area(
        "Enter text for sentiment analysis",
        value="The cat sat on the mat and looked very content.",
        height=80
    )

    if st.button("Analyze sentiment", type="primary"):
        with st.spinner("Running sentiment analysis..."):
            sentiment_pipe = load_sentiment()
            result = sentiment_pipe(sent_input)[0]

        label = result["label"]
        score = result["score"]
        color = "#16a34a" if label == "POSITIVE" else "#dc2626"
        bg    = "#f0fdf4" if label == "POSITIVE" else "#fef2f2"

        st.markdown(f"""
        <div style="background:{bg};border:2px solid {color};border-radius:10px;
                    padding:16px 20px;text-align:center;margin:12px 0;">
          <div style="font-size:1.4rem;font-weight:800;color:{color};">{label}</div>
          <div style="font-size:1rem;color:#374151;margin-top:4px;">
            Confidence: <strong>{score*100:.1f}%</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar
        fig, ax = plt.subplots(figsize=(6, 1.2))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        pos_score = score if label == "POSITIVE" else 1 - score
        neg_score = 1 - pos_score
        ax.barh([""], [pos_score], color="#16a34a", label="POSITIVE")
        ax.barh([""], [neg_score], left=[pos_score], color="#dc2626", label="NEGATIVE")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color="#64748b")
        ax.tick_params(left=False, labelleft=False)
        ax.spines[:].set_color("#e2e8f0")
        ax.legend(loc="upper right", fontsize=8, framealpha=0)
        ax.set_title("Sentiment probability split", fontsize=9, color="#374151", pad=6)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    else:
        info("Click the button to run sentiment analysis on the text above.")

    with st.expander("Show the fine-tuning code for text classification", expanded=False):
        st.code("""
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2   # positive / negative
)

# Tokenize a batch of sentences
texts  = ["I love this movie!", "This film was terrible."]
labels = torch.tensor([1, 0])   # 1=positive, 0=negative

enc = tokenizer(texts, padding=True, truncation=True,
                return_tensors="pt", max_length=128)

# Forward pass -- the model adds the classification head internally
outputs = model(**enc, labels=labels)
loss    = outputs.loss          # cross-entropy on [CLS] output
logits  = outputs.logits        # (batch, 2)

# Fine-tune with small LR -- BERT weights are sensitive to large updates
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss.backward()
optimizer.step()
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: ENCODER TO DECODER
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("7"):
    section_header("7", "Sending Encoder Output to a Decoder",
                   "When you need to generate text: cross-attention bridges BERT to the decoder")

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:16px;'>
    BERT alone cannot generate text: it is an encoder only. But its output, a rich
    (seq_len, 768) matrix of contextual embeddings, can be passed to a Transformer decoder
    via cross-attention. This is exactly how sequence-to-sequence models like T5 and BART work.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("### The two paths from BERT output")
    st.markdown("""
    <div class="enc-cards">
      <div class="enc-card" style="background:#eef2ff;border-left:4px solid #4f46e5;">
        <div class="enc-card-title" style="color:#3730a3;">Path A: Classification head (covered in Sec. 6)</div>
        <div class="enc-card-body">
          Take the [CLS] output vector (768) and feed it to a linear layer.
          No decoder needed. Produces a fixed-size output (class label, regression score).
          Fast and simple. Used for: sentiment, NER, QA span extraction.
        </div>
      </div>
      <div class="enc-card" style="background:#f0fdf4;border-left:4px solid #16a34a;">
        <div class="enc-card-title" style="color:#14532d;">Path B: Decoder with cross-attention (this section)</div>
        <div class="enc-card-body">
          Pass the full encoder output matrix (seq_len, 768) to a Transformer decoder.
          The decoder generates tokens one at a time. Each decoder layer attends to
          the encoder output via cross-attention.
          Used for: translation, summarization, text generation.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### How cross-attention works")
    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:12px;'>
    In a standard self-attention block (what you studied), Q, K, and V all come from the
    same sequence. In cross-attention, <strong>Q comes from the decoder</strong> and
    <strong>K and V come from the encoder output</strong>.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="math-box">
      <div class="math-step">
        <div class="math-step-num">1</div>
        <div>
          The encoder (BERT) processes the source sentence and produces
          <strong>K</strong> and <strong>V</strong> matrices from its final hidden states:
          shape (src_len, 768).
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">2</div>
        <div>
          The decoder generates output tokens one at a time. At each step, it produces
          a <strong>Q</strong> vector from its current partial output sequence.
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">3</div>
        <div>
          Cross-attention computes:
          <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;">
          Attention(Q<sub>dec</sub>, K<sub>enc</sub>, V<sub>enc</sub>)
          = softmax(Q<sub>dec</sub> K<sub>enc</sub><sup>T</sup> / &radic;d) V<sub>enc</sub>
          </span><br>
          The decoder can now attend to any position in the source sentence.
        </div>
      </div>
      <div class="math-step" style="margin-bottom:0;">
        <div class="math-step-num">4</div>
        <div>
          Each decoder layer has three sub-layers: masked self-attention (on target tokens),
          cross-attention (attending to encoder output), and a feed-forward network.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Encoder-decoder architecture (full picture)")
    st.markdown("""
    <div style="overflow-x:auto;margin:8px 0 20px;">
    <table style="border-collapse:collapse;width:100%;font-family:Inter,sans-serif;font-size:0.82rem;">
      <thead>
        <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
          <th style="padding:10px 14px;color:#374151;font-weight:700;width:25%;">Stage</th>
          <th style="padding:10px 14px;color:#1d4ed8;font-weight:700;width:35%;">Encoder (BERT side)</th>
          <th style="padding:10px 14px;color:#14532d;font-weight:700;width:40%;">Decoder (generation side)</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 14px;font-weight:600;">Input</td>
          <td style="padding:10px 14px;color:#1d4ed8;">Source tokens (e.g., English sentence)</td>
          <td style="padding:10px 14px;color:#14532d;">Target tokens generated so far (e.g., French)</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;background:#fafbff;">
          <td style="padding:10px 14px;font-weight:600;">Attention type</td>
          <td style="padding:10px 14px;color:#1d4ed8;">Self-attention (bidirectional)</td>
          <td style="padding:10px 14px;color:#14532d;">
            1. Masked self-attention (causal, left-to-right)<br>
            2. Cross-attention (Q from decoder, K+V from encoder)
          </td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:10px 14px;font-weight:600;">Output</td>
          <td style="padding:10px 14px;color:#1d4ed8;">(src_len, 768) context matrix, passed to decoder</td>
          <td style="padding:10px 14px;color:#14532d;">Probability over vocabulary for next token</td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;background:#fafbff;">
          <td style="padding:10px 14px;font-weight:600;">Models using this</td>
          <td style="padding:10px 14px;color:#1d4ed8;">T5 encoder, BART encoder</td>
          <td style="padding:10px 14px;color:#14532d;">T5 decoder, BART decoder, mT5</td>
        </tr>
      </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    tip("""T5 ("Text-to-Text Transfer Transformer") treats every NLP task as text-to-text:
    classification becomes "Input: sentence. Output: positive or negative."
    BART uses a BERT-like encoder but corrupts the input (masking, shuffling) and trains
    a decoder to reconstruct the original. Both are examples of this encoder-decoder pattern.""")

    with st.expander("Show the cross-attention mechanism in PyTorch", expanded=False):
        st.code("""
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads     # 64
        # Q comes from the decoder, K and V come from the encoder
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, decoder_hidden, encoder_output):
        # decoder_hidden:  (B, tgt_len, 768) -- Q source
        # encoder_output:  (B, src_len, 768) -- K, V source
        B, T, D = decoder_hidden.shape
        S = encoder_output.shape[1]

        Q = self.W_q(decoder_hidden).reshape(B, T, self.n_heads, self.head_dim).transpose(1,2)
        K = self.W_k(encoder_output).reshape(B, S, self.n_heads, self.head_dim).transpose(1,2)
        V = self.W_v(encoder_output).reshape(B, S, self.n_heads, self.head_dim).transpose(1,2)

        # Each decoder position attends to ALL encoder positions
        attn = (Q @ K.transpose(-2,-1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(-1)                            # (B, H, T, S)

        out = (attn @ V).transpose(1,2).reshape(B, T, D)  # (B, T, 768)
        return self.W_o(out)
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: BERT VARIANTS
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("8"):
    section_header("8", "BERT Variants",
                   "How the community extended BERT after the original 2018 paper")

    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:16px;'>
    BERT's architecture became a starting point. Researchers found that pretraining details
    (data size, masking strategy, NSP removal) matter as much as the architecture itself.
    The table below shows the most important variants you will encounter.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="comp-table" style="table-layout:fixed;width:100%;">
      <colgroup>
        <col style="width:14%;">
        <col style="width:10%;">
        <col style="width:8%;">
        <col style="width:8%;">
        <col style="width:8%;">
        <col style="width:52%;">
      </colgroup>
      <thead>
        <tr>
          <th>Model</th>
          <th>Params</th>
          <th>Layers</th>
          <th>Hidden</th>
          <th>Heads</th>
          <th style="text-align:left;">Key difference from BERT-Base</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background:#eef2ff;font-weight:600;">
          <td style="color:#4f46e5;">BERT-Base</td>
          <td>110 M</td><td>12</td><td>768</td><td>12</td>
          <td style="text-align:left;font-weight:400;color:#374151;">
            The original. MLM + NSP on Wikipedia + BooksCorpus (3.3B words).
            Trained for 1M steps with batch size 256.
          </td>
        </tr>
        <tr>
          <td><strong>BERT-Large</strong></td>
          <td>340 M</td><td>24</td><td>1024</td><td>16</td>
          <td style="text-align:left;color:#374151;">
            3x deeper and wider than Base. Better accuracy, much more compute.
            Fine-tuning often requires 32 GB GPU.
          </td>
        </tr>
        <tr>
          <td><strong>RoBERTa</strong></td>
          <td>125 M</td><td>12</td><td>768</td><td>12</td>
          <td style="text-align:left;color:#374151;">
            Same architecture as BERT-Base but: removes NSP (found unhelpful),
            trains on 10x more data, uses dynamic masking (new mask each epoch),
            larger batches (8k). Consistently outperforms BERT on benchmarks.
          </td>
        </tr>
        <tr>
          <td><strong>DistilBERT</strong></td>
          <td>66 M</td><td>6</td><td>768</td><td>12</td>
          <td style="text-align:left;color:#374151;">
            Knowledge distillation from BERT-Base. 40% fewer parameters,
            60% faster, retains 97% of BERT's performance on GLUE.
            Good for edge deployment or Colab with limited RAM.
          </td>
        </tr>
        <tr>
          <td><strong>ALBERT</strong></td>
          <td>12 M</td><td>12</td><td>768</td><td>12</td>
          <td style="text-align:left;color:#374151;">
            Shares weights across all 12 encoder layers (same weights, 12 passes).
            Factorizes the embedding matrix. Uses SOP (sentence order prediction)
            instead of NSP. Very small parameter count.
          </td>
        </tr>
        <tr>
          <td><strong>BioBERT</strong></td>
          <td>110 M</td><td>12</td><td>768</td><td>12</td>
          <td style="text-align:left;color:#374151;">
            BERT-Base initialized, then further pretrained on PubMed abstracts
            and PMC full-text articles (18B words of biomedical text).
            Significantly better at biomedical NER, relation extraction, QA.
          </td>
        </tr>
        <tr>
          <td><strong>SciBERT</strong></td>
          <td>110 M</td><td>12</td><td>768</td><td>12</td>
          <td style="text-align:left;color:#374151;">
            Trained from scratch on 1.14M scientific papers from Semantic Scholar.
            Custom scientific vocabulary. Stronger than BioBERT on general science tasks.
          </td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("### Which variant should you use?")
    st.markdown("""
    <div class="enc-cards">
      <div class="enc-card" style="background:#f0fdf4;border-left:4px solid #16a34a;">
        <div class="enc-card-title" style="color:#14532d;">General NLP tasks</div>
        <div class="enc-card-body">
          Start with <strong>RoBERTa-base</strong>. It consistently outperforms BERT-Base
          with no extra complexity. If you need smaller: use DistilBERT.
        </div>
      </div>
      <div class="enc-card" style="background:#dbeafe;border-left:4px solid #2563eb;">
        <div class="enc-card-title" style="color:#1d4ed8;">Domain-specific text</div>
        <div class="enc-card-body">
          Biomedical: <strong>BioBERT</strong> or <strong>PubMedBERT</strong>.
          Legal: <strong>LegalBERT</strong>. Scientific: <strong>SciBERT</strong>.
          Domain-specific pretraining often gives +5-10% on specialized tasks.
        </div>
      </div>
      <div class="enc-card" style="background:#fffbeb;border-left:4px solid #d97706;">
        <div class="enc-card-title" style="color:#92400e;">Production / low resource</div>
        <div class="enc-card-body">
          <strong>DistilBERT</strong> or <strong>ALBERT-base</strong> for inference speed
          and memory. Both run well on CPU and fit in 8 GB RAM.
        </div>
      </div>
      <div class="enc-card" style="background:#fdf4ff;border-left:4px solid #a855f7;">
        <div class="enc-card-title" style="color:#7e22ce;">Maximum accuracy</div>
        <div class="enc-card-body">
          <strong>RoBERTa-large</strong> or <strong>DeBERTa-large</strong>
          (DeBERTa uses disentangled attention and outperforms RoBERTa on GLUE/SuperGLUE).
          Requires 16-40 GB GPU for fine-tuning.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Show how to load any variant from HuggingFace", expanded=False):
        st.code("""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Swap the model name to switch variants -- the rest of the code stays the same
MODEL = "roberta-base"           # or "distilbert-base-uncased", "allenai/scibert_scivocab_uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

# AutoTokenizer and AutoModel automatically detect the architecture
# from the model card -- no need to import BertTokenizer vs RobertaTokenizer separately.

enc = tokenizer("The cat sat on the mat.", return_tensors="pt",
                truncation=True, max_length=512)
out = model(**enc)
print(out.logits.shape)   # (1, 2)
        """, language="python")

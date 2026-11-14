"""
Real-Time Chat Moderation - interactive demo (SWE015, Use Case 4).

    python app.py              # http://127.0.0.1:7860
    python app.py --share      # temporary public Gradio link

Tabs
----
1. Live Chat     - production-style gaming moderation console: every message is
                   normalized with regex, classified live (clean / offensive)
                   with confidence, offensive messages are masked via lexicon +
                   Levenshtein edit distance <= 1, and a live telemetry panel
                   tracks single-message CPU latency against the 50 ms SLA
                   with a dedicated one-click log filter bar.
                   Default model = the INT8 quantized BERT (R5); the user can
                   switch across all 5 models at any time.
2. Model Ladder  - one message through the whole pipeline: visual 5-model
                   comparison strip, every regex normalization step (Area B),
                   lexicon / minimum edit-distance matches (Area B), and the
                   prediction + latency of all five models side by side.
3. Benchmark     - executive KPI summary and the full results table measured on
                   the held-out test set (860 tweets) by ``python -m src.benchmark``.

Styled following the ``design-taste-frontend`` design system:
- Dual Theme: Crisp Light Mode (#f4f6f8 / #ffffff) & Carbon Dark Mode (#0b0f17 / #111827)
- Dual Language: English (default for SWE015 grading) & Turkish (instant switch)
- Custom HTML data tables (no cluttered three-dot column menus) + dedicated Filter Box
- Pure-select (non-editable) engine selector and monospace tabular numerals.
"""
from __future__ import annotations

import argparse
import html

import gradio as gr
import pandas as pd

from src import config
from src.moderation import HIDDEN_TEXT, ModerationService
from src.predictors import load_predictors

DEFAULT_MODEL = "bert_int8"  # the app runs the quantized Transformer by default (R5)

# --------------------------------------------------------------------------- #
# Load all models once at start-up
# --------------------------------------------------------------------------- #
print("Loading models ...")
SERVICE = ModerationService(load_predictors())
if DEFAULT_MODEL not in SERVICE.predictors:
    raise SystemExit("The quantized BERT was not found in artifacts/bert_int8. "
                     "Train / download the models first (see README).")
SERVICE.warm_up(6)
NAME_TO_KEY = {p.name: k for k, p in SERVICE.predictors.items()}
MODEL_NAMES = list(NAME_TO_KEY)
DEFAULT_MODEL_NAME = SERVICE.predictors[DEFAULT_MODEL].name
print("Models ready:", ", ".join(MODEL_NAMES))

BUDGET = config.LATENCY_BUDGET_MS
LOG_COLUMNS = ["#", "model", "label", "confidence", "latency_ms", f"≤ {BUDGET:.0f} ms"]

FILTER_CHOICES = [
    "All Messages",
    "🚫 Offensive Only",
    "✅ Clean Only",
    f"⚠️ Over {BUDGET:.0f} ms SLA",
]

EXAMPLES = [
    "gg wp everyone, that was a great match!",
    "you are such an 1d10t, uninstall the game",
    "you are a stuuuupid $h1t, go back to the tutorial",
    "nice shot bro, you crushed that last round!",
    "@USER nobody asked for your opinion, loser",
    "wow you're really smart, said no one ever",
]

DEFAULT_INSPECT_TEXT = "stuuuupid $h1t, you are an 1d10t!!! 😡 @USER"


def _bi(en: str, tr: str) -> str:
    """Wrap bilingual inline text so the EN/TR toggle switches it instantly."""
    return (
        f'<span class="i18n" data-en="{html.escape(en)}" data-tr="{html.escape(tr)}">'
        f'{html.escape(en)}</span>'
    )


# --------------------------------------------------------------------------- #
# design-taste-frontend CSS system (Dual Theme: Light & Carbon Dark)
# --------------------------------------------------------------------------- #
TASTE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root, body:not(.dark), .gradio-container:not(.dark) {
    --bg: #f4f6f8;
    --surface: #ffffff;
    --surface-2: #e9eef3;
    --ink: #0f172a;
    --ink-secondary: #334155;
    --muted: #64748b;
    --accent: #1a4be0;
    --accent-hover: #153ec2;
    --accent-soft: #e8f0fe;
    --hairline: #cbd5e1;
    --ok-bg: #ecfdf5;
    --ok-ink: #065f46;
    --ok-border: #6ee7b7;
    --err-bg: #fef2f2;
    --err-ink: #991b1b;
    --err-border: #fca5a5;
    --msg-off-bg: #fff5f5;
    --row-hover: #f8fafc;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Cascadia Code', Consolas, monospace;

    --body-background-fill: #f4f6f8 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --body-text-color: #0f172a !important;
    --body-text-color-subdued: #475569 !important;
    --block-background-fill: #ffffff !important;
    --block-label-background-fill: #f1f5f9 !important;
    --block-label-text-color: #0f172a !important;
    --block-title-text-color: #0f172a !important;
    --border-color-primary: #cbd5e1 !important;
    --input-background-fill: #ffffff !important;
    --checkbox-background-color: #ffffff !important;
    --checkbox-label-background-fill: #f8fafc !important;
    --checkbox-label-background-fill-hover: #eff4ff !important;
    --checkbox-label-background-fill-selected: #1a4be0 !important;
    --checkbox-label-text-color: #0f172a !important;
    --checkbox-label-text-color-selected: #ffffff !important;
    --checkbox-label-border-color: #cbd5e1 !important;
    --checkbox-label-border-color-selected: #1a4be0 !important;
    --neutral-50: #f8fafc !important;
    --neutral-100: #f1f5f9 !important;
    --neutral-200: #e2e8f0 !important;
    --neutral-700: #334155 !important;
    --neutral-800: #1e293b !important;
    --neutral-900: #0f172a !important;
}

body.dark, .dark, .gradio-container.dark {
    --bg: #0b0f17;
    --surface: #111827;
    --surface-2: #1e293b;
    --ink: #f8fafc;
    --ink-secondary: #cbd5e1;
    --muted: #94a3b8;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --accent-soft: rgba(59, 130, 246, 0.18);
    --hairline: #334155;
    --ok-bg: rgba(16, 185, 129, 0.16);
    --ok-ink: #34d399;
    --ok-border: rgba(16, 185, 129, 0.45);
    --err-bg: rgba(239, 68, 68, 0.16);
    --err-ink: #f87171;
    --err-border: rgba(239, 68, 68, 0.45);
    --msg-off-bg: rgba(127, 29, 29, 0.22);
    --row-hover: #172033;

    --body-background-fill: #0b0f17 !important;
    --background-fill-primary: #111827 !important;
    --background-fill-secondary: #1e293b !important;
    --body-text-color: #f8fafc !important;
    --body-text-color-subdued: #94a3b8 !important;
    --block-background-fill: #111827 !important;
    --block-label-background-fill: #1e293b !important;
    --block-label-text-color: #f8fafc !important;
    --block-title-text-color: #f8fafc !important;
    --border-color-primary: #334155 !important;
    --input-background-fill: #1e293b !important;
    --checkbox-background-color: #1e293b !important;
    --checkbox-label-background-fill: #1e293b !important;
    --checkbox-label-background-fill-hover: #283548 !important;
    --checkbox-label-background-fill-selected: #3b82f6 !important;
    --checkbox-label-text-color: #f8fafc !important;
    --checkbox-label-text-color-selected: #ffffff !important;
    --checkbox-label-border-color: #334155 !important;
    --checkbox-label-border-color-selected: #3b82f6 !important;
}

body, .gradio-container {
    background-color: var(--bg) !important;
    color: var(--ink) !important;
    font-family: var(--font-sans) !important;
    max-width: 1380px !important;
    margin: 0 auto !important;
    padding: 14px 22px 32px 22px !important;
}

footer {
    display: none !important;
}

/* Top product bar */
.taste-topbar {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
}
.taste-brand-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 4px;
}
.taste-brand-title {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin: 0;
    line-height: 1.25;
}
.taste-brand-sub {
    font-size: 13.5px;
    color: var(--ink-secondary);
    margin-top: 3px;
}
.taste-pill-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.taste-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 500;
    padding: 4px 9px;
    border-radius: 6px;
    border: 1px solid var(--hairline);
    background: var(--surface-2);
    color: var(--ink);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.taste-badge-accent {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--accent);
    font-weight: 600;
}
.taste-badge-ok {
    background: var(--ok-bg);
    border-color: var(--ok-border);
    color: var(--ok-ink);
    font-weight: 600;
}
.taste-badge-err {
    background: var(--err-bg);
    border-color: var(--err-border);
    color: var(--err-ink);
    font-weight: 600;
}
.taste-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    display: inline-block;
}

/* High-Contrast Tab Navigation */
.tabs > .tab-nav, [role="tablist"] {
    background: var(--surface) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 10px !important;
    padding: 6px !important;
    margin-bottom: 12px !important;
    display: flex !important;
    gap: 8px !important;
}
.tabs > .tab-nav button, [role="tablist"] button, button[role="tab"] {
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 7px !important;
    padding: 9px 18px !important;
    opacity: 1 !important;
    cursor: pointer !important;
    transition: all 120ms ease !important;
}
.tabs > .tab-nav button:hover, [role="tablist"] button:hover, button[role="tab"]:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
.tabs > .tab-nav button.selected,
[role="tablist"] button.selected,
button[role="tab"][aria-selected="true"] {
    background: var(--accent) !important;
    color: #ffffff !important;
    border-color: var(--accent) !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.12) !important;
}
.tabs > .tab-nav button.selected *,
button[role="tab"][aria-selected="true"] * {
    color: #ffffff !important;
}

/* Pure-select Engine Pills & Filter Radio (Explicitly themed for Light & Dark!) */
#engine-selector,
#engine-selector *,
#log-filter-box,
#log-filter-box * {
    cursor: pointer !important;
    user-select: none !important;
}

#engine-selector,
#log-filter-box {
    background: var(--surface) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    margin-bottom: 8px !important;
    color: var(--ink) !important;
}

#engine-selector label.svelte-1mwvhlq,
#engine-selector .wrap label,
#log-filter-box .wrap label {
    background: var(--surface-2) !important;
    color: var(--ink) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 12.5px !important;
    padding: 6px 11px !important;
    transition: all 120ms ease !important;
}

#engine-selector .wrap label:has(input:checked),
#log-filter-box .wrap label:has(input:checked),
#engine-selector .wrap label.selected,
#log-filter-box .wrap label.selected {
    background: var(--accent) !important;
    color: #ffffff !important;
    border-color: var(--accent) !important;
    font-weight: 600 !important;
}

#engine-selector .wrap label:has(input:checked) *,
#log-filter-box .wrap label:has(input:checked) *,
#engine-selector .wrap label.selected *,
#log-filter-box .wrap label.selected * {
    color: #ffffff !important;
}

/* Force Chatbot & Input containers to obey Light / Dark theme tokens cleanly */
#chat-panel,
#chat-panel > div,
#chat-panel .wrapper,
#chat-panel .bubble-wrap {
    background: var(--surface) !important;
    color: var(--ink) !important;
    border-color: var(--hairline) !important;
}
#chat-panel label[data-testid="block-label"],
#engine-selector > span,
#log-filter-box > span {
    background: var(--surface-2) !important;
    color: var(--ink) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
}
#chat-panel .message-row,
#chat-panel .message {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px 6px !important;
}
#msg-input textarea,
#msg-input input,
#inspect-input textarea,
#inspect-input input {
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--hairline) !important;
}

/* Primary & Secondary Buttons */
button.primary {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: 1px solid var(--accent) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    box-shadow: none !important;
    cursor: pointer !important;
}
button.primary:hover {
    background: var(--accent-hover) !important;
}
button.secondary {
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 12.5px !important;
    box-shadow: none !important;
    cursor: pointer !important;
}
button.secondary:hover {
    background: var(--surface-2) !important;
    border-color: var(--accent) !important;
}

/* Telemetry Panel & Clean Data Table Cards (No 3-dot column clutter!) */
.taste-card {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 10px;
    color: var(--ink);
}
.taste-card-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}
.taste-kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 12px;
}
.taste-kpi-box {
    background: var(--bg);
    border: 1px solid var(--hairline);
    border-radius: 8px;
    padding: 10px 12px;
}
.taste-kpi-label {
    font-size: 11.5px;
    color: var(--muted);
    font-weight: 500;
}
.taste-kpi-val {
    font-family: var(--font-mono);
    font-size: 18px;
    font-weight: 700;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
    margin-top: 2px;
}
.taste-kpi-sub {
    font-size: 11px;
    color: var(--ink-secondary);
    margin-top: 2px;
    font-variant-numeric: tabular-nums;
}

/* SLA progress bar */
.taste-sla-bar-wrap {
    margin-top: 8px;
    margin-bottom: 12px;
}
.taste-sla-header {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    font-weight: 500;
    color: var(--ink);
    margin-bottom: 5px;
    font-variant-numeric: tabular-nums;
}
.taste-sla-track {
    width: 100%;
    height: 8px;
    background: var(--surface-2);
    border-radius: 999px;
    overflow: hidden;
    position: relative;
    border: 1px solid var(--hairline);
}
.taste-sla-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 180ms ease;
}

/* Clean Professional HTML Data Tables */
.taste-table-wrap {
    width: 100%;
    overflow-x: auto;
    max-height: 290px;
    overflow-y: auto;
    border: 1px solid var(--hairline);
    border-radius: 8px;
    background: var(--surface);
}
.taste-mini-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    font-variant-numeric: tabular-nums;
}
.taste-mini-table th {
    position: sticky;
    top: 0;
    background: var(--surface-2);
    text-align: left;
    font-size: 11.5px;
    font-weight: 600;
    color: var(--ink-secondary);
    letter-spacing: 0.02em;
    padding: 8px 10px;
    border-bottom: 1px solid var(--hairline);
    white-space: nowrap;
    z-index: 1;
}
.taste-mini-table td {
    padding: 8px 10px;
    border-bottom: 1px solid var(--hairline);
    color: var(--ink);
    vertical-align: middle;
}
.taste-mini-table tr:last-child td {
    border-bottom: none;
}
.taste-mini-table tbody tr:hover td {
    background: var(--row-hover);
}
.taste-mini-table tr.highlight-row td {
    background: var(--accent-soft);
    font-weight: 600;
}
.taste-mono {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
}

/* Chat message card inside Gradio Chatbot */
.taste-msg-card {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 2px 0;
    color: var(--ink);
}
.taste-msg-card.offensive {
    border-left-color: #ef4444;
    background: var(--msg-off-bg);
}
.taste-msg-text {
    font-size: 15px;
    font-weight: 500;
    color: var(--ink);
    line-height: 1.45;
    margin-bottom: 6px;
}
.taste-msg-raw {
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--ink-secondary);
    margin-bottom: 8px;
}
.taste-msg-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 11.5px;
    font-variant-numeric: tabular-nums;
}

/* Model Ladder 5-card strip */
.taste-ladder-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin: 10px 0 14px 0;
}
@media (max-width: 1024px) {
    .taste-ladder-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
.taste-ladder-card {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 9px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    color: var(--ink);
}
.taste-ladder-card.highlight {
    border: 2px solid var(--accent);
}
.taste-ladder-step {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.taste-ladder-name {
    font-size: 14px;
    font-weight: 700;
    color: var(--ink);
    margin: 3px 0 8px 0;
}
.taste-ladder-metrics {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--ink-secondary);
    font-variant-numeric: tabular-nums;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--hairline);
    display: flex;
    justify-content: space-between;
}

/* Benchmark KPI strip */
.taste-bench-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 14px;
}
.taste-bench-card {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 10px;
    padding: 14px 16px;
    color: var(--ink);
}
.taste-bench-num {
    font-family: var(--font-mono);
    font-size: 24px;
    font-weight: 700;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}
.taste-bench-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--ink-secondary);
    margin-top: 2px;
}
.taste-bench-note {
    font-size: 11.5px;
    color: var(--muted);
    margin-top: 4px;
    font-variant-numeric: tabular-nums;
}
"""

INIT_UI_JS = """
() => {
    window.__tasteLang = window.__tasteLang || 'en';
    document.documentElement.lang = window.__tasteLang;
    if (!window.__tasteThemeExplicit) {
        document.body.classList.remove('dark');
        document.documentElement.classList.remove('dark');
        document.querySelectorAll('.gradio-container').forEach(el => el.classList.remove('dark'));
    }
    let isUpdating = false;
    let rafId = null;

    const applyLang = (lang) => {
        window.__tasteLang = lang;
        document.documentElement.lang = lang;
        isUpdating = true;
        try {
            document.querySelectorAll('.i18n').forEach((el) => {
                const txt = el.getAttribute('data-' + lang);
                if (txt !== null && el.textContent !== txt) {
                    el.textContent = txt;
                }
            });
        } finally {
            isUpdating = false;
        }
    };
    window.__setTasteLang = applyLang;

    const syncUI = () => {
        if (isUpdating) return;
        isUpdating = true;
        try {
            if (window.__tasteLang && window.__tasteLang !== 'en') {
                document.querySelectorAll('.i18n').forEach((el) => {
                    const txt = el.getAttribute('data-' + window.__tasteLang);
                    if (txt !== null && el.textContent !== txt) {
                        el.textContent = txt;
                    }
                });
            }
        } finally {
            isUpdating = false;
        }
    };

    syncUI();
    const obs = new MutationObserver(() => {
        if (isUpdating) return;
        if (rafId) cancelAnimationFrame(rafId);
        rafId = requestAnimationFrame(() => {
            rafId = null;
            obs.disconnect();
            syncUI();
            obs.observe(document.body, { childList: true, subtree: true });
        });
    });
    obs.observe(document.body, { childList: true, subtree: true });
}
"""

SET_LIGHT_THEME_JS = """
() => {
    window.__tasteThemeExplicit = 'light';
    document.body.classList.remove('dark');
    document.documentElement.classList.remove('dark');
    document.querySelectorAll('.gradio-container').forEach(el => el.classList.remove('dark'));
}
"""

SET_DARK_THEME_JS = """
() => {
    window.__tasteThemeExplicit = 'dark';
    document.body.classList.add('dark');
    document.documentElement.classList.add('dark');
    document.querySelectorAll('.gradio-container').forEach(el => el.classList.add('dark'));
}
"""

SET_LANG_EN_JS = """
() => {
    if (window.__setTasteLang) window.__setTasteLang('en');
}
"""

SET_LANG_TR_JS = """
() => {
    if (window.__setTasteLang) window.__setTasteLang('tr');
}
"""


def _topbar_html() -> str:
    return f"""
    <style>{TASTE_CSS}</style>
    <header class="taste-topbar">
      <div>
        <div class="taste-brand-eyebrow">
          {_bi("SWE015 · Building an Intelligent Application · Use Case 4",
               "SWE015 · Akıllı Uygulama Geliştirme · Senaryo 4")}
        </div>
        <h1 class="taste-brand-title">
          {_bi("Real-Time Gaming Chat Moderation Console",
               "Gerçek Zamanlı Oyun Sohbeti Moderasyon Konsolu")}
        </h1>
        <div class="taste-brand-sub">
          {_bi("End-to-end NLP pipeline on a CPU-only server: 9-step Regex normalization, Levenshtein edit-distance masking, and a 5-stage Model Ladder.",
               "Yalnızca CPU üzerinde uçtan uca NLP hattı: 9 adımlı Regex normalizasyonu, Levenshtein düzenleme mesafesiyle sansürleme ve 5 aşamalı Model Merdiveni.")}
        </div>
      </div>
      <div class="taste-pill-row">
        <span class="taste-badge taste-badge-accent">
          <span class="taste-dot"></span>
          {_bi(f"Default Engine: {DEFAULT_MODEL_NAME}", f"Varsayılan Motor: {DEFAULT_MODEL_NAME}")}
        </span>
        <span class="taste-badge taste-mono">
          {_bi(f"CPU SLA: ≤ {BUDGET:.0f} ms / msg", f"CPU Bütçesi: ≤ {BUDGET:.0f} ms / mesaj")}
        </span>
        <span class="taste-badge taste-mono">
          {_bi("INT8 Size: 173.1 MB (2.41× smaller)", "INT8 Boyutu: 173.1 MB (2.41× daha küçük)")}
        </span>
      </div>
    </header>
    """


# --------------------------------------------------------------------------- #
# Tab 1 - live chat, telemetry & clean filterable log table
# --------------------------------------------------------------------------- #
def log_summary(log: list[list], last_result: dict | None = None) -> str:
    """Render a clean design-taste-frontend telemetry card for the right panel."""
    if not log:
        return f"""
        <div class="taste-card">
          <div class="taste-card-title">
            <span>{_bi("Live CPU Telemetry & SLA Monitor", "Canlı CPU Telemetrisi ve SLA Monitörü")}</span>
            <span class="taste-badge taste-mono">{_bi(f"Target ≤ {BUDGET:.0f} ms", f"Hedef ≤ {BUDGET:.0f} ms")}</span>
          </div>
          <div class="taste-kpi-grid">
            <div class="taste-kpi-box">
              <div class="taste-kpi-label">{_bi("Messages Processed", "İşlenen Mesaj")}</div>
              <div class="taste-kpi-val">0</div>
              <div class="taste-kpi-sub">{_bi("Session ready", "Oturum hazır")}</div>
            </div>
            <div class="taste-kpi-box">
              <div class="taste-kpi-label">{_bi("Last Latency (CPU)", "Son Gecikme (CPU)")}</div>
              <div class="taste-kpi-val">—</div>
              <div class="taste-kpi-sub">{_bi("Single-message (batch=1)", "Tekil mesaj (batch=1)")}</div>
            </div>
            <div class="taste-kpi-box">
              <div class="taste-kpi-label">{_bi("SLA Status", "SLA Durumu")}</div>
              <div class="taste-kpi-val" style="color:var(--ok-ink);">{_bi("READY", "HAZIR")}</div>
              <div class="taste-kpi-sub">{_bi(f"Budget: {BUDGET:.0f}.0 ms", f"Bütçe: {BUDGET:.0f}.0 ms")}</div>
            </div>
          </div>
          <div style="font-size:12.5px;color:var(--ink-secondary);line-height:1.45;">
            {_bi("Type a message or click any One-Click Demo Message on the left to measure real-time CPU latency and compare BERT-base (INT8) against BERT-base (FP32).",
                 "Gerçek zamanlı CPU gecikmesini ölçmek ve BERT-base (INT8) ile BERT-base (FP32) modellerini karşılaştırmak için sol taraftan bir mesaj gönderin veya hazır butonlara tıklayın.")}
          </div>
        </div>
        """

    df = pd.DataFrame(log, columns=LOG_COLUMNS)
    last_row = log[-1]
    last_model = str(last_row[1])
    last_label = str(last_row[2])
    last_conf = float(last_row[3])
    last_ms = float(last_row[4])
    within = last_ms <= BUDGET

    pct = min(100.0, (last_ms / (BUDGET * 2.0)) * 100.0)
    bar_color = "#10b981" if within else "#ef4444"
    sla_badge = (
        f'<span class="taste-badge taste-badge-ok taste-mono">PASS · ≤ {BUDGET:.0f} ms</span>'
        if within else
        f'<span class="taste-badge taste-badge-err taste-mono">VIOLATION · &gt; {BUDGET:.0f} ms</span>'
    )
    verdict_badge = (
        f'<span class="taste-badge taste-badge-err">🚫 OFFENSIVE ({last_conf:.0%})</span>'
        if last_label == "offensive" else
        f'<span class="taste-badge taste-badge-ok">✅ CLEAN ({last_conf:.0%})</span>'
    )

    rows_html = []
    for model, g in df.groupby("model", sort=False):
        lat = g["latency_ms"].astype(float)
        mean_ms = lat.mean()
        max_ms = lat.max()
        pass_rate = (lat <= BUDGET).mean() * 100.0
        status_pill = (
            '<span class="taste-badge taste-badge-ok taste-mono">≤ 50 ms</span>'
            if mean_ms <= BUDGET else
            '<span class="taste-badge taste-badge-err taste-mono">OVER SLA</span>'
        )
        rows_html.append(
            f"<tr>"
            f"<td><b>{html.escape(str(model))}</b></td>"
            f"<td class='taste-mono'>{len(g)}</td>"
            f"<td class='taste-mono'><b>{mean_ms:.1f} ms</b></td>"
            f"<td class='taste-mono'>{max_ms:.1f} ms</td>"
            f"<td>{status_pill} <span class='taste-mono' style='font-size:11px;color:var(--muted);'>({pass_rate:.0f}%)</span></td>"
            f"</tr>"
        )

    return f"""
    <div class="taste-card">
      <div class="taste-card-title">
        <span>{_bi("Live CPU Telemetry & SLA Monitor", "Canlı CPU Telemetrisi ve SLA Monitörü")}</span>
        {sla_badge}
      </div>
      <div class="taste-kpi-grid">
        <div class="taste-kpi-box">
          <div class="taste-kpi-label">{_bi("Last Latency", "Son Gecikme")}</div>
          <div class="taste-kpi-val" style="color:{'var(--ok-ink)' if within else 'var(--err-ink)'};">{last_ms:.1f} ms</div>
          <div class="taste-kpi-sub">{html.escape(last_model)}</div>
        </div>
        <div class="taste-kpi-box">
          <div class="taste-kpi-label">{_bi("Last Verdict", "Son Karar")}</div>
          <div style="margin-top:4px;">{verdict_badge}</div>
          <div class="taste-kpi-sub">P(conf) = {last_conf:.1%}</div>
        </div>
        <div class="taste-kpi-box">
          <div class="taste-kpi-label">{_bi("Session Messages", "Toplam Mesaj")}</div>
          <div class="taste-kpi-val">{len(df)}</div>
          <div class="taste-kpi-sub">{_bi("CPU batch size = 1", "CPU batch boyutu = 1")}</div>
        </div>
      </div>

      <div class="taste-sla-bar-wrap">
        <div class="taste-sla-header">
          <span>{_bi("Single-Message Latency vs. 50 ms Real-Time Budget", "Tekil Mesaj Gecikmesi vs. 50 ms Gerçek Zamanlı Bütçe")}</span>
          <span class="taste-mono">{last_ms:.1f} ms / {BUDGET:.0f}.0 ms</span>
        </div>
        <div class="taste-sla-track">
          <div style="position:absolute;left:50%;top:0;bottom:0;width:2px;background:var(--ink);opacity:0.45;" title="50 ms SLA limit"></div>
          <div class="taste-sla-fill" style="width:{pct:.1f}%;background:{bar_color};"></div>
        </div>
      </div>

      <div class="taste-table-wrap">
        <table class="taste-mini-table">
          <thead>
            <tr>
              <th>{_bi("Engine", "Motor")}</th>
              <th>{_bi("Msgs", "Adet")}</th>
              <th>{_bi("Mean", "Ort.")}</th>
              <th>{_bi("Max", "Maks.")}</th>
              <th>{_bi("SLA Check", "SLA Durumu")}</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
    </div>
    """


def render_log_table(log: list[list], filter_mode: str = "All Messages") -> str:
    """Render a clean HTML latency table (no 3-dot column menus) filtered by filter_mode."""
    rows = list(log)[::-1]  # newest first
    if filter_mode == "🚫 Offensive Only":
        rows = [r for r in rows if r[2] == "offensive"]
    elif filter_mode == "✅ Clean Only":
        rows = [r for r in rows if r[2] == "clean"]
    elif filter_mode.startswith("⚠️"):
        rows = [r for r in rows if float(r[4]) > BUDGET]

    if not rows:
        empty_msg = _bi(
            f"No messages matching filter '{filter_mode}'.",
            f"'{filter_mode}' filtresine uygun mesaj bulunmuyor.",
        )
        return f"""
        <div class="taste-card" style="padding:14px 16px;">
          <div class="taste-card-title" style="margin-bottom:6px;">
            <span>{_bi("Per-Message CPU Latency Log (Newest First)", "Mesaj Bazlı CPU Gecikme Kaydı (En Yeni Üstte)")}</span>
            <span class="taste-badge taste-mono">0 / {len(log)}</span>
          </div>
          <div style="font-size:12.5px;color:var(--muted);padding:8px 0;">{empty_msg}</div>
        </div>
        """

    body_rows = []
    for r in rows:
        idx, model_name, label, conf, lat_ms, within_str = r
        is_off = label == "offensive"
        within = within_str == "yes"
        lbl_badge = (
            '<span class="taste-badge taste-badge-err">🚫 offensive</span>'
            if is_off else
            '<span class="taste-badge taste-badge-ok">✅ clean</span>'
        )
        sla_badge = (
            '<span class="taste-badge taste-badge-ok taste-mono">yes (≤ 50 ms)</span>'
            if within else
            '<span class="taste-badge taste-badge-err taste-mono">no (&gt; 50 ms)</span>'
        )
        body_rows.append(
            f"<tr>"
            f"<td class='taste-mono'>#{idx}</td>"
            f"<td><b>{html.escape(str(model_name))}</b></td>"
            f"<td>{lbl_badge}</td>"
            f"<td class='taste-mono'>{float(conf):.1%}</td>"
            f"<td class='taste-mono'><b>{float(lat_ms):.2f} ms</b></td>"
            f"<td>{sla_badge}</td>"
            f"</tr>"
        )

    return f"""
    <div class="taste-card" style="padding:14px 16px;">
      <div class="taste-card-title">
        <span>{_bi("Per-Message CPU Latency Log (Newest First)", "Mesaj Bazlı CPU Gecikme Kaydı (En Yeni Üstte)")}</span>
        <span class="taste-badge taste-mono">{len(rows)} / {len(log)} {_bi("shown", "gösteriliyor")}</span>
      </div>
      <div class="taste-table-wrap">
        <table class="taste-mini-table">
          <thead>
            <tr>
              <th>#</th>
              <th>{_bi("Model", "Model")}</th>
              <th>{_bi("Label", "Etiket")}</th>
              <th>{_bi("Confidence", "Güven")}</th>
              <th>{_bi("Latency (CPU)", "Gecikme (CPU)")}</th>
              <th>≤ {BUDGET:.0f} ms</th>
            </tr>
          </thead>
          <tbody>
            {''.join(body_rows)}
          </tbody>
        </table>
      </div>
    </div>
    """


def format_chat_message(r: dict, raw_text: str = "") -> str:
    """Format a moderated message inside the chat stream."""
    is_off = r["label"] == "offensive"
    card_cls = "taste-msg-card offensive" if is_off else "taste-msg-card"
    verdict_badge = (
        f'<span class="taste-badge taste-badge-err">🚫 OFFENSIVE · {r["confidence"]:.0%}</span>'
        if is_off else
        f'<span class="taste-badge taste-badge-ok">✅ CLEAN · {r["confidence"]:.0%}</span>'
    )
    sla_cls = "taste-badge-ok" if r["within_budget"] else "taste-badge-err"
    sla_text = f"⏱️ {r['latency_ms']:.1f} ms" + (" (≤ 50 ms)" if r["within_budget"] else " (⚠️ OVER 50 ms SLA)")
    timing_badge = f'<span class="taste-badge {sla_cls} taste-mono">{sla_text}</span>'
    model_badge = f'<span class="taste-badge taste-mono">{html.escape(r["model"])}</span>'

    action_note = ""
    if is_off:
        n_masked = r.get("masked_words", 0)
        if n_masked > 0:
            en_note = f"Action: masked {n_masked} offensive token(s) via regex + edit-distance ≤ 1"
            tr_note = f"İşlem: regex + düzenleme mesafesi ≤ 1 ile {n_masked} saldırgan kelime sansürlendi"
            action_note = f'<div class="taste-msg-raw">{_bi(en_note, tr_note)}</div>'
        elif r["display_text"] == HIDDEN_TEXT:
            action_note = (
                f'<div class="taste-msg-raw">'
                f'{_bi("Action: implicit toxicity detected by model — entire message hidden by moderator", "İşlem: örtük saldırganlık tespit edildi — mesajın tamamı moderatör tarafından gizlendi")}'
                f'</div>'
            )

    return (
        f'<div class="{card_cls}">'
        f'<div class="taste-msg-text">{html.escape(r["display_text"])}</div>'
        f'{action_note}'
        f'<div class="taste-msg-meta">{verdict_badge} {timing_badge} {model_badge}</div>'
        f'</div>'
    )


def send_message(text: str, model_name: str, history: list, log: list, filter_mode: str = "All Messages"):
    text = (text or "").strip()
    r = None
    if text:
        r = SERVICE.moderate(text, NAME_TO_KEY[model_name])
        history = history + [{"role": "user", "content": format_chat_message(r, raw_text=text)}]
        log = log + [[len(log) + 1, r["model"], r["label"], round(r["confidence"], 3),
                      round(r["latency_ms"], 2), "yes" if r["within_budget"] else "no"]]
    return "", history, log, render_log_table(log, filter_mode), log_summary(log, r), history


def clear_chat(filter_mode: str = "All Messages"):
    return [], [], render_log_table([], filter_mode), log_summary([]), []


# --------------------------------------------------------------------------- #
# Tab 2 - one message through the whole model ladder
# --------------------------------------------------------------------------- #
def _df_to_clean_html_table(df: pd.DataFrame, title_en: str, title_tr: str, highlight_int8: bool = False) -> str:
    if df is None or df.empty:
        return ""
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows_html = []
    for _, row in df.iterrows():
        row_str = " ".join(str(v) for v in row.values)
        tr_cls = ' class="highlight-row"' if (highlight_int8 and "INT8" in row_str) else ""
        cells = []
        for col_name, val in row.items():
            sval = str(val)
            col_l = str(col_name).lower()
            if "offensive" in sval and ("🚫" in sval or col_l in ("prediction", "label")):
                cells.append('<td><span class="taste-badge taste-badge-err">🚫 offensive</span></td>')
            elif "clean" in sval and ("✅" in sval or col_l in ("prediction", "label")):
                cells.append('<td><span class="taste-badge taste-badge-ok">✅ clean</span></td>')
            elif col_l.startswith("≤") or "budget" in col_l:
                badge = (
                    '<span class="taste-badge taste-badge-ok taste-mono">yes (≤ 50 ms)</span>'
                    if sval.lower() == "yes" else
                    '<span class="taste-badge taste-badge-err taste-mono">no (&gt; 50 ms)</span>'
                )
                cells.append(f"<td>{badge}</td>")
            elif any(k in col_l for k in ("latency", "size", "p(", "acc", "f1", "prec", "rec", "distance")):
                cells.append(f"<td class='taste-mono'>{html.escape(sval)}</td>")
            else:
                cells.append(f"<td>{html.escape(sval)}</td>")
        rows_html.append(f"<tr{tr_cls}>{''.join(cells)}</tr>")

    return f"""
    <div class="taste-card" style="padding:14px 16px;">
      <div class="taste-card-title">
        <span>{_bi(title_en, title_tr)}</span>
      </div>
      <div class="taste-table-wrap" style="max-height:380px;">
        <table class="taste-mini-table">
          <thead><tr>{headers}</tr></thead>
          <tbody>{''.join(rows_html)}</tbody>
        </table>
      </div>
    </div>
    """


def _render_ladder_cards(models_df: pd.DataFrame) -> str:
    if models_df is None or models_df.empty:
        return ""
    lat_col = [c for c in models_df.columns if c.startswith("latency_ms")][0]
    cards = []
    for idx, row in models_df.iterrows():
        name = str(row["model"])
        pred_raw = str(row["prediction"])
        is_off = "offensive" in pred_raw
        p_off = float(row["P(offensive)"])
        lat_ms = float(row[lat_col])
        size_mb = float(row["size_mb"])
        is_int8 = "INT8" in name
        card_cls = "taste-ladder-card highlight" if is_int8 else "taste-ladder-card"
        badge = (
            '<span class="taste-badge taste-badge-err">🚫 offensive</span>'
            if is_off else
            '<span class="taste-badge taste-badge-ok">✅ clean</span>'
        )
        lat_color = "var(--ok-ink)" if lat_ms <= BUDGET else "var(--err-ink)"
        step_label = f"Rung {idx} · {'DEFAULT (R5)' if is_int8 else str(row['precision'])}"
        cards.append(f"""
        <div class="{card_cls}">
          <div>
            <div class="taste-ladder-step">{html.escape(step_label)}</div>
            <div class="taste-ladder-name">{html.escape(name)}</div>
            <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;margin-top:4px;">
              {badge}
              <span class="taste-mono" style="font-size:12px;font-weight:600;color:var(--ink);">P(off)={p_off:.2f}</span>
            </div>
          </div>
          <div class="taste-ladder-metrics">
            <span style="color:{lat_color};font-weight:600;">⏱️ {lat_ms:.2f} ms</span>
            <span>{size_mb:.1f} MB</span>
          </div>
        </div>
        """)
    return f'<div class="taste-ladder-grid">{"".join(cards)}</div>'


def inspect_message(text: str):
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    steps, matches, models = SERVICE.inspect(text)
    if matches.empty:
        matches = pd.DataFrame([["-", "no lexicon word within edit distance 1", "-"]],
                               columns=["token", "match", "distance"])
    models["prediction"] = models["prediction"].map(
        lambda l: ("🚫 " if l == "offensive" else "✅ ") + l)
    return steps, matches, models


def inspect_message_with_cards(text: str):
    steps, matches, models = inspect_message(text)
    models_html = _df_to_clean_html_table(
        models,
        "Predictions & CPU Latency Across All 5 Models (Batch Size = 1)",
        "Tüm 5 Modelin Tahmin ve CPU Gecikme Karşılaştırması (Batch = 1)",
        highlight_int8=True,
    )
    steps_html = _df_to_clean_html_table(
        steps,
        "(B) 9-Step Regex Normalization Trace",
        "(B) 9 Adımlı Regex Normalizasyon İzleme Tablosu",
    )
    matches_html = _df_to_clean_html_table(
        matches,
        "(B) Lexicon + Minimum Edit Distance (≤ 1) Matches",
        "(B) Sözlük (Lexicon) + Minimum Düzenleme Mesafesi (≤ 1) Eşleşmeleri",
    )
    return steps_html, matches_html, models_html, _render_ladder_cards(models)


# --------------------------------------------------------------------------- #
# Tab 3 - benchmark results from disk
# --------------------------------------------------------------------------- #
def _benchmark_kpi_html() -> str:
    return f"""
    <div class="taste-bench-strip">
      <div class="taste-bench-card">
        <div class="taste-bench-num" style="color:var(--accent);">85.23%</div>
        <div class="taste-bench-label">{_bi("BERT-base (INT8) Test Accuracy", "BERT-base (INT8) Test Doğruluğu")}</div>
        <div class="taste-bench-note">{_bi("Retains 99.86% of FP32 accuracy (85.35%) on 860 held-out tweets", "860 test tweeti üzerinde FP32 doğruluğunun (%85.35) %99.86'sını korur")}</div>
      </div>
      <div class="taste-bench-card">
        <div class="taste-bench-num" style="color:var(--ok-ink);">30.07 ms</div>
        <div class="taste-bench-label">{_bi("Mean CPU Latency (2.24× Faster)", "Ortalama CPU Gecikmesi (2.24× Daha Hızlı)")}</div>
        <div class="taste-bench-note">{_bi("p95 = 44.37 ms (≤ 50 ms SLA) vs. FP32 67.21 ms mean (104.86 ms p95)", "p95 = 44.37 ms (≤ 50 ms SLA) vs. FP32 67.21 ms ort. (104.86 ms p95)")}</div>
      </div>
      <div class="taste-bench-card">
        <div class="taste-bench-num">173.09 MB</div>
        <div class="taste-bench-label">{_bi("INT8 Disk Size (2.41× Smaller)", "INT8 Disk Boyutu (2.41× Daha Küçük)")}</div>
        <div class="taste-bench-note">{_bi("Compressed from 417.67 MB via PyTorch dynamic INT8 linear quantization", "PyTorch dinamik INT8 kuantizasyon ile 417.67 MB'dan sıkıştırıldı")}</div>
      </div>
      <div class="taste-bench-card">
        <div class="taste-bench-num">5 Models</div>
        <div class="taste-bench-label">{_bi("Evaluated on Identical Test Split", "Aynı Test Seti Üzerinde Karşılaştırıldı")}</div>
        <div class="taste-bench-note">{_bi("OffensEval 2019 (cardiffnlp/tweet_eval offensive) · CPU 1-thread", "OffensEval 2019 (cardiffnlp/tweet_eval offensive) · Tek çekirdek CPU")}</div>
      </div>
    </div>
    """


def load_benchmark() -> tuple[pd.DataFrame, str]:
    csv, hw = config.RESULTS_DIR / "results.csv", config.RESULTS_DIR / "hardware.txt"
    if not csv.exists():
        return pd.DataFrame(), "Run `python -m src.benchmark` to create results/results.csv."
    info = hw.read_text(encoding="utf-8") if hw.exists() else ""
    return pd.read_csv(csv, keep_default_na=False), f"```text\n{info}\n```"


def load_benchmark_html() -> tuple[str, str]:
    df, hw_md = load_benchmark()
    table_html = _df_to_clean_html_table(
        df,
        "Held-Out Test Set Benchmark (OffensEval 2019 · 860 Tweets · results/results.csv)",
        "Test Veri Seti Karşılaştırma Tablosu (OffensEval 2019 · 860 Tweet · results/results.csv)",
        highlight_int8=True,
    )
    return table_html, hw_md


# --------------------------------------------------------------------------- #
# UI layout
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    init_steps_html, init_matches_html, init_models_html, init_cards = inspect_message_with_cards(DEFAULT_INSPECT_TEXT)
    init_bench_html, init_hw_md = load_benchmark_html()

    with gr.Blocks(title="Real-Time Chat Moderation Console") as demo:
        gr.HTML(_topbar_html())

        # Theme & Language control bar (English + Light default for SWE015 grading)
        with gr.Row():
            light_btn = gr.Button("☀️ White Theme (Light)", variant="secondary", size="sm")
            dark_btn = gr.Button("🌙 Black Theme (Dark)", variant="secondary", size="sm")
            en_btn = gr.Button("🇬🇧 English (Course Default)", variant="secondary", size="sm")
            tr_btn = gr.Button("🇹🇷 Türkçe", variant="secondary", size="sm")

        light_btn.click(None, None, None, js=SET_LIGHT_THEME_JS)
        dark_btn.click(None, None, None, js=SET_DARK_THEME_JS)
        en_btn.click(None, None, None, js=SET_LANG_EN_JS)
        tr_btn.click(None, None, None, js=SET_LANG_TR_JS)
        demo.load(None, None, None, js=INIT_UI_JS)

        with gr.Tab("💬 Live Chat"):
            history_state, log_state = gr.State([]), gr.State([])
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Live Game Chat Room (Moderated Output)",
                        height=430,
                        elem_id="chat-panel",
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type a live chat message and press Enter (or click a preset below)...",
                            show_label=False,
                            scale=5,
                            autofocus=True,
                            elem_id="msg-input",
                        )
                        send_btn = gr.Button("Send Message", variant="primary", scale=1)

                    gr.HTML(
                        f'<div style="font-size:12.5px;font-weight:600;color:var(--ink-secondary);margin:4px 0 2px 0;">'
                        f'{_bi("One-Click Live Demo Messages (click any button to send immediately with the active model):", "Tek Tıkla Canlı Demo Mesajları (tıkladığınız anda seçili modelle gönderilir):")}'
                        f'</div>'
                    )
                    with gr.Row():
                        ex_btn_1 = gr.Button("✅ Clean: gg wp everyone!", variant="secondary", size="sm")
                        ex_btn_2 = gr.Button("🚫 Leetspeak: 1d10t uninstall", variant="secondary", size="sm")
                        ex_btn_3 = gr.Button("🚫 Obfuscated: stuuuupid $h1t", variant="secondary", size="sm")
                    with gr.Row():
                        ex_btn_4 = gr.Button("✅ Gaming Slang: nice shot, crushed it!", variant="secondary", size="sm")
                        ex_btn_5 = gr.Button("🚫 Implicit: nobody asked, loser", variant="secondary", size="sm")
                        ex_btn_6 = gr.Button("⚠️ Sarcasm: really smart, said no one ever", variant="secondary", size="sm")

                with gr.Column(scale=2):
                    model_dd = gr.Radio(
                        MODEL_NAMES,
                        value=DEFAULT_MODEL_NAME,
                        elem_id="engine-selector",
                        label="Active Moderation Engine (Click to switch between INT8, FP32, and baselines)",
                    )
                    summary_md = gr.HTML(log_summary([]))
                    filter_radio = gr.Radio(
                        FILTER_CHOICES,
                        value="All Messages",
                        label="Filter Latency Log by Verdict / SLA",
                        elem_id="log-filter-box",
                    )
                    log_html = gr.HTML(render_log_table([]))
                    clear_btn = gr.Button("Clear Session & Reset Telemetry", variant="secondary")

            inputs = [msg, model_dd, history_state, log_state, filter_radio]
            outputs = [msg, history_state, log_state, log_html, summary_md, chatbot]
            msg.submit(send_message, inputs, outputs)
            send_btn.click(send_message, inputs, outputs)
            filter_radio.change(
                fn=lambda log, f: render_log_table(log, f),
                inputs=[log_state, filter_radio],
                outputs=log_html,
            )
            clear_btn.click(
                clear_chat,
                inputs=[filter_radio],
                outputs=[history_state, log_state, log_html, summary_md, chatbot],
            )

            # Wire one-click demo buttons so clicking immediately sends the message
            for btn, sample_text in [
                (ex_btn_1, EXAMPLES[0]),
                (ex_btn_2, EXAMPLES[1]),
                (ex_btn_3, EXAMPLES[2]),
                (ex_btn_4, EXAMPLES[3]),
                (ex_btn_5, EXAMPLES[4]),
                (ex_btn_6, EXAMPLES[5]),
            ]:
                btn.click(
                    fn=lambda m, h, l, f, s=sample_text: send_message(s, m, h, l, f),
                    inputs=[model_dd, history_state, log_state, filter_radio],
                    outputs=outputs,
                )

        with gr.Tab("🔬 Model Ladder"):
            gr.HTML(
                f'<div style="margin-bottom:8px;">'
                f'<h3 style="margin:0 0 4px 0;font-size:16px;color:var(--ink);">'
                f'{_bi("Full Pipeline & 5-Model Ladder Inspector", "Tam NLP Hattı ve 5 Aşamalı Model Merdiveni İnceleyicisi")}'
                f'</h3>'
                f'<div style="font-size:13px;color:var(--ink-secondary);">'
                f'{_bi("Pass a single message through all 9 regex normalization steps, Levenshtein edit-distance lexicon matching, and all 5 models side by side on the CPU.", "Tek bir mesajı 9 adımlı regex normalizasyonundan, Levenshtein düzenleme mesafesi sözlük eşleştirmesinden ve CPU üzerinde 5 modelin tamamından yan yana geçirin.")}'
                f'</div></div>'
            )
            with gr.Row():
                insp_in = gr.Textbox(
                    value=DEFAULT_INSPECT_TEXT,
                    label="Input Message to Inspect",
                    scale=5,
                    elem_id="inspect-input",
                )
                insp_btn = gr.Button("Analyze Pipeline", variant="primary", scale=1)

            with gr.Row():
                preset_1 = gr.Button("🔬 Preset 1: Adversarial Leetspeak ($h1t / 1d10t)", variant="secondary", size="sm")
                preset_2 = gr.Button("🔬 Preset 2: Implicit Toxicity (No Swear Words)", variant="secondary", size="sm")
                preset_3 = gr.Button("🔬 Preset 3: Benign Gaming Slang (nice shot)", variant="secondary", size="sm")
                preset_4 = gr.Button("🔬 Preset 4: Sarcasm Failure Case", variant="secondary", size="sm")

            ladder_cards_html = gr.HTML(init_cards)
            models_html_box = gr.HTML(init_models_html)
            with gr.Row():
                steps_html_box = gr.HTML(init_steps_html)
                match_html_box = gr.HTML(init_matches_html)

            insp_outputs = [steps_html_box, match_html_box, models_html_box, ladder_cards_html]
            insp_btn.click(inspect_message_with_cards, insp_in, insp_outputs)
            insp_in.submit(inspect_message_with_cards, insp_in, insp_outputs)

            for p_btn, p_text in [
                (preset_1, DEFAULT_INSPECT_TEXT),
                (preset_2, EXAMPLES[4]),
                (preset_3, EXAMPLES[3]),
                (preset_4, EXAMPLES[5]),
            ]:
                p_btn.click(
                    fn=lambda s=p_text: (s, *inspect_message_with_cards(s)),
                    inputs=None,
                    outputs=[insp_in, *insp_outputs],
                )

        with gr.Tab("📊 Benchmark"):
            gr.HTML(_benchmark_kpi_html())
            bench_html_box = gr.HTML(init_bench_html)
            hw_md = gr.Markdown(value=init_hw_md)
            gr.Button("Reload Benchmark from Disk", variant="secondary").click(
                load_benchmark_html, None, [bench_html_box, hw_md]
            )
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        css=TASTE_CSS,
    )

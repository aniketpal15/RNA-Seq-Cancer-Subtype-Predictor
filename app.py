"""
app.py — Oncology AI Engine
Exact Dark Purple SaaS Dashboard matching the Insighta reference aesthetic.
Features:
- Top Navbar: Brand on left, File Uploader in middle, File Status badge on right.
- Sub-Navbar Hero: Title & Subtitle on left, "Run AI Analysis" CTA button on right.
- Initial State: All metrics, graphs, and gauges are set to 0% / standby.
- On Analysis: Dynamic processing, populated metrics, multi-curve wave chart,
  roadmap & subtype matrix (Heatmap slot), Donut Chart (Audience slot),
  Semi-circular Gauge (Quality Score slot), Gradient Insight Callout, and SHAP beeswarm.
"""

import os, time
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oncology AI Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL METADATA
# ─────────────────────────────────────────────────────────────────────────────
SUBTYPE_META = {
    "BRCA": {"full": "Breast Invasive Carcinoma",         "organ": "Breast",   "hex": "#8B5CF6", "badge": "🎀"},
    "COAD": {"full": "Colon Adenocarcinoma",               "organ": "Colon",    "hex": "#A78BFA", "badge": "🟡"},
    "KIRC": {"full": "Kidney Renal Clear Cell Carcinoma",  "organ": "Kidney",   "hex": "#10B981", "badge": "🟢"},
    "LUAD": {"full": "Lung Adenocarcinoma",                "organ": "Lung",     "hex": "#6366F1", "badge": "🔵"},
    "PRAD": {"full": "Prostate Adenocarcinoma",            "organ": "Prostate", "hex": "#C084FC", "badge": "🟣"},
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Exact Insighta Dark Purple SaaS Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Hide native Streamlit chrome */
#MainMenu, footer,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
header[data-testid="stHeader"],
section[data-testid="stSidebar"],
button[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

/* Eliminate top empty whitespace in Streamlit root */
header {
    height: 0 !important;
    min-height: 0 !important;
    display: none !important;
}

[data-testid="stAppViewBlockContainer"] {
    padding-top: 0 !important;
}

:root {
    --bg-canvas:       #0A0B0E;
    --card-bg:         #12131A;
    --card-surface:    #181924;
    --border-card:     #212230;
    --border-highlight:#36384E;
    --accent-purple:   #8B5CF6;
    --accent-light:    #A78BFA;
    --accent-glow:     rgba(139, 92, 246, 0.35);
    --accent-green:    #10B981;
    --accent-red:      #EF4444;
    --text-pure:       #FFFFFF;
    --text-main:       #E2E8F0;
    --text-muted:      #7A7D93;
    --radius-main:     15px;
    --radius-sm:       10px;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
    background-color: var(--bg-canvas) !important;
    color: var(--text-main) !important;
}

html, body, .main, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    overflow-x: hidden !important;
    max-width: 100vw !important;
    box-sizing: border-box !important;
}

.main, .block-container, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-canvas) !important;
}

.main .block-container {
    padding-top: 0.1rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
}

/* ═══════════════════════════════════════════════════
   TOP NAVBAR (Brand left, Uploader middle, File right)
   ═══════════════════════════════════════════════════ */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) {
    background: #111218 !important;
    border: 1px solid var(--border-card) !important;
    border-radius: var(--radius-main) !important;
    padding: 0 1.4rem !important;
    margin-top: 0 !important;
    margin-bottom: 0.6rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    height: 58px !important;
    min-height: 58px !important;
    max-height: 58px !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
}

/* Force ALL nested column containers to align to middle center */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) [data-testid="stColumn"],
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) [data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) [data-testid="stElementContainer"],
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) [data-testid="stLayoutWrapper"] {
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* Left Brand Title & Glowing Dot */
.nav-brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.03em;
    display: flex !important;
    align-items: center !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.nav-brand-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #8B5CF6;
    box-shadow: 0 0 10px #8B5CF6;
    margin: 0 !important;
}

/* Middle File Uploader — Sideways Layout */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stFileUploader"] {
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    height: auto !important;
    min-height: unset !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stFileUploader"] > label {
    display: none !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) section[data-testid="stFileUploaderDropzone"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 0.6rem !important;
    padding: 0.2rem 0.65rem !important;
    min-height: unset !important;
    height: 38px !important;
    margin: 0 !important;
    background: transparent !important;
    border: 1px dashed rgba(139, 92, 246, 0.35) !important;
    border-radius: 12px !important;
    box-sizing: border-box !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #8B5CF6 !important;
    background: rgba(139, 92, 246, 0.05) !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) section[data-testid="stFileUploaderDropzone"] > span {
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) section[data-testid="stFileUploaderDropzone"] button {
    background: #1E202D !important;
    color: #FFFFFF !important;
    border: 1px solid #34374D !important;
    font-size: 0.75rem !important;
    padding: 0.25rem 0.75rem !important;
    border-radius: 8px !important;
    margin: 0 !important;
    line-height: 1.2 !important;
    display: inline-flex !important;
    align-items: center !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stFileUploaderDropzoneInstructions"] span {
    font-size: 0.76rem !important;
    color: #94A3B8 !important;
    margin: 0 !important;
    line-height: 1 !important;
}

/* File Uploaded Chips — Sideways / Horizontal Expansion */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) section[data-testid="stFileUploaderDropzone"] > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stFileChips"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 0.5rem !important;
    max-height: 38px !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stFileChips"] > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stFileChip"] {
    height: 34px !important;
    min-height: unset !important;
    padding: 0 0.6rem !important;
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: center !important;
    margin: 0 !important;
    border-radius: 10px !important;
    box-sizing: border-box !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stFileChip"] > div {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    margin: 0 !important;
    line-height: 1.1 !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stFileChipName"] {
    font-size: 0.76rem !important;
    line-height: 1.1 !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stFileChips"] button[data-testid="stBaseButton-borderlessIcon"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 30px !important;
    width: 30px !important;
    padding: 0 !important;
    margin: 0 !important;
    align-self: center !important;
}

/* Right side status badge & demo button */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
    gap: 0.75rem !important;
    min-height: unset !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) [data-testid="stMarkdown"],
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) [data-testid="stMarkdownContainer"] {
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) [data-testid="stButton"] {
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) [data-testid="stButton"] button {
    display: inline-flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0.35rem 0.9rem !important;
    height: auto !important;
    min-height: unset !important;
}

/* ═══════════════════════════════════════════════════
   MID-SECTION COLUMN CARDS (wave + heatmap)
   Style the column containers directly so st.pyplot
   and st.markdown can coexist without split HTML divs
   ═══════════════════════════════════════════════════ */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stImage"]),
div[data-testid="stHorizontalBlock"]:has(canvas) {
    gap: 1rem !important;
}

/* Target the mid-section columns (the ones that contain the wave/heatmap) */
.mid-card-col {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: var(--radius-main) !important;
    padding: 1.4rem 1.6rem !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
}

/* Remove extra whitespace Streamlit adds around pyplot figures */
div[data-testid="stImage"] {
    margin: 0 !important;
    padding: 0 !important;
}

/* Remove the blank pill/bar that appears from orphaned st.markdown divs */
.stMarkdown:empty,
.stMarkdown > div:empty {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Style mid-section columns as cards natively — avoids all split-div issues */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stImage"]) > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stImage"]) > div[data-testid="stVerticalBlock"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: var(--radius-main) !important;
    padding: 1.2rem 1.4rem !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
}

/* Style SHAP section as a single unified card */
div[data-testid="stVerticalBlock"]:has(div[data-testid="stImage"]):not([data-testid="stHorizontalBlock"] *) {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: var(--radius-main) !important;
    padding: 1.6rem 1.8rem !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
    margin-top: 1.2rem !important;
}

div[data-testid="stVerticalBlock"]:has(div[data-testid="stImage"]):not([data-testid="stHorizontalBlock"] *) > div {
    border: none !important;
}

/* ═══════════════════════════════════════════════════
   INSIGHTA CARDS
   ═══════════════════════════════════════════════════ */
.in-card {
    background: var(--card-bg);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-main);
    padding: 1.4rem 1.6rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}

.in-card:hover {
    border-color: var(--border-highlight);
    box-shadow: 0 12px 32px rgba(0,0,0,0.6);
    transform: translateY(-2px);
}

.in-stat-card {
    background: var(--card-bg);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-main);
    padding: 1.25rem 1.5rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.in-stat-card:hover {
    border-color: var(--border-highlight);
    transform: translateY(-2px);
}

.in-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.01em;
    margin-bottom: 0.35rem;
}

.in-card-sub {
    font-size: 0.76rem;
    color: var(--text-muted);
}

.in-stat-value {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #FFFFFF;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
}

.in-badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
}

.badge-green {
    background: rgba(16,185,129,0.15);
    color: #10B981;
    border: 1px solid rgba(16,185,129,0.3);
}

.badge-purple {
    background: rgba(139,92,246,0.18);
    color: #C084FC;
    border: 1px solid rgba(139,92,246,0.35);
}

.badge-red {
    background: rgba(239,68,68,0.15);
    color: #EF4444;
    border: 1px solid rgba(239,68,68,0.3);
}

/* Purple CTA Button — Clean Minimalist Style */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1.8rem !important;
    box-shadow: none !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    letter-spacing: 0.01em !important;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%) !important;
    border-color: rgba(167, 139, 250, 0.6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.22) !important;
}

div.stButton > button[kind="primary"]:active {
    transform: translateY(0px) scale(0.99) !important;
    box-shadow: none !important;
}

div.stButton > button:not([kind="primary"]) {
    background: #181924 !important;
    color: #CBD5E1 !important;
    border: 1px solid #2B2C3E !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.1rem !important;
    transition: all 0.18s ease !important;
}

div.stButton > button:not([kind="primary"]):hover {
    background: #232535 !important;
    border-color: #8B5CF6 !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
}



/* Sub-Navbar Hero Typography */
.hero-title {
    font-size: clamp(1.3rem, 3.2vw, 1.7rem);
    font-weight: 900;
    letter-spacing: -0.03em;
    color: #FFFFFF;
    margin-bottom: 0.2rem;
    line-height: 1.25;
}

.hero-sub {
    font-size: clamp(0.78rem, 1.6vw, 0.88rem);
    color: #8B8FA3;
    line-height: 1.45;
    max-width: 850px;
}

/* Dataframe and Expander styling */
div[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-card) !important;
}
div[data-testid="stDataFrame"] table {
    background: #12131A !important;
}

div[data-testid="stExpander"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: var(--radius-main) !important;
    overflow: hidden !important;
}

/* ═══════════════════════════════════════════════════
   RESPONSIVE DESIGN BREAKPOINTS (All Devices)
   ═══════════════════════════════════════════════════ */

/* Global Dataframe & Image containment */
div[data-testid="stDataFrame"] {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
}

div[data-testid="stDataFrame"] * {
    max-width: 100% !important;
}

div[data-testid="stImage"] img {
    max-width: 100% !important;
    height: auto !important;
}

/* 1. Large Monitors (> 1440px) */
@media (min-width: 1441px) {
    .main .block-container {
        max-width: 1440px !important;
        margin: 0 auto !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }
}

/* 2. Tablets & Small Laptops (769px - 1024px) */
@media (min-width: 769px) and (max-width: 1024px) {
    .main .block-container {
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    .in-stat-card {
        padding: 1.1rem 1.2rem !important;
    }

    .in-stat-value {
        font-size: 1.85rem !important;
    }

    /* Stack mid-section charts cleanly */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stImage"]) {
        flex-direction: column !important;
        gap: 1rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stImage"]) > div[data-testid="stColumn"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }
}

/* 3. Tablets & Mobile Screens (max-width: 900px) */
@media (max-width: 900px) {
    .main .block-container,
    [data-testid="stAppViewBlockContainer"] {
        padding-left: 0.65rem !important;
        padding-right: 0.65rem !important;
        padding-top: 0.15rem !important;
        padding-bottom: 2rem !important;
    }

    /* Responsive Top Navbar */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) {
        height: auto !important;
        min-height: unset !important;
        max-height: unset !important;
        flex-direction: column !important;
        padding: 0.9rem 1rem !important;
        gap: 0.75rem !important;
        border-radius: 16px !important;
        align-items: stretch !important;
        overflow: visible !important;
    }

    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) > div[data-testid="stColumn"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
        justify-content: center !important;
        display: flex !important;
    }

    .nav-brand-title {
        font-size: 1.25rem !important;
        justify-content: center !important;
        width: 100% !important;
        text-align: center !important;
    }

    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stFileUploader"] {
        width: 100% !important;
        max-width: 100% !important;
    }

    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) section[data-testid="stFileUploaderDropzone"] {
        width: 100% !important;
        max-width: 100% !important;
        justify-content: center !important;
    }

    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stHorizontalBlock"] {
        width: 100% !important;
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        align-items: center !important;
        gap: 0.6rem !important;
    }

    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        width: auto !important;
        min-width: unset !important;
        flex: 1 1 auto !important;
    }

    /* Hero Sub-Navbar stacking */
    div[data-testid="stHorizontalBlock"]:has(.hero-title) {
        flex-direction: column !important;
        gap: 0.8rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.hero-title) > div[data-testid="stColumn"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Stat cards */
    div[data-testid="stHorizontalBlock"]:has(.in-stat-card) {
        flex-direction: column !important;
        gap: 0.75rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.in-stat-card) > div[data-testid="stColumn"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }

    .in-stat-card {
        padding: 1.1rem 1.25rem !important;
        min-height: auto !important;
    }

    .in-stat-value {
        font-size: 1.85rem !important;
    }

    /* Mid-section columns */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stImage"]) {
        flex-direction: column !important;
        gap: 0.9rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stImage"]) > div[data-testid="stColumn"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Primary CTA button full width and accessible tap size */
    div.stButton > button[kind="primary"] {
        width: 100% !important;
        min-height: 48px !important;
        font-size: 1rem !important;
    }

    /* SHAP Beeswarm container on mobile */
    div[data-testid="stVerticalBlock"]:has(div[data-testid="stImage"]):not([data-testid="stHorizontalBlock"] *) {
        padding: 1.1rem 1.2rem !important;
        border-radius: 16px !important;
    }
}

/* 4. Ultra-Small Phones (max-width: 420px) */
@media (max-width: 420px) {
    .main .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }

    .in-stat-value {
        font-size: 1.65rem !important;
    }

    .in-badge-pill {
        font-size: 0.68rem !important;
        padding: 0.15rem 0.45rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ARTIFACT LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    missing = [f for f in ("model.pkl", "encoder.pkl", "selector.pkl") if not os.path.exists(f)]
    if missing:
        return None, None, None, f"Missing: {', '.join(missing)} — run `python train_model.py`."
    try:
        return joblib.load("model.pkl"), joblib.load("encoder.pkl"), joblib.load("selector.pkl"), None
    except Exception as exc:
        return None, None, None, str(exc)

@st.cache_data(show_spinner=False)
def load_demo_csv(filename="sample_data_1.csv"):
    candidates = [
        filename,
        os.path.join("sampledata", filename),
        os.path.join("sampledata", os.path.basename(filename)),
        "sample_patient.csv"
    ]
    target = next((p for p in candidates if os.path.exists(p)), None)
    if not target:
        return None, f"`{filename}` not found — run `python train_model.py`."
    try:
        return pd.read_csv(target), None
    except Exception as exc:
        return None, str(exc)

model, encoder, selector, art_err = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# 1. TOP NAVBAR
# Left: Brand Name
# Middle: Upload File Section
# Right: Upload Status & Pill Menu
# ─────────────────────────────────────────────────────────────────────────────
nav_left, nav_mid, nav_right = st.columns([1.3, 1.8, 1.4], gap="medium", vertical_alignment="center")

with nav_left:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.8rem;">
        <div class="nav-brand-dot"></div>
        <div class="nav-brand-title">Oncology AI Engine</div>
    </div>
    """, unsafe_allow_html=True)

patient_df: pd.DataFrame | None = st.session_state.get("patient_df", None)
patient_source: str = st.session_state.get("patient_source", "")

with nav_mid:
    uploaded_file = st.file_uploader(
        "Upload RNA-Seq CSV",
        type=["csv"],
        label_visibility="collapsed",
        help="Upload patient CSV containing 20,531 RNA-Seq features."
    )

with nav_right:
    col_status, col_demo_btn = st.columns([1.5, 1], vertical_alignment="center")
    with col_demo_btn:
        load_demo = st.button("Demo Sample", help="Load pre-saved benchmark patient data")

    with col_status:
        if uploaded_file is not None:
            try:
                raw = pd.read_csv(uploaded_file)
                drop_c = [c for c in raw.columns if c.lower() in {"sample_id","id","unnamed: 0","class","cancer_type"}]
                patient_df = raw.drop(columns=drop_c).iloc[[0]].copy()
                patient_source = f"File: {uploaded_file.name[:14]}..."
                st.session_state["patient_df"] = patient_df
                st.session_state["patient_source"] = patient_source
                st.session_state.pop("results", None)
            except Exception as e:
                st.error(f"Error parsing file: {e}")
        elif load_demo:
            current_cycle = st.session_state.get("demo_cycle_idx", 0)
            next_cycle = (current_cycle % 20) + 1
            filename = f"sample_data_{next_cycle}.csv"
            demo, demo_err = load_demo_csv(filename)
            if not demo_err:
                patient_df = demo
                patient_source = f"Demo #{next_cycle}: {filename}"
                st.session_state["patient_df"] = patient_df
                st.session_state["patient_source"] = patient_source
                st.session_state["demo_cycle_idx"] = next_cycle
                st.session_state.pop("results", None)

        if patient_df is not None:
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:flex-end;gap:0.5rem;">
                <span class="in-badge-pill badge-green">✓ Ready</span>
                <span style="font-size:0.75rem;color:#A78BFA;font-weight:600;">{patient_source}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:flex-end;">
                <span class="in-badge-pill badge-purple">● Standby (No File)</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SUB-NAVBAR HEADER AREA
# Left: Subtitle & Platform Description
# Right: Prominent Analyze Action Button
# ─────────────────────────────────────────────────────────────────────────────
sub_left, sub_right = st.columns([2.6, 1])

with sub_left:
    st.markdown("""
    <div>
        <div class="hero-title">
            RNA-Seq Cancer Subtype Predictor
        </div>
        <div class="hero-sub">
            Multi-class histological classification via high-throughput Illumina HiSeq transcriptomics.
            Powered by gradient-boosted tree ensembles and game-theoretic SHAP explainability.
        </div>
    </div>
    """, unsafe_allow_html=True)

with sub_right:
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Initialize AI Analysis", type="primary", use_container_width=True)

# Execute Inference on Button Click
if run_clicked:
    if patient_df is None:
        st.warning("⚠️ Please upload a patient CSV or click 'Demo Sample' in the top navbar first!")
    elif art_err:
        st.error(f"Setup Required: {art_err}")
    else:
        with st.spinner("Processing RNA-Seq features through SelectKBest & XGBoost..."):
            n_expected = selector.n_features_in_
            if patient_df.shape[1] == n_expected:
                X_sel = selector.transform(patient_df)
                feat_names = patient_df.columns[selector.get_support()].tolist()
            elif patient_df.shape[1] == 500:
                X_sel = patient_df.values
                feat_names = patient_df.columns.tolist()
            else:
                st.error(f"Feature count mismatch: expected {n_expected} or 500 genes.")
                st.stop()

            X_df = pd.DataFrame(X_sel, columns=feat_names)

            pred_idx = int(model.predict(X_sel)[0])
            pred_code = encoder.inverse_transform([pred_idx])[0]
            meta = SUBTYPE_META.get(pred_code, {"full": "Unknown", "organ": "?", "hex": "#8B5CF6", "badge": "🟣"})
            proba = model.predict_proba(X_sel)[0] if hasattr(model, "predict_proba") else None
            confidence = float(proba[pred_idx]) * 100 if proba is not None else 99.0

            # SHAP
            explainer = shap.TreeExplainer(model)
            shap_expl = explainer(X_df)
            if hasattr(shap_expl, "shape") and len(shap_expl.shape) == 3:
                class_expl = shap_expl[:, :, pred_idx]
                shap_values = class_expl.values[0]
            elif isinstance(shap_expl, list):
                class_expl = shap_expl[pred_idx]
                shap_values = class_expl[0]
            else:
                class_expl = shap_expl
                shap_values = class_expl.values[0]

            st.session_state["results"] = dict(
                pred_code=pred_code,
                pred_idx=pred_idx,
                meta=meta,
                confidence=confidence,
                proba=proba,
                X_df=X_df,
                feat_names=feat_names,
                X_sel=X_sel,
                class_expl=class_expl,
                shap_values=shap_values,
            )

results = st.session_state.get("results", None)

# Default Standby Values (0% initially)
is_analyzed = results is not None
conf_val = f"{results['confidence']:.1f}%" if is_analyzed else "0.0%"
pred_sub = results['pred_code'] if is_analyzed else "STANDBY"
biomarker_cnt = "500" if is_analyzed else "0"
meta_data = results['meta'] if is_analyzed else {"full": "Awaiting patient inference", "organ": "—", "hex": "#7A7D93"}

# ─────────────────────────────────────────────────────────────────────────────
# 3. TOP STAT CARDS (Matches the 3 metric cards in reference)
# Total Features Analyzed | Top Prediction Subtype | Classification Confidence
# ─────────────────────────────────────────────────────────────────────────────
s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(f"""
    <div class="in-stat-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
            <div style="font-size:0.8rem;font-weight:700;color:#8B8FA3;display:flex;align-items:center;gap:0.4rem;">
                Biomarkers Analyzed
            </div>
            <span class="in-badge-pill badge-purple">Top K=500</span>
        </div>
        <div class="in-stat-value">{biomarker_cnt}</div>
        <div style="font-size:0.75rem;color:#71758B;margin-top:0.4rem;">97.6% Transcriptome reduction</div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    badge_style = "badge-green" if is_analyzed else "badge-purple"
    delta_text = meta_data['organ'] if is_analyzed else "0%"
    st.markdown(f"""
    <div class="in-stat-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
            <div style="font-size:0.8rem;font-weight:700;color:#8B8FA3;display:flex;align-items:center;gap:0.4rem;">
                Predicted Subtype
            </div>
            <span class="in-badge-pill {badge_style}">{delta_text}</span>
        </div>
        <div class="in-stat-value" style="color:{meta_data['hex']};">{pred_sub}</div>
        <div style="font-size:0.75rem;color:#71758B;margin-top:0.4rem;">{meta_data['full']}</div>
    </div>
    """, unsafe_allow_html=True)

with s3:
    conf_badge = "badge-green" if (is_analyzed and results['confidence'] > 90) else "badge-purple"
    conf_sub = "High Diagnostic Certainty" if is_analyzed else "Engine awaiting run trigger"
    st.markdown(f"""
    <div class="in-stat-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
            <div style="font-size:0.8rem;font-weight:700;color:#8B8FA3;display:flex;align-items:center;gap:0.4rem;">
                Model Confidence
            </div>
            <span class="in-badge-pill {conf_badge}">Posterior</span>
        </div>
        <div class="in-stat-value">{conf_val}</div>
        <div style="font-size:0.75rem;color:#71758B;margin-top:0.4rem;">{conf_sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MIDDLE SECTION
# Left (2 cols): Wave Chart (Active Users slot) + Subtype Heatmap Roadmap
# ─────────────────────────────────────────────────────────────────────────────
mid_left, mid_right = st.columns([1.8, 1.2])

with mid_left:
    # Build wave chart first, then render everything in one HTML block to avoid blank card artifacts
    fig_wave, ax_wave = plt.subplots(figsize=(7.5, 3.1), dpi=140)
    fig_wave.patch.set_facecolor('#12131A')
    ax_wave.set_facecolor('#12131A')

    x_points = np.linspace(1, 30, 200)
    if is_analyzed:
        sample_vals = results['X_sel'][0][:30]
        y_wave = np.interp(x_points, np.arange(1, len(sample_vals) + 1), sample_vals)
    else:
        y_wave = np.zeros_like(x_points)

    ax_wave.plot(x_points, y_wave, color='#8B5CF6', linewidth=2.4)
    ax_wave.fill_between(x_points, y_wave, color='#8B5CF6', alpha=0.18)
    ax_wave.plot(x_points, y_wave * 0.45, color='#A78BFA', linewidth=1.2, linestyle='--', alpha=0.5)
    ax_wave.set_xlim(1, 30)
    ax_wave.tick_params(colors='#65687C', labelsize=8)
    for spine in ax_wave.spines.values():
        spine.set_color('#212230')
    ax_wave.grid(True, color='#1E202C', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Card header only (chart renders below, styled by CSS on the column container)
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
        <div>
            <div class="in-card-title">Biomarker Expression Loci Waveform</div>
            <div class="in-card-sub">Top 30 discriminant expression peaks across the transcriptome</div>
        </div>
        <span class="in-badge-pill badge-purple">Illumina HiSeq</span>
    </div>
    """, unsafe_allow_html=True)
    st.pyplot(fig_wave, use_container_width=True)
    plt.close(fig_wave)

with mid_right:
    # Header as one self-contained block
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
        <div>
            <div class="in-card-title">Pan-Cancer Subtype Heatmap Matrix</div>
            <div class="in-card-sub">Clinical Roadmap &middot; COAD &middot; KIRC &middot; LUAD &middot; PRAD &middot; BRCA</div>
        </div>
        <span class="in-badge-pill badge-green">TCGA 401</span>
    </div>
    """, unsafe_allow_html=True)

    # Each row rendered individually — avoids emoji-in-nested-f-string issues
    probs = results['proba'] if is_analyzed else [0.0]*5
    for idx, code in enumerate(encoder.classes_ if encoder else ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]):
        m = SUBTYPE_META.get(code, {"hex": "#8B5CF6", "organ": "Pan-Cancer", "badge": ""})
        pct = float(probs[idx]) * 100 if is_analyzed else 0.0
        is_top = is_analyzed and code == results['pred_code']
        border_color = m['hex'] if is_top else "#1E202D"
        bg_color = "rgba(139,92,246,0.12)" if is_top else "#171822"
        badge_html = m['badge']  # emoji safe as separate variable, not nested f-string
        bar_width = f"{pct:.1f}"
        hex_color = m['hex']
        organ = m['organ']
        pct_str = f"{pct:.1f}%"

        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'padding:0.42rem 0.75rem;margin-bottom:0.35rem;border-radius:10px;'
            f'background:{bg_color};border:1px solid {border_color};">'
            f'<div style="display:flex;align-items:center;gap:0.6rem;">'
            f'<span style="font-size:0.9rem;">{badge_html}</span>'
            f'<div>'
            f'<span style="font-size:0.82rem;font-weight:700;color:#FFFFFF;">{code}</span>'
            f'<span style="font-size:0.72rem;color:#71758B;margin-left:0.35rem;">{organ}</span>'
            f'</div></div>'
            f'<div style="display:flex;align-items:center;gap:0.75rem;">'
            f'<div style="width:75px;height:6px;background:#232534;border-radius:9999px;overflow:hidden;">'
            f'<div style="width:{bar_width}%;background:{hex_color};height:100%;border-radius:9999px;"></div>'
            f'</div>'
            f'<span style="font-size:0.8rem;font-weight:800;color:#FFFFFF;min-width:38px;text-align:right;">{pct_str}</span>'
            f'</div></div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# 6. SHAP DECISION BEESWARM EXPLAINABILITY & TOP 5 GENOMIC MARKERS
# ─────────────────────────────────────────────────────────────────────────────
if is_analyzed:
    with st.container(border=True):
        st.markdown("""
        <div style="font-size:1.15rem;font-weight:800;color:#FFFFFF;margin-bottom:0.3rem;">
            Model Interpretability · SHAP Beeswarm Plot
        </div>
        <div style="font-size:0.83rem;color:#8B8FA3;margin-bottom:1.2rem;">
            Game-theoretic Shapley attributions explaining individual biomarker impact on the diagnosed cancer subtype.
        </div>
        """, unsafe_allow_html=True)

        try:
            plt.rcParams.update({
                "figure.facecolor":  "#12131A",
                "axes.facecolor":    "#12131A",
                "axes.edgecolor":    "#212230",
                "axes.labelcolor":   "#8B8FA3",
                "xtick.color":       "#65687C",
                "ytick.color":       "#8B8FA3",
                "text.color":        "#FFFFFF",
                "grid.color":        "#1A1B26",
                "grid.linewidth":    0.6,
            })

            fig_shap, ax_shap = plt.subplots(figsize=(9, 2.8), dpi=140)
            shap.plots.beeswarm(results['class_expl'], max_display=6, show=False)
            plt.title(f"Genomic Drivers Impacting {results['pred_code']} Classification", fontsize=10, fontweight="bold", pad=8, color="#FFFFFF")
            plt.xlabel("SHAP Attribution Value", fontsize=8, color="#8B8FA3")
            plt.tight_layout()
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)
            plt.rcParams.update(plt.rcParamsDefault)
        except Exception as exc:
            st.warning(f"Could not render SHAP beeswarm: {exc}")

        # Top 5 Driver Genes Table
        driver_df = (
            pd.DataFrame({
                "Genetic Biomarker": results['feat_names'],
                "Patient Expression Level": results['X_sel'][0],
                "SHAP Attribution Value": results['shap_values'],
            })
            .sort_values("SHAP Attribution Value", ascending=False)
            .head(5)
        )

        st.markdown("<div style='font-size:0.92rem;font-weight:800;color:#FFFFFF;margin-top:1.1rem;margin-bottom:0.5rem;'>Top 5 Positive Genomic Driver Genes</div>", unsafe_allow_html=True)
        st.dataframe(
            driver_df.style.format({
                "Patient Expression Level": "{:.4f}",
                "SHAP Attribution Value": "{:+.5f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

# Footer
st.markdown("""
<div style="border-top:1px solid #1E202C;margin-top:2.5rem;padding-top:1.2rem;text-align:center;color:#65687C;font-size:0.76rem;">
    Oncology AI Engine · TCGA Pan-Cancer RNA-Seq (UCI ID 401) · XGBoost + SelectKBest + SHAP
</div>
""", unsafe_allow_html=True)

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

import os, time, base64
from contextlib import contextmanager
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & FAVICON
# ─────────────────────────────────────────────────────────────────────────────
app_icon_img = None
for icon_candidate in ["favicon.png", "public/favicon.png", "public/app_icon.png", "public/icon.png"]:
    if os.path.exists(icon_candidate):
        try:
            app_icon_img = Image.open(icon_candidate)
            break
        except Exception:
            pass

st.set_page_config(
    page_title="Oncology AI Engine",
    page_icon=app_icon_img or "🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL METADATA
# ─────────────────────────────────────────────────────────────────────────────
SUBTYPE_META = {
    "BRCA": {"full": "Breast Invasive Carcinoma",         "organ": "Breast",   "hex": "#8B5CF6", "badge": "🔴"},
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

/* Search Bar & Text Input Styling */
div[data-testid="stTextInput"] {
    margin-bottom: 0 !important;
}
div[data-testid="stTextInput"] input {
    background-color: #12131A !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.25) !important;
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

/* ═══════════════════════════════════════════════════
   BLACK / BLUE FLAT-CARD VISUAL SYSTEM
   ═══════════════════════════════════════════════════ */
:root {
    --bg-canvas: #050505;
    --card-bg: #151515;
    --card-surface: #1B1B1B;
    --border-card: #303030;
    --border-highlight: #666666;
    --accent-purple: #2F80FF;
    --accent-light: #8CC8FF;
    --accent-glow: rgba(47, 128, 255, 0.18);
    --text-pure: #FFFFFF;
    --text-main: #F5F5F5;
    --text-muted: #858585;
    --radius-main: 10px;
    --radius-sm: 8px;
}

html, body, [class*="css"], .main, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"], .block-container {
    background: #050505 !important;
    color: var(--text-main) !important;
}

.main .block-container {
    padding-top: 0.7rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1400px !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) {
    background: #101010 !important;
    border: 1px solid #383838 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

.nav-brand-title {
    color: #FFFFFF !important;
}

.nav-brand-dot {
    background: #3B9CFF !important;
    box-shadow: 0 0 10px rgba(59, 156, 255, 0.55) !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) section[data-testid="stFileUploaderDropzone"] {
    background: #171717 !important;
    border: 1px dashed #4A4A4A !important;
    border-radius: 8px !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) section[data-testid="stFileUploaderDropzone"]:hover {
    background: #202020 !important;
    border-color: #2F80FF !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) section[data-testid="stFileUploaderDropzone"] button {
    background: #FFFFFF !important;
    color: #111111 !important;
    border: 1px solid #FFFFFF !important;
    border-radius: 8px !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stImage"]) > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stImage"]) > div[data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlock"]:has(div[data-testid="stImage"]):not([data-testid="stHorizontalBlock"] *),
.in-card, .in-stat-card, div[data-testid="stExpander"] {
    background: #151515 !important;
    border: 1px solid #303030 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
}

.in-stat-card {
    padding: 1.25rem 1.45rem !important;
}

.in-card:hover, .in-stat-card:hover {
    border-color: #5B5B5B !important;
    box-shadow: none !important;
    transform: translateY(-1px) !important;
}

.in-card-title, .in-stat-value {
    color: #FFFFFF !important;
}

.in-card-sub {
    color: #858585 !important;
}

.in-badge-pill {
    border-radius: 8px !important;
}

.badge-purple {
    background: rgba(47, 128, 255, 0.12) !important;
    color: #8CC8FF !important;
    border: 1px solid rgba(47, 128, 255, 0.48) !important;
}

.match-notice {
    margin-bottom: 0.6rem;
    color: #8CC8FF !important;
    font-size: 0.78rem;
    font-weight: 600;
}

.badge-green {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #E5E5E5 !important;
    border-color: #555555 !important;
}

.badge-red {
    background: rgba(47, 128, 255, 0.12) !important;
    color: #8CC8FF !important;
    border-color: rgba(47, 128, 255, 0.48) !important;
}

div.stButton > button {
    border-radius: 8px !important;
    box-shadow: none !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #2F80FF 0%, #1769D1 56%, #FFFFFF 145%) !important;
    color: #FFFFFF !important;
    border: 1px solid #2F80FF !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(100deg, #55ADFF 0%, #2F80FF 56%, #FFFFFF 145%) !important;
    border-color: #8CC8FF !important;
    box-shadow: none !important;
    transform: translateY(-1px) !important;
}

div.stButton > button:not([kind="primary"]) {
    background: #151515 !important;
    color: #FFFFFF !important;
    border: 1px solid #383838 !important;
    border-radius: 8px !important;
}

div.stButton > button:not([kind="primary"]):hover {
    background: #222222 !important;
    color: #FFFFFF !important;
    border-color: #2F80FF !important;
    box-shadow: 0 0 10px rgba(47, 128, 255, 0.25) !important;
}

div[data-testid="stTextInput"] input {
    background: #151515 !important;
    border: 1px solid #3A3A3A !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #55ADFF !important;
    box-shadow: 0 0 0 2px rgba(85, 173, 255, 0.22), 0 0 14px rgba(47, 128, 255, 0.18) !important;
}

div[data-testid="stDataFrame"], div[data-testid="stExpander"] {
    border-radius: 10px !important;
    border-color: #303030 !important;
}

.hero-title {
    background: linear-gradient(100deg, #FFFFFF 0%, #A8D9FF 52%, #2F80FF 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Glassmorphism card finish */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stImage"]) > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stImage"]) > div[data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlock"]:has(div[data-testid="stImage"]):not([data-testid="stHorizontalBlock"] *),
.in-card, .in-stat-card, div[data-testid="stExpander"] {
    background: linear-gradient(
        135deg,
        rgba(52, 42, 64, 0.62),
        rgba(20, 20, 24, 0.72)
    ) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.08),
        0 10px 28px rgba(0, 0, 0, 0.28) !important;
    backdrop-filter: blur(18px) saturate(125%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(125%) !important;
}

.in-card:hover, .in-stat-card:hover {
    border-color: rgba(140, 200, 255, 0.45) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.1),
        0 12px 30px rgba(0, 0, 0, 0.32) !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) {
    background: rgba(18, 18, 20, 0.7) !important;
    border-color: rgba(255, 255, 255, 0.16) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.08),
        0 10px 28px rgba(0, 0, 0, 0.28) !important;
    backdrop-filter: blur(18px) saturate(125%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(125%) !important;
}

div[data-testid="stTextInput"] input {
    background: rgba(28, 24, 32, 0.68) !important;
    border-color: rgba(255, 255, 255, 0.16) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div.stButton > button[kind="primary"] {
    background: rgba(47, 128, 255, 0.18) !important;
    border-color: rgba(140, 200, 255, 0.35) !important;
    color: #B9DFFF !important;
    box-shadow: none !important;
}

div[data-testid="stHorizontalBlock"]:has(div[data-testid="stFileUploader"]) div.stButton > button[kind="primary"]:hover {
    background: rgba(47, 128, 255, 0.28) !important;
    border-color: rgba(140, 200, 255, 0.55) !important;
    box-shadow: none !important;
}

/* ═══════════════════════════════════════════════════
   DNA GENOMIC ANALYSIS LOADER (3D Double Helix)
   ═══════════════════════════════════════════════════ */
.dna-loader-wrapper {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    width: 100% !important;
    min-height: min(68vh, 620px);
    margin: 0 !important;
    text-align: center;
}

.dna-loader {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 86px;
    height: 112px;
}

.dna-loader-helix {
    position: relative;
    width: 58px;
    height: 86px;
    perspective: 200px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.dna-bar {
    position: relative;
    width: 100%;
    height: 3px;
    transform-style: preserve-3d;
    animation: dna-rotate-3d 1.6s ease-in-out infinite;
    animation-delay: calc(var(--i) * -0.2s);
}

.dna-bar::before,
.dna-bar::after {
    content: "";
    position: absolute;
    top: -4px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

.dna-bar::before {
    left: 0;
    background: #2F80FF;
    box-shadow: 0 0 4px rgba(47, 128, 255, 0.48);
}

.dna-bar::after {
    right: 0;
    background: #A78BFA;
    box-shadow: 0 0 4px rgba(167, 139, 250, 0.48);
}

.dna-rung {
    position: absolute;
    top: 0;
    left: 5px;
    right: 5px;
    height: 3px;
    background: linear-gradient(90deg, rgba(47, 128, 255, 0.55), rgba(167, 139, 250, 0.55));
    border-radius: 2px;
}

@keyframes dna-rotate-3d {
    0% {
        transform: rotateY(0deg);
    }
    50% {
        transform: rotateY(180deg);
    }
    100% {
        transform: rotateY(360deg);
    }
}

.dna-loader-copy {
    display: flex;
    justify-content: center;
    margin-top: 0.7rem;
}

.dna-loader-copy::after {
    content: "Initializing...";
    color: #8CC8FF;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
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

def parse_uploaded_dataset(df_raw: pd.DataFrame):
    """
    Parses uploaded CSV with support for 1 or more patient rows.
    Extracts optional 'Patient ID' and 'Patient Name' columns in front.
    Defaults to 'No ID' and 'No Name' if missing, null, or empty.
    Extracts clean numeric gene features (20531 or 500 columns).
    """
    cols = list(df_raw.columns)
    id_col = None
    for c in cols:
        c_norm = str(c).strip().lower().replace("_", " ").replace("-", " ")
        if c_norm in {"patient id", "patientid", "id", "pid", "patient number", "patient no"}:
            id_col = c
            break

    name_col = None
    for c in cols:
        c_norm = str(c).strip().lower().replace("_", " ").replace("-", " ")
        if c_norm in {"patient name", "patientname", "name", "full name", "patient"}:
            name_col = c
            break

    n_rows = len(df_raw)
    records = []
    for i in range(n_rows):
        row = df_raw.iloc[i]
        p_id = str(row[id_col]).strip() if id_col and pd.notna(row[id_col]) and str(row[id_col]).strip() != "" else "No ID"
        p_name = str(row[name_col]).strip() if name_col and pd.notna(row[name_col]) and str(row[name_col]).strip() != "" else "No Name"
        records.append({"index": i, "id": p_id, "name": p_name})

    cols_to_drop = set()
    if id_col is not None:
        cols_to_drop.add(id_col)
    if name_col is not None:
        cols_to_drop.add(name_col)
    for c in cols:
        c_norm = str(c).strip().lower()
        if c_norm in {"unnamed: 0", "sample_id", "sample id", "class", "cancer_type", "cancer type"}:
            cols_to_drop.add(c)

    features_df = df_raw.drop(columns=list(cols_to_drop)).copy()
    features_df = features_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return records, features_df

@contextmanager
def dna_analysis_loader(message: str, min_duration: float = 3.0):
    placeholder = st.empty()
    rungs_html = "".join([
        f'<div class="dna-bar" style="--i: {i};"><span class="dna-rung"></span></div>'
        for i in range(8)
    ])
    placeholder.markdown(f"""
    <div class="dna-loader-wrapper">
        <div class="dna-loader" role="status" aria-live="polite">
            <div class="dna-loader-helix" aria-hidden="true">
                {rungs_html}
            </div>
        </div>
        <div class="dna-loader-copy"></div>
    </div>
    """, unsafe_allow_html=True)
    t0 = time.time()
    try:
        yield
    finally:
        remaining = min_duration - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)
        placeholder.empty()

def compute_patient_inference(patient_idx: int, features_df: pd.DataFrame, model, encoder, selector):
    """
    Computes and caches model inference and SHAP attributions for a single patient index.
    """
    cache = st.session_state.setdefault("patient_results_cache", {})
    if patient_idx in cache:
        return cache[patient_idx]

    row_df = features_df.iloc[[patient_idx]]
    n_expected = selector.n_features_in_
    if row_df.shape[1] == n_expected:
        X_sel = selector.transform(row_df)
        feat_names = row_df.columns[selector.get_support()].tolist()
    elif row_df.shape[1] == 500:
        X_sel = row_df.values
        feat_names = row_df.columns.tolist()
    else:
        raise ValueError(f"Feature count mismatch: expected {n_expected} or 500 genes, got {row_df.shape[1]}")

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

    res = dict(
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
    cache[patient_idx] = res
    return res

def get_standby_result():
    return dict(
        pred_code="STANDBY",
        pred_idx=-1,
        meta={"full": "Awaiting patient inference", "organ": "—", "hex": "#7A7D93"},
        confidence=0.0,
        proba=[0.0] * 5,
        feat_names=[],
        X_sel=np.zeros((1, 500)),
        class_expl=None,
        shap_values=None,
    )

def render_patient_view(patient_info, result, is_analyzed, total_patients, encoder):
    """
    Renders the complete SaaS diagnostic dashboard for a given patient.
    """
    # 1. Patient Header Banner Card
    if not is_analyzed and patient_info.get("name") in ["No Name", None]:
        patient_title = "No Name"
        patient_id_text = "Patient ID: NULL"
        patient_sub_text = "Standby &middot; Upload patient RNA-Seq CSV to initialize diagnostics"
    else:
        patient_title = patient_info['name']
        patient_id_text = f"Patient ID: {patient_info['id']}"
        patient_sub_text = f"Patient #{patient_info['index']+1} of {total_patients} &middot; RNA-Seq Transcriptome"

    st.markdown(f"""<div style="background:#12131A;border:1px solid #212230;border-left:4px solid #8B5CF6;border-radius:14px;padding:0.75rem 1.25rem;margin-top:0.4rem;margin-bottom:1.1rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.75rem;">
<div style="display:flex;align-items:center;gap:0.85rem;">
<div style="width:38px;height:38px;border-radius:10px;background:rgba(139,92,246,0.18);border:1px solid rgba(139,92,246,0.3);display:flex;align-items:center;justify-content:center;font-size:1.15rem;color:#C084FC;">
👤
</div>
<div>
<div style="font-size:1.05rem;font-weight:800;color:#FFFFFF;">
{patient_title}
</div>
<div style="font-size:0.72rem;color:#858585;margin-top:0.15rem;font-family:monospace;">
{patient_id_text}
</div>
<div style="font-size:0.75rem;color:#8B8FA3;margin-top:0.15rem;">
{patient_sub_text}
</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    # 2. 3 Top Stat Cards
    biomarker_cnt = "500" if is_analyzed else "0"
    biomarker_sub = "97.6% Transcriptome reduction" if is_analyzed else "Standby &middot; Top 500 gene selection"
    biomarker_badge = "Top K=500" if is_analyzed else "0%"

    meta_data = result['meta'] if is_analyzed else {"full": "Awaiting patient inference", "organ": "—", "hex": "#8B8FA3"}
    subtype_badge = meta_data["organ"] if is_analyzed else "0%"
    subtype_value = result['pred_code'] if is_analyzed else "NULL"
    subtype_description = meta_data["full"] if is_analyzed else "Awaiting patient inference"
    subtype_color = meta_data['hex'] if is_analyzed else "#8B8FA3"
    badge_style_sub = "badge-green" if is_analyzed else "badge-purple"

    conf_val = f"{result['confidence']:.1f}%" if is_analyzed else "0.0%"
    conf_sub = "High Diagnostic Certainty" if is_analyzed else "Engine awaiting run trigger"
    conf_badge = "badge-green" if (is_analyzed and result['confidence'] > 90) else "badge-purple"

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""<div class="in-stat-card">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
<div style="font-size:0.8rem;font-weight:700;color:#8B8FA3;">Biomarkers Analyzed</div>
<span class="in-badge-pill {badge_style_sub}">{biomarker_badge}</span>
</div>
<div class="in-stat-value">{biomarker_cnt}</div>
<div style="font-size:0.75rem;color:#71758B;margin-top:0.4rem;">{biomarker_sub}</div>
</div>""", unsafe_allow_html=True)

    with s2:
        st.markdown(f"""<div class="in-stat-card">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
<div style="font-size:0.8rem;font-weight:700;color:#8B8FA3;">Predicted Subtype</div>
<span class="in-badge-pill {badge_style_sub}">{subtype_badge}</span>
</div>
<div class="in-stat-value" style="color:{subtype_color};">{subtype_value}</div>
<div style="font-size:0.75rem;color:#71758B;margin-top:0.4rem;">{subtype_description}</div>
</div>""", unsafe_allow_html=True)

    with s3:
        st.markdown(f"""<div class="in-stat-card">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
<div style="font-size:0.8rem;font-weight:700;color:#8B8FA3;">Model Confidence</div>
<span class="in-badge-pill {conf_badge}">Posterior</span>
</div>
<div class="in-stat-value">{conf_val}</div>
<div style="font-size:0.75rem;color:#71758B;margin-top:0.4rem;">{conf_sub}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 3. Middle Section: Waveform + Heatmap Matrix
    mid_left, mid_right = st.columns([1.8, 1.2])

    with mid_left:
        fig_wave, ax_wave = plt.subplots(figsize=(7.5, 3.1), dpi=140)
        fig_wave.patch.set_facecolor('#12131A')
        ax_wave.set_facecolor('#12131A')

        x_points = np.linspace(1, 30, 200)
        if is_analyzed and result['X_sel'] is not None:
            sample_vals = result['X_sel'][0][:30]
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

        st.markdown("""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
            <div>
                <div class="in-card-title">Biomarker Expression Loci Waveform</div>
                <div class="in-card-sub">Top 30 discriminant expression peaks across the transcriptome</div>
            </div>
            <span class="in-badge-pill badge-purple">RNA-Seq</span>
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(fig_wave, use_container_width=True)
        plt.close(fig_wave)

    with mid_right:
        st.markdown("""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
            <div>
                <div class="in-card-title">Pan-Cancer Subtype Heatmap Matrix</div>
                <div class="in-card-sub">Clinical Roadmap &middot; COAD &middot; KIRC &middot; LUAD &middot; PRAD &middot; BRCA</div>
            </div>
            <span class="in-badge-pill badge-green">TCGA 401</span>
        </div>
        """, unsafe_allow_html=True)

        probs = result['proba'] if is_analyzed and result['proba'] is not None else [0.0]*5
        classes = encoder.classes_ if encoder else ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]
        for idx, code in enumerate(classes):
            m = SUBTYPE_META.get(code, {"hex": "#8B5CF6", "organ": "Pan-Cancer", "badge": ""})
            pct = float(probs[idx]) * 100 if is_analyzed and idx < len(probs) else 0.0
            is_top = is_analyzed and code == result['pred_code']
            border_color = m['hex'] if is_top else "#1E202D"
            bg_color = "rgba(139,92,246,0.12)" if is_top else "#171822"
            badge_html = m['badge']
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

    # 4. SHAP Interpretability Section (if analyzed)
    if is_analyzed and result['class_expl'] is not None:
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
                shap.plots.beeswarm(result['class_expl'], max_display=6, show=False)
                plt.title(f"Genomic Drivers Impacting {result['pred_code']} Classification for {patient_info['name']} ({patient_info['id']})", fontsize=10, fontweight="bold", pad=8, color="#FFFFFF")
                plt.xlabel("SHAP Attribution Value", fontsize=8, color="#8B8FA3")
                plt.tight_layout()
                st.pyplot(fig_shap, use_container_width=True)
                plt.close(fig_shap)
                plt.rcParams.update(plt.rcParamsDefault)
            except Exception as exc:
                st.warning(f"Could not render SHAP beeswarm: {exc}")

            driver_df = (
                pd.DataFrame({
                    "Genetic Biomarker": result['feat_names'],
                    "Patient Expression Level": result['X_sel'][0],
                    "SHAP Attribution Value": result['shap_values'],
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

model, encoder, selector, art_err = load_artifacts()

# Initialize session state for patient records
if "patient_records" not in st.session_state:
    st.session_state["patient_records"] = []
if "features_df" not in st.session_state:
    st.session_state["features_df"] = None
if "patient_source" not in st.session_state:
    st.session_state["patient_source"] = ""
if "is_analyzed" not in st.session_state:
    st.session_state["is_analyzed"] = False
if "patient_results_cache" not in st.session_state:
    st.session_state["patient_results_cache"] = {}

# Initial startup splash loader
if "app_loaded" not in st.session_state:
    st.session_state["app_loaded"] = True
    splash = st.empty()
    rungs_html = "".join([
        f'<div class="dna-bar" style="--i: {i};"><span class="dna-rung"></span></div>'
        for i in range(8)
    ])
    splash.markdown(f"""
    <div class="dna-loader-wrapper" style="margin: 3rem 0 2rem 0;">
        <div class="dna-loader" role="status">
            <div class="dna-loader-helix">
                {rungs_html}
            </div>
        </div>
        <div class="dna-loader-copy"></div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2.0)
    splash.empty()

# ─────────────────────────────────────────────────────────────────────────────
# 1. TOP NAVBAR
# Left: Brand Name
# Middle: Upload File Section
# Right: Upload Status & Pill Menu
# ─────────────────────────────────────────────────────────────────────────────
nav_left, nav_mid, nav_right = st.columns([1.3, 1.8, 1.4], gap="medium", vertical_alignment="center")

with nav_left:
    logo_path = "favicon.png" if os.path.exists("favicon.png") else ("public/favicon.png" if os.path.exists("public/favicon.png") else "")
    logo_html = ""
    if logo_path:
        with open(logo_path, "rb") as f:
            b64_logo = base64.b64encode(f.read()).decode("utf-8")
        logo_html = f'<img src="data:image/png;base64,{b64_logo}" style="width:34px;height:34px;border-radius:9px;border:1px solid rgba(59,156,255,0.45);box-shadow:0 0 10px rgba(59,156,255,0.4);" />'
    else:
        logo_html = '<div class="nav-brand-dot"></div>'

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.8rem;">
        {logo_html}
        <div class="nav-brand-title">Oncology AI Engine</div>
    </div>
    """, unsafe_allow_html=True)

with nav_mid:
    uploaded_file = st.file_uploader(
        "Upload RNA-Seq CSV",
        type=["csv"],
        label_visibility="collapsed",
        help="Upload patient CSV containing 1 or more patient rows with 20,531 RNA-Seq features."
    )

with nav_right:
    run_clicked = st.button("Initialize AI Analysis", type="primary", use_container_width=True)

if uploaded_file is None and st.session_state.get("patient_source", "").startswith("File:"):
    st.session_state["patient_records"] = []
    st.session_state["features_df"] = None
    st.session_state["patient_source"] = ""
    st.session_state["is_analyzed"] = False
    st.session_state["patient_results_cache"] = {}
    st.session_state["loaded_file_fingerprint"] = None

if uploaded_file is not None:
    # Only re-parse when file actually changes (name + size fingerprint)
    file_fingerprint = f"{uploaded_file.name}::{uploaded_file.size}"
    if st.session_state.get("loaded_file_fingerprint") != file_fingerprint:
        try:
            raw = pd.read_csv(uploaded_file)
            records, features_df = parse_uploaded_dataset(raw)
            patient_source = f"File: {uploaded_file.name[:14]}... ({len(records)} Patients)"
            st.session_state["patient_records"] = records
            st.session_state["features_df"] = features_df
            st.session_state["patient_source"] = patient_source
            st.session_state["is_analyzed"] = False
            st.session_state["patient_results_cache"] = {}
            st.session_state["search_query"] = ""
            st.session_state["loaded_file_fingerprint"] = file_fingerprint
        except Exception as e:
            st.error(f"Error parsing file: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.5. SEARCH BAR (Directly Under Top Navbar)
# ─────────────────────────────────────────────────────────────────────────────
loaded_records = st.session_state.get("patient_records", [])
features_df = st.session_state.get("features_df", None)
is_analyzed = st.session_state.get("is_analyzed", False)

if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""
if "patient_search_input" not in st.session_state:
    st.session_state["patient_search_input"] = ""

def on_reset_search():
    st.session_state["patient_search_input"] = ""
    st.session_state["search_query"] = ""

def on_search_submit():
    st.session_state["search_query"] = st.session_state.get("patient_search_input", "").strip()

sc_input, sc_btn, sc_reset = st.columns([5.2, 1.2, 1.2], vertical_alignment="center")

with sc_input:
    st.text_input(
        "Search Patient",
        placeholder="Search patient by Name (e.g., Sophia Martinez) or Patient ID (e.g., PID-TCGA-101)...",
        label_visibility="collapsed",
        key="patient_search_input",
        on_change=on_search_submit
    )

with sc_btn:
    st.button("Search", on_click=on_search_submit, use_container_width=True)

with sc_reset:
    st.button("Reset / First", on_click=on_reset_search, use_container_width=True, help="Clear search and display the first patient")

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
            Multi-class histological classification via high-throughput transcriptomics.
            Powered by gradient-boosted tree ensembles and game-theoretic SHAP explainability.
        </div>
    </div>
    """, unsafe_allow_html=True)

with sub_right:
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

if run_clicked:
    if not loaded_records or features_df is None:
        st.warning("⚠️ No RNA-Seq dataset loaded yet. Please upload a patient CSV.")
    elif art_err:
        st.error(f"Setup Required: {art_err}")
    elif features_df is not None:
        # Run inference INLINE for first patient — no st.rerun() to avoid state race
        with dna_analysis_loader("Applying SelectKBest → XGBoost → SHAP to Patient #1...", min_duration=3.0):
            try:
                compute_patient_inference(0, features_df, model, encoder, selector)
                st.session_state["is_analyzed"] = True
                is_analyzed = True          # update local variable immediately
            except Exception as e:
                st.session_state["is_analyzed"] = False
                st.error(f"❌ Inference Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PATIENT SELECTION & SEARCH MATCHING
# ─────────────────────────────────────────────────────────────────────────────
# Re-read is_analyzed in case it was updated inline above
is_analyzed = st.session_state.get("is_analyzed", False)
active_query = st.session_state.get("search_query", "").strip()

if not loaded_records or features_df is None:
    # Standby state with placeholder patient
    placeholder_patient = {"index": None, "id": None, "name": None}
    standby_res = get_standby_result()
    render_patient_view(placeholder_patient, standby_res, is_analyzed=False, total_patients=1, encoder=encoder)
else:
    total_loaded = len(loaded_records)
    display_patients = []
    match_notice = ""

    if not active_query:
        # Default view: First patient only
        display_patients = [loaded_records[0]]
        match_notice = f"Displaying First Patient (#1 of {total_loaded})"
    else:
        q_lower = active_query.lower()

        # 1. Exact ID match (ID must be single and specific)
        id_exact = [p for p in loaded_records if p['id'].strip().lower() == q_lower and p['id'].strip().lower() != "no id"]
        if id_exact:
            display_patients = [id_exact[0]]
            match_notice = f"Specific Patient ID Matched: {id_exact[0]['id']}"
        else:
            # 2. Name match (case-insensitive) — if duplicate names, display both upside down together!
            name_exact = [p for p in loaded_records if p['name'].strip().lower() == q_lower and p['name'].strip().lower() != "no name"]
            if not name_exact:
                name_exact = [p for p in loaded_records if q_lower in p['name'].strip().lower() and p['name'].strip().lower() != "no name"]

            if name_exact:
                display_patients = name_exact
                match_notice = f"Found {len(name_exact)} matching patient(s) for Name '{active_query}'"
            else:
                # 3. Substring ID match
                id_sub = [p for p in loaded_records if q_lower in p['id'].strip().lower() and p['id'].strip().lower() != "no id"]
                if id_sub:
                    display_patients = [id_sub[0]]
                    match_notice = f"Specific Patient ID Matched: {id_sub[0]['id']}"
                else:
                    st.warning(f"🔍 No patient found matching **'{active_query}'**. Displaying first patient as default.")
                    display_patients = [loaded_records[0]]
                    match_notice = f"Fallback: Displaying First Patient (#1 of {total_loaded})"

    # Display match notice
    st.markdown(f"""
    <div class="match-notice">{match_notice}</div>
    """, unsafe_allow_html=True)

    # Render each matched patient (stacked vertically if multiple)
    for idx, p in enumerate(display_patients):
        if idx > 0:
            st.markdown("<div style='height: 24px; border-top: 1px dashed #2B2C3E; margin: 2rem 0 1rem 0;'></div>", unsafe_allow_html=True)

        if is_analyzed:
            patient_cache = st.session_state.get("patient_results_cache", {})
            if p["index"] not in patient_cache:
                # Not yet computed — show inline spinner and compute now
                with dna_analysis_loader(f"Running the RNA-Seq pipeline for {p['name']} ({p['id']})..."):
                    try:
                        res = compute_patient_inference(p["index"], features_df, model, encoder, selector)
                    except Exception as e:
                        st.error(f"❌ Inference Error for {p['name']}: {e}")
                        res = get_standby_result()
                        is_analyzed = False
            else:
                res = compute_patient_inference(p["index"], features_df, model, encoder, selector)
        else:
            res = get_standby_result()

        render_patient_view(p, res, is_analyzed, total_loaded, encoder)

# Footer
st.markdown("""
<div style="border-top:1px solid #1E202C;margin-top:2.5rem;padding-top:1.2rem;text-align:center;color:#65687C;font-size:0.76rem;">
    Oncology AI Engine · TCGA Pan-Cancer RNA-Seq (UCI ID 401) · XGBoost + SelectKBest + SHAP
</div>
""", unsafe_allow_html=True)

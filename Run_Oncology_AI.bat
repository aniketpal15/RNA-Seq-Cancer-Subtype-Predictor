@echo off
title Oncology AI Engine - Windows Launcher
color 0B
echo ========================================================
echo         Oncology AI Engine - Desktop Launcher
echo ========================================================
echo Starting local clinical engine...

cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    start "" http://localhost:8501
    ".venv\Scripts\streamlit.exe" run app.py
) else if exist "venv\Scripts\streamlit.exe" (
    start "" http://localhost:8501
    "venv\Scripts\streamlit.exe" run app.py
) else (
    start "" http://localhost:8501
    streamlit run app.py
)

pause

@echo off
cd /d "%~dp0"
REM Run with UTF-8 so app.py loads correctly on Windows
set PYTHONUTF8=1
python app.py
pause

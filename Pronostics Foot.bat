@echo off
rem Lance l'application Pronostics Foot (sans fenetre noire) et ouvre l'interface dans le navigateur.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" app.py

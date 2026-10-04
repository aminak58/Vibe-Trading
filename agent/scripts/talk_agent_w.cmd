@echo off
rem Talk to the Vibe-Trading research agent WITHOUT opening a console window.
rem Usage: talk_agent_w.cmd "message" [--session ID] [--timeout SEC]
rem (pythonw = no window; output is lost, so log file still records the exchange)
setlocal
cd /d "%~dp0"
pythonw.exe talk_agent.py %*
endlocal

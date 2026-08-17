@echo off
REM Opens the web UI. The server on the GPU box binds localhost only - there is
REM no authentication on it - so this tunnels port 8000 over SSH instead of
REM exposing it publicly. Leave this window open while you use the app.
setlocal
if "%~1"=="" (set IP=52.66.222.7) else (set IP=%~1)
echo Tunnelling http://localhost:8000 to %IP% ...
start "" http://localhost:8000
ssh -i "C:\Users\hp\.ssh\face-attendance.pem" -N -L 8000:localhost:8000 ubuntu@%IP%

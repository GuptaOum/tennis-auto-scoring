@echo off
REM Analyse a tennis video.
REM
REM   analyze.bat path\to\video.mp4
REM   analyze.bat path\to\video.mp4 --limit 300     (first 300 frames only)
REM
REM Results land in output\ : annotated.mp4, report.json, ball_track.json

setlocal

if "%~1"=="" (
    echo Usage: analyze.bat ^<video-file^> [extra options]
    echo.
    echo   analyze.bat input_videos\match.mp4
    echo   analyze.bat input_videos\match.mp4 --limit 300
    echo   analyze.bat input_videos\match.mp4 --no-video
    exit /b 1
)

if not exist "%~1" (
    echo Video not found: %~1
    exit /b 1
)

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo No virtualenv found. Run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

set VIDEO=%~1
shift

"%~dp0.venv\Scripts\python.exe" -m tennis.cli --input "%VIDEO%" --out "%~dp0output" %1 %2 %3 %4 %5 %6

endlocal

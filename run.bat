@echo off
echo Starting FastAPI Backend...
start cmd /k "python main.py"
echo Starting React Frontend...
cd frontend
start cmd /k "npm run dev"
echo Both servers starting up!
pause

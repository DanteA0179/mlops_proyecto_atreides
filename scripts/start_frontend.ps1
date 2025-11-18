Write-Host "🚀 Iniciando Stack..." -ForegroundColor Green

Write-Host "📡 Iniciando API..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "poetry run uvicorn src.api.main:app --reload"

Start-Sleep -Seconds 5

Write-Host "🎨 Iniciando Frontend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; poetry run streamlit run app.py"

Write-Host ""
Write-Host "✅ URLs:" -ForegroundColor Green
Write-Host "   - API: http://localhost:8000" -ForegroundColor Yellow
Write-Host "   - Frontend: http://localhost:8501" -ForegroundColor Yellow

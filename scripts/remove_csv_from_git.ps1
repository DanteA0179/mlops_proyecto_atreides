# Script PowerShell para remover archivos CSV del índice de Git
# Esto NO borra los archivos localmente, solo deja de rastrearlos en Git

Write-Host "🔍 Buscando archivos CSV rastreados por Git..." -ForegroundColor Cyan

# Remover todos los CSV del índice de Git (pero mantenerlos localmente)
try {
    git rm --cached -r *.csv 2>$null
    git rm --cached -r data/**/*.csv 2>$null
    Write-Host "✅ Archivos CSV removidos del índice de Git" -ForegroundColor Green
} catch {
    Write-Host "⚠️ No se encontraron archivos CSV para remover o ya fueron removidos" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📝 Los archivos siguen en tu disco, pero Git ya no los rastrea" -ForegroundColor White
Write-Host ""
Write-Host "Próximos pasos:" -ForegroundColor Yellow
Write-Host "1. Verifica los cambios: git status"
Write-Host "2. Commitea los cambios: git commit -m 'chore: remove CSV files from Git tracking'"
Write-Host "3. Push: git push"

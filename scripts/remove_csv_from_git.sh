#!/bin/bash
# Script para remover archivos CSV del índice de Git
# Esto NO borra los archivos localmente, solo deja de rastrearlos en Git

echo "🔍 Buscando archivos CSV rastreados por Git..."

# Remover todos los CSV del índice de Git (pero mantenerlos localmente)
git rm --cached -r *.csv 2>/dev/null || true
git rm --cached -r data/**/*.csv 2>/dev/null || true

echo "✅ Archivos CSV removidos del índice de Git"
echo "📝 Los archivos siguen en tu disco, pero Git ya no los rastrea"
echo ""
echo "Próximos pasos:"
echo "1. Verifica los cambios: git status"
echo "2. Commitea los cambios: git commit -m 'chore: remove CSV files from Git tracking'"
echo "3. Push: git push"

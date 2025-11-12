#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start Dagster development server

.DESCRIPTION
    Starts Dagster development server with automatic configuration.
    Sets DAGSTER_HOME and launches the web UI.

.PARAMETER Port
    Port for Dagster UI (default: 3000)

.PARAMETER BindHost
    Host to bind to (default: 127.0.0.1)

.EXAMPLE
    .\start-dagster.ps1
    Starts Dagster on port 3000

.EXAMPLE
    .\start-dagster.ps1 -Port 3001
    Starts Dagster on port 3001

.EXAMPLE
    .\start-dagster.ps1 -Port 3000 -BindHost 0.0.0.0
    Starts Dagster on port 3000, accessible from all interfaces
#>

param(
    [int]$Port = 3000,
    [string]$BindHost = "127.0.0.1"
)

# Set working directory to project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# Set DAGSTER_HOME
$env:DAGSTER_HOME = Join-Path $ProjectRoot "dagster_home"

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Energy Optimization Copilot" -ForegroundColor Cyan
Write-Host "  Dagster Development Server" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project Root:  $ProjectRoot" -ForegroundColor Yellow
Write-Host "DAGSTER_HOME:  $env:DAGSTER_HOME" -ForegroundColor Yellow
Write-Host "UI Address:    http://${BindHost}:${Port}" -ForegroundColor Yellow
Write-Host ""
Write-Host "Available Jobs:" -ForegroundColor Green
Write-Host "  - complete_training_job      (XGBoost, LightGBM, CatBoost, Ensembles)" -ForegroundColor White
Write-Host "  - chronos_zeroshot_job       (Chronos-2 zero-shot)" -ForegroundColor White
Write-Host "  - chronos_finetuned_job      (Chronos-2 fine-tuned)" -ForegroundColor White
Write-Host "  - chronos_covariates_job     (Chronos-2 with covariates)" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

# Start Dagster dev server
poetry run dagster dev -m src.dagster_pipeline.definitions -h $BindHost -p $Port

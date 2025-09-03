# Project Nova - Data Generation Script
# This script generates multiple datasets with different configurations

param(
    [switch]$All,
    [switch]$Dev,
    [switch]$Test,
    [switch]$Prod,
    [switch]$Scenarios,
    [string]$OutputDir = "data"
)

Write-Host "🚀 Project Nova Data Generation Script" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Create output directory if it doesn't exist
if (-not (Test-Path $OutputDir)) {
    Write-Host "📁 Creating output directory: $OutputDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

function Generate-Dataset {
    param(
        [int]$Size,
        [int]$Seed,
        [string]$OutputFile,
        [string]$Description
    )
    
    Write-Host "📊 Generating $Description..." -ForegroundColor Green
    Write-Host "   Size: $Size records, Seed: $Seed, Output: $OutputFile" -ForegroundColor Gray
    
    $command = "python src/generate_data.py --n $Size --seed $Seed --out $OutputFile"
    
    try {
        Invoke-Expression $command
        Write-Host "   ✅ Success!" -ForegroundColor Green
    }
    catch {
        Write-Host "   ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host ""
}

# Development datasets (small, fast)
if ($Dev -or $All) {
    Write-Host "🔧 Generating Development Datasets" -ForegroundColor Magenta
    Write-Host "===================================" -ForegroundColor Magenta
    
    Generate-Dataset -Size 1000 -Seed 42 -OutputFile "$OutputDir/dev_tiny.csv" -Description "Tiny Dev Dataset"
    Generate-Dataset -Size 5000 -Seed 42 -OutputFile "$OutputDir/dev_small.csv" -Description "Small Dev Dataset"
    Generate-Dataset -Size 15000 -Seed 43 -OutputFile "$OutputDir/dev_medium.csv" -Description "Medium Dev Dataset"
}

# Test datasets
if ($Test -or $All) {
    Write-Host "🧪 Generating Test Datasets" -ForegroundColor Magenta
    Write-Host "============================" -ForegroundColor Magenta
    
    Generate-Dataset -Size 25000 -Seed 100 -OutputFile "$OutputDir/test_partners.csv" -Description "Main Test Dataset"
    Generate-Dataset -Size 25000 -Seed 101 -OutputFile "$OutputDir/test_partners_alt.csv" -Description "Alternative Test Dataset"
    Generate-Dataset -Size 10000 -Seed 999 -OutputFile "$OutputDir/holdout_test.csv" -Description "Holdout Test Dataset"
}

# Production datasets
if ($Prod -or $All) {
    Write-Host "🏭 Generating Production Datasets" -ForegroundColor Magenta
    Write-Host "==================================" -ForegroundColor Magenta
    
    Generate-Dataset -Size 50000 -Seed 1000 -OutputFile "$OutputDir/prod_baseline.csv" -Description "Production Baseline"
    Generate-Dataset -Size 100000 -Seed 1001 -OutputFile "$OutputDir/prod_large.csv" -Description "Large Production Dataset"
    Generate-Dataset -Size 30000 -Seed 2000 -OutputFile "$OutputDir/validation_set.csv" -Description "Validation Dataset"
}

# Specialized scenario datasets
if ($Scenarios -or $All) {
    Write-Host "🎭 Generating Scenario Datasets" -ForegroundColor Magenta
    Write-Host "===============================" -ForegroundColor Magenta
    
    # ML Training Pipeline
    Generate-Dataset -Size 35000 -Seed 42 -OutputFile "$OutputDir/train.csv" -Description "Training Set (70%)"
    Generate-Dataset -Size 7500 -Seed 43 -OutputFile "$OutputDir/val.csv" -Description "Validation Set (15%)"
    Generate-Dataset -Size 7500 -Seed 44 -OutputFile "$OutputDir/test.csv" -Description "Test Set (15%)"
    
    # A/B Testing
    Generate-Dataset -Size 20000 -Seed 500 -OutputFile "$OutputDir/ab_control.csv" -Description "A/B Test Control Group"
    Generate-Dataset -Size 20000 -Seed 501 -OutputFile "$OutputDir/ab_treatment.csv" -Description "A/B Test Treatment Group"
    
    # Fairness Analysis
    Generate-Dataset -Size 30000 -Seed 1000 -OutputFile "$OutputDir/fairness_test.csv" -Description "Fairness Analysis Dataset"
    
    # Time Series (Multiple periods)
    Generate-Dataset -Size 40000 -Seed 2022 -OutputFile "$OutputDir/partners_2022.csv" -Description "Historical Data 2022"
    Generate-Dataset -Size 45000 -Seed 2023 -OutputFile "$OutputDir/partners_2023.csv" -Description "Historical Data 2023"
    Generate-Dataset -Size 50000 -Seed 2024 -OutputFile "$OutputDir/partners_2024.csv" -Description "Current Data 2024"
}

# Default behavior - generate standard dataset
if (-not ($All -or $Dev -or $Test -or $Prod -or $Scenarios)) {
    Write-Host "📊 Generating Standard Dataset" -ForegroundColor Magenta
    Write-Host "==============================" -ForegroundColor Magenta
    Generate-Dataset -Size 50000 -Seed 42 -OutputFile "$OutputDir/partners.csv" -Description "Standard Project Nova Dataset"
}

Write-Host "🎉 Data Generation Complete!" -ForegroundColor Cyan
Write-Host "📁 All datasets saved to: $OutputDir" -ForegroundColor Yellow

# Display directory contents
Write-Host "`n📋 Generated Files:" -ForegroundColor Cyan
Get-ChildItem -Path $OutputDir -Filter "*.csv" | ForEach-Object {
    $size = [math]::Round($_.Length / 1MB, 2)
    Write-Host "   $($_.Name) - $size MB" -ForegroundColor Gray
}

Write-Host "`n💡 Usage Examples:" -ForegroundColor Yellow
Write-Host "   .\scripts\generate_datasets.ps1 -Dev      # Generate development datasets" -ForegroundColor Gray
Write-Host "   .\scripts\generate_datasets.ps1 -Test     # Generate test datasets" -ForegroundColor Gray
Write-Host "   .\scripts\generate_datasets.ps1 -Prod     # Generate production datasets" -ForegroundColor Gray
Write-Host "   .\scripts\generate_datasets.ps1 -All      # Generate all datasets" -ForegroundColor Gray
Write-Host "   .\scripts\generate_datasets.ps1           # Generate standard dataset only" -ForegroundColor Gray

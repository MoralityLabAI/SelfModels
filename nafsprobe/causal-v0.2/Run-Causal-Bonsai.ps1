param(
    [string]$BaseUrl = 'http://127.0.0.1:8101/v1',
    [string]$Model = 'Bonsai-8B-Q1_0',
    [int]$Replicates = 3,
    [int]$Seed = 241016
)
$ErrorActionPreference = 'Stop'
$probeStamp = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff')
$probeOutput = Join-Path $PSScriptRoot "results\causal-bonsai-$probeStamp-UTC"
& python -u (Join-Path $PSScriptRoot 'causal_probe.py') --base-url $BaseUrl --model $Model --replicates $Replicates --seed $Seed --out-dir $probeOutput
if ($LASTEXITCODE -ne 0) { throw "Causal experiment failed with exit code $LASTEXITCODE. Partial evidence, if any, is in $probeOutput" }
Write-Host "Report: $(Join-Path $probeOutput 'REPORT.md')"

param(
    [string]$BaseUrl = 'http://127.0.0.1:8101/v1',
    [string]$Model = 'Bonsai-8B-Q1_0'
)
$ErrorActionPreference = 'Stop'
$probeRunId = 'bonsai-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
$probeResultsDir = Join-Path $PSScriptRoot 'results'
New-Item -ItemType Directory -Path $probeResultsDir -Force | Out-Null
$probeOutput = Join-Path $probeResultsDir ($probeRunId + '.json')
$probeDatabase = Join-Path $probeResultsDir ($probeRunId + '.sqlite3')
& python -u (Join-Path $PSScriptRoot 'nafs_probe.py') --backend local --model $Model --base-url $BaseUrl --phase compare --run-id $probeRunId --db $probeDatabase --out $probeOutput
if ($LASTEXITCODE -ne 0) { throw "NafsProbe failed with exit code $LASTEXITCODE; state is retained in $probeDatabase" }
Write-Host "Results: $probeOutput"

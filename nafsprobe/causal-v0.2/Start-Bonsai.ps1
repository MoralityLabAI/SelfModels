param([int]$Port = 8101, [int]$Context = 32768)
$ErrorActionPreference = 'Stop'
$probeExe = 'C:\Users\patri\Documents\Codex\BitAgent-gym\llama-prism\bin\llama-server.exe'
$probeModel = 'C:\Users\patri\Documents\Codex\BitAgent-gym\prime-artifacts\Bonsai-8B-Q1_0.gguf'
$probeUrl = "http://127.0.0.1:$Port"
try { $probeModels = Invoke-RestMethod "$probeUrl/v1/models" -TimeoutSec 3 } catch { $probeModels = $null }
if ($probeModels) {
    if ($probeModels.data.id -contains 'Bonsai-8B-Q1_0') { Write-Host "Bonsai already running at $probeUrl/v1"; return }
    throw "A different server is already using port $Port."
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) { throw "Port $Port is occupied." }
if (!(Test-Path -LiteralPath $probeExe) -or !(Test-Path -LiteralPath $probeModel)) { throw 'Existing Bonsai model or llama-server executable is missing.' }
$probeLogs = Join-Path $PSScriptRoot 'server-logs'
New-Item -ItemType Directory -Path $probeLogs -Force | Out-Null
$probeStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$probeServer = Start-Process -FilePath $probeExe -ArgumentList @('-m',$probeModel,'--alias','Bonsai-8B-Q1_0','-ngl','99','-c',"$Context",'--host','127.0.0.1','--port',"$Port",'--no-webui','-np','1','-b','256','-ub','128') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $probeLogs "$probeStamp.out.log") -RedirectStandardError (Join-Path $probeLogs "$probeStamp.err.log") -PassThru
$probeServer.Id | Set-Content -LiteralPath (Join-Path $probeLogs 'server.pid')
for ($probeAttempt = 0; $probeAttempt -lt 20; $probeAttempt++) {
    if ($probeServer.HasExited) { throw "Server exited. See $probeLogs" }
    try {
        if ((Invoke-RestMethod "$probeUrl/health" -TimeoutSec 1).status -eq 'ok') {
            Write-Host "Bonsai ready at $probeUrl/v1 (PID $($probeServer.Id), context $Context)."
            return
        }
    } catch {}
    Start-Sleep -Seconds 1
}
throw "Server is still loading or unhealthy. Check $probeLogs before retrying."

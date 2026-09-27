#requires -Version 5.1
<#
.SYNOPSIS
  Kill any existing Conduix server on the configured port, then restart.

.DESCRIPTION
  Stops whatever is listening on the port together with its child process
  tree (the Codex app-server that the SDK spawns), then starts uvicorn in the
  conda env. OPENAI_API_KEY / CODEX_API_KEY are removed from the server's
  environment so usage always bills to the ChatGPT subscription.

.PARAMETER Port
  Port to bind. Resolution order: -Port flag > $env:CONDUIX_PORT > CONDUIX_PORT
  in .env > 8766.

.PARAMETER Host_
  Bind address. Default: 127.0.0.1 (or CONDUIX_HOST in .env / env).

.PARAMETER Env
  Conda env name. Default: conduix.

.PARAMETER Reload
  Run uvicorn with --reload (dev mode).
#>
[CmdletBinding()]
param(
    [int]    $Port  = 0,            # 0 = unspecified; resolved below
    [string] $Host_ = '',           # '' = unspecified; resolved below
    [string] $Env   = 'conduix',
    [switch] $Reload
)

$ErrorActionPreference = 'Stop'

# Read a KEY=value from the project-root .env (pydantic reads this file too, but
# PowerShell doesn't auto-load it, so we parse it here to keep the script in sync).
function Get-DotEnvValue([string]$key) {
    $envFile = Join-Path $PSScriptRoot '.env'
    if (-not (Test-Path $envFile)) { return $null }
    foreach ($line in Get-Content $envFile) {
        if ($line -match "^\s*$([regex]::Escape($key))\s*=\s*(.+?)\s*$") {
            return $Matches[1].Trim('"', "'")
        }
    }
    return $null
}

function Resolve-Port {
    if ($Port -ne 0) { return $Port }                                 # explicit -Port
    if ($env:CONDUIX_PORT) { return [int]$env:CONDUIX_PORT }           # shell env
    $fromFile = Get-DotEnvValue 'CONDUIX_PORT'
    if ($fromFile -and $fromFile -match '^\d+$') { return [int]$fromFile }  # .env
    return 8766
}

function Resolve-Host {
    if ($Host_) { return $Host_ }                                     # explicit -Host_
    if ($env:CONDUIX_HOST) { return $env:CONDUIX_HOST }               # shell env
    $fromFile = Get-DotEnvValue 'CONDUIX_HOST'
    if ($fromFile) { return $fromFile }                               # .env
    return '127.0.0.1'
}

$Port  = Resolve-Port
$Host_ = Resolve-Host

function Stop-OnPort([int]$p) {
    $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Host "[start] port $p is free" -ForegroundColor DarkGray
        return
    }
    foreach ($procId in ($conns.OwningProcess | Sort-Object -Unique)) {
        $name = (Get-Process -Id $procId -ErrorAction SilentlyContinue).ProcessName
        Write-Host "[start] killing PID $procId ($name) and its child tree on port $p" -ForegroundColor Yellow
        # /T kills the whole tree, so the Codex app-server child doesn't linger
        # as an orphan holding the ChatGPT session.
        & taskkill.exe /PID $procId /T /F 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0 -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
            Write-Warning "[start] could not stop PID $procId (exit $LASTEXITCODE)"
        }
    }
    # Give Windows a moment to release the socket
    $deadline = (Get-Date).AddSeconds(5)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)) {
            return
        }
        Start-Sleep -Milliseconds 200
    }
    Write-Warning "[start] port $p still in use after 5s, attempting start anyway"
}

if (-not (Test-Path (Join-Path $PSScriptRoot 'conduix\app.py'))) {
    Write-Host "[start] conduix\app.py not found - the server hasn't been built yet (see BRIEF.md section 6)." -ForegroundColor Red
    exit 1
}

Stop-OnPort -p $Port

# Billing guard: never let an API key reach the Codex child process. Env vars
# are process-wide, so stash them and put them back in `finally` to leave the
# caller's shell as it was.
$stashed = @{}
foreach ($k in 'OPENAI_API_KEY', 'CODEX_API_KEY') {
    $v = [Environment]::GetEnvironmentVariable($k, 'Process')
    if ($v) {
        Write-Host "[start] unsetting $k for the server (bills to ChatGPT plan instead)" -ForegroundColor DarkYellow
        $stashed[$k] = $v
        [Environment]::SetEnvironmentVariable($k, $null, 'Process')
    }
}

$uvicornArgs = @('conduix.app:app', '--host', $Host_, '--port', $Port)
if ($Reload) { $uvicornArgs += '--reload' }

Write-Host "[start] starting Conduix on http://${Host_}:${Port}  (docs: /docs, health: /health)" -ForegroundColor Green
Push-Location $PSScriptRoot
try {
    & conda run -n $Env --no-capture-output python -m uvicorn @uvicornArgs
} finally {
    Pop-Location
    foreach ($k in $stashed.Keys) {
        [Environment]::SetEnvironmentVariable($k, $stashed[$k], 'Process')
    }
}

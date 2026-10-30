<# RAM-SENSITIVE. Restart Windows immediately beforehand; separate approval. #>
[CmdletBinding()]
param([switch]$Run,[switch]$ApproveOneDecode,[string]$HarnessCommit)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
if(-not $Run){Write-Output 'DRY_RUN: no WSL, model, neural marker or CPU tail. RAM-SENSITIVE; execution HOLD.';exit 0}
if(-not $ApproveOneDecode -or $HarnessCommit -notmatch '^[a-f0-9]{40}$'){throw 'Separate one-decode approval and exact published commit required'}
$python='D:\AI\Trellis2AMD\venv\Scripts\python.exe'
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'){throw 'Qualified Windows interpreter differs'}
Write-Output 'RAM-SENSITIVE: restart Windows immediately before this run; keep Codex/ChatGPT/browsers closed. Exactly one decode; no retry.'
& $python -B -I -S (Join-Path $PSScriptRoot 'amd_closed_solid_recovery.py') --run --approved-one-decode --harness-commit $HarnessCommit
exit $LASTEXITCODE

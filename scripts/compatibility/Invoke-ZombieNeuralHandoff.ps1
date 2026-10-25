<# User-controlled single attempt. Default dry run; never invoked with -Run by Codex. #>
[CmdletBinding()]
param([switch]$Run,[string]$ExpectedCommit)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$python='D:\AI\Trellis2AMD\venv\Scripts\python.exe'
$worker=Join-Path $PSScriptRoot 'amd_zombie_neural_handoff.py'
if(-not $Run) {
    Write-Output 'DRY_RUN: no child, memory query, marker or WSL. -Run requires an exact published -ExpectedCommit.'
    exit 0
}
if($ExpectedCommit -notmatch '^[0-9a-f]{40}$'){throw 'Exact published -ExpectedCommit required'}
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14') {
    throw 'Original qualified interpreter identity differs'
}
& $python -B -I -S $worker --run --harness-commit $ExpectedCommit
exit $LASTEXITCODE

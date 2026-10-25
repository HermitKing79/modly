<# One user-controlled Zombie D generation. Default is no-work dry run. #>
[CmdletBinding()]
param([switch]$Run,[string]$ExpectedCommit)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not $Run){Write-Output 'DRY_RUN: no process, memory reading, model, marker or WSL. -Run requires exact -ExpectedCommit.';exit 0}
if($ExpectedCommit -notmatch '^[0-9a-f]{40}$'){throw 'Exact published -ExpectedCommit required'}
$python='D:\AI\Trellis2AMD\venv\Scripts\python.exe'
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'){
    throw 'Qualified interpreter identity differs'
}
& $python -B -I -S (Join-Path $PSScriptRoot 'amd_zombie_d_handoff.py') --run --harness-commit $ExpectedCommit
exit $LASTEXITCODE

<# One user-controlled CPU render recovery. Default performs no work. #>
[CmdletBinding()]
param([switch]$Run,[string]$ExpectedCommit)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not $Run){Write-Output 'DRY_RUN: no child, model, WSL, extraction or rendering. -Run requires exact -ExpectedCommit.';exit 0}
if($ExpectedCommit -notmatch '^[0-9a-f]{40}$'){throw 'Exact published -ExpectedCommit required'}
$python='D:\AI\Trellis2AMD\venv\Scripts\python.exe'
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'){
    throw 'Qualified interpreter identity differs'
}
& $python -B -I -S (Join-Path $PSScriptRoot 'amd_zombie_c_render_recovery.py') --run --harness-commit $ExpectedCommit
exit $LASTEXITCODE

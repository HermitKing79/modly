<# Execution HOLD until separate approval. Default starts no child. #>
[CmdletBinding()]
param([switch]$Run,[switch]$ApproveOneNeuralAttempt,[string]$ExpectedCommit)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not $Run){Write-Output 'DRY_RUN: no process, RAM reading, WSL, model or marker. Separate approval required.';exit 0}
if(-not $ApproveOneNeuralAttempt){throw 'Execution HOLD: separate approval for exactly one neural attempt required'}
if($ExpectedCommit -notmatch '^[0-9a-f]{40}$'){throw 'Exact published -ExpectedCommit required'}
$python='D:\AI\Trellis2AMD\venv\Scripts\python.exe'
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'){throw 'Qualified interpreter identity differs'}
& $python -B -I -S (Join-Path $PSScriptRoot 'amd_closed_solid_handoff.py') --run --approved-one-attempt --harness-commit $ExpectedCommit
exit $LASTEXITCODE

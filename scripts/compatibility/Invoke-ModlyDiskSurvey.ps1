<# Preparation only until separate approval. Default launches nothing. #>
[CmdletBinding()]
param([switch]$Run,[switch]$ApproveReadOnlySurvey,[string]$ExpectedCommit)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not $Run){Write-Output 'DRY_RUN: no child, RAM admission, WSL or survey. Separate approval required.';exit 0}
if(-not $ApproveReadOnlySurvey){throw 'Execution HOLD: separate approval for one read-only disk survey required'}
if($ExpectedCommit -notmatch '^[0-9a-f]{40}$'){throw 'Exact published -ExpectedCommit required'}
$python='D:\AI\Trellis2AMD\venv\Scripts\python.exe'
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14'){throw 'Qualified interpreter differs'}
& $python -B -I -S (Join-Path $PSScriptRoot 'amd_wsl_disk_survey.py') --run --approve-survey --commit $ExpectedCommit
exit $LASTEXITCODE

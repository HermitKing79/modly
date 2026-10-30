<# Read-only Phase 6C handoff; separate execution approval. Default inert. #>
[CmdletBinding()]
param([switch]$Run,[switch]$ApproveCapacityCheck,[string]$ExpectedCommit)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
if(-not $Run){Write-Output 'DRY_RUN: no admission, disk access or WSL. Execution HOLD.';exit 0}
if(-not $ApproveCapacityCheck -or $ExpectedCommit -notmatch '^[a-f0-9]{40}$'){throw 'Separate capacity-check approval and exact commit required'}
$python='C:\Users\wattz\AppData\Local\Programs\Python\Python312\python.exe'
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a'){throw 'Base Python differs'}
& $python -B -I -S (Join-Path $PSScriptRoot 'amd_wsl_post_resize.py') --run --approve-capacity-check --commit $ExpectedCommit
exit $LASTEXITCODE

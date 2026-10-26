<# Proposed operator-only profile; no work by default. Separate user approval required. #>
[CmdletBinding()]
param([switch]$Run,[switch]$ApproveOperatorProfileV3,[string]$ExpectedCommit)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not $Run){Write-Output 'DRY_RUN: operator-only v3 pending separate user approval. No RAM admission, WSL, install, download, model or GPU work.';exit 0}
if(-not $ApproveOperatorProfileV3){throw 'Separate approval required for proposed operator-only RAM profile; no execution'}
if($ExpectedCommit -notmatch '^[0-9a-f]{40}$'){throw 'Exact published ExpectedCommit required'}
$python='C:\Users\wattz\AppData\Local\Programs\Python\Python312\python.exe'
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a'){throw 'Qualified standalone interpreter differs'}
& $python -B -I -S (Join-Path $PSScriptRoot 'amd_pixal3d_operator_host_v3.py') --run --approve-operator-profile-v3 --commit $ExpectedCommit
exit $LASTEXITCODE

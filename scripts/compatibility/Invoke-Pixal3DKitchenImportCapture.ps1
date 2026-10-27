<# Separately approved one-shot import forensics. Does not release the operator hold. #>
[CmdletBinding()]
param([switch]$CaptureKitchenImportOnly,[switch]$ApproveKitchenImportCapture,[string]$ExpectedCommit)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not $CaptureKitchenImportOnly){Write-Output 'DRY_RUN: import capture pending separate approval; no admission, WSL, GPU, downloads or installation. Operator hold retained.';exit 0}
if(-not $ApproveKitchenImportCapture -or $ExpectedCommit -notmatch '^[a-f0-9]{40}$'){throw 'Separate import-capture approval and exact reviewed commit required'}
$python='C:\Users\wattz\AppData\Local\Programs\Python\Python312\python.exe'
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a'){throw 'Qualified Python differs'}
& $python -B -I -S (Join-Path $PSScriptRoot 'amd_pixal3d_kitchen_capture.py') --capture-kitchen-import --approve-kitchen-import-capture --commit $ExpectedCommit
exit $LASTEXITCODE

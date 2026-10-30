<# Attended one-shot storage operation. Default does nothing. #>
[CmdletBinding()]
param([switch]$Run,[switch]$ApproveResize,[switch]$AcceptNativeCriticalSection,[string]$ExpectedCommit)
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
if(-not $Run){Write-Output 'DRY_RUN: no child, disk access, admission or WSL. Separate approval required.';exit 0}
if(-not $ApproveResize -or -not $AcceptNativeCriticalSection){throw 'Execution HOLD: separate resize approval and accepted critical-section policy required'}
if($ExpectedCommit -notmatch '^[a-f0-9]{40}$'){throw 'Exact published commit required'}
$python='C:\Users\wattz\AppData\Local\Programs\Python\Python312\python.exe'
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a'){throw 'Authenticated base Python differs'}
Write-Output 'ATTENDED STORAGE: do not close this console. Faults latch FAIL; an active resize settles naturally.'
& $python -B -I -S (Join-Path $PSScriptRoot 'amd_wsl_resize_operator.py') --run --approve-resize --accept-native-critical-section --commit $ExpectedCommit
exit $LASTEXITCODE

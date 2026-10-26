<# Standalone model-free Pixal3D HIP proof. No-work default; no Codex dependency. #>
[CmdletBinding()]
param([switch]$Run,[string]$ExpectedCommit)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if(-not $Run){
    Write-Output 'DRY_RUN: no admission, WSL, download, environment or GPU work. Live -Run downloads only hash-pinned runtime wheels/source files; no model weights. Close Codex/ChatGPT Desktop first. Exact -ExpectedCommit required.'
    exit 0
}
if($ExpectedCommit -notmatch '^[0-9a-f]{40}$'){throw 'Exact published -ExpectedCommit required'}
$python='C:\Users\wattz\AppData\Local\Programs\Python\Python312\python.exe'
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a'){
    throw 'Qualified standalone Windows interpreter identity differs'
}
& $python -B -I -S (Join-Path $PSScriptRoot 'amd_pixal3d_standalone.py') --run --commit $ExpectedCommit
exit $LASTEXITCODE

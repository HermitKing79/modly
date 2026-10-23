# Invoke once only through supported exec_command require_escalated approval.
# That tool mode is outside the Codex sandbox, not Windows administrator RunAs.
$ErrorActionPreference = 'Stop'
$proofRoot = 'C:\Users\wattz\Documents\ChatGPT\modly\proof'
$privateRoot = 'C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-wsl-upstream-authority-7152764'
$evidenceRoot = Join-Path $proofRoot 'docs\compatibility\amd-wsl-upstream-authority-evidence'
$hostPath = Join-Path $evidenceRoot 'qualification.py'
$hostRelative = 'docs/compatibility/amd-wsl-upstream-authority-evidence/qualification.py'

function Write-ExclusiveUtf8([string] $path, [string] $text) {
    $stream = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($text)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    } finally { $stream.Dispose() }
}

$expected = Get-Content -LiteralPath (Join-Path $privateRoot 'expected-context.json') -Raw | ConvertFrom-Json
$admission = Get-Content -LiteralPath (Join-Path $evidenceRoot 'admission.json') -Raw | ConvertFrom-Json
$pythonPath = $admission.python_executable
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$context = [ordered]@{
    utc = [DateTime]::UtcNow.ToString('o')
    name = $identity.Name
    sid = $identity.User.Value
    administrator_role_enabled = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    is_system = $identity.IsSystem
    is_anonymous = $identity.IsAnonymous
    requested_tool_override = 'require_escalated'
    context_limit = 'Accepted supported tool override plus same-shell token evidence; no independent child-token or sandbox detector.'
}
Write-ExclusiveUtf8 (Join-Path $privateRoot 'outside-context.json') ($context | ConvertTo-Json -Depth 8)
if ($context.administrator_role_enabled -or $context.is_system -or $context.is_anonymous) { throw 'NOT RUN: normal non-administrator-enabled Windows user token required' }
if ($context.sid -ne $expected.sid -or $context.name -ne $expected.name) { throw 'NOT RUN: qualified Windows user identity differs' }
if ((Get-FileHash -LiteralPath $pythonPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.python_sha256) { throw 'NOT RUN: authenticated API interpreter differs' }
if ($admission.python_executable_sha256 -ne $expected.python_sha256) { throw 'NOT RUN: admission interpreter differs' }
$registration = Get-Content -LiteralPath (Join-Path $privateRoot 'registration-before-private.json') -Raw | ConvertFrom-Json
$actualDistro = @(Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss\*' | Where-Object DistributionName -eq 'Modly-AMD-Ubuntu-24.04')
if ($actualDistro.Count -ne 1 -or $actualDistro[0].Version -ne 2 -or $actualDistro[0].DefaultUid -ne $registration.registration.DefaultUid -or $actualDistro[0].BasePath -ne $registration.registration.BasePath) { throw 'NOT RUN: named distro registration changed' }
if ((Get-FileHash -LiteralPath $registration.wsl_path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $admission.wsl_executable_sha256) { throw 'NOT RUN: WSL executable changed' }
$hostPin = $admission.source_pins.PSObject.Properties[$hostRelative].Value
if (-not $hostPin -or (Get-FileHash -LiteralPath $hostPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hostPin) { throw 'NOT RUN: reviewed host source differs' }
if (Test-Path -LiteralPath (Join-Path $privateRoot 'cpu-once')) { throw 'NOT RUN: fresh exclusive attempt required' }

$timer = [Diagnostics.Stopwatch]::StartNew()
# One direct host invocation. Host owns the sole synthetic WSL conditioning invocation and all
# bounded Job/resource/deadline enforcement. No retry or alternative route.
$ErrorActionPreference = 'Continue'
try {
    $lines = & $pythonPath -B -I -S $hostPath 2>&1
    $hostExit = $LASTEXITCODE
} finally { $ErrorActionPreference = 'Stop' }
$timer.Stop()
$console = ($lines | Out-String)
Write-ExclusiveUtf8 (Join-Path $privateRoot 'outside-console.txt') $console
$result = [ordered]@{
    schema = 'modly.wsl-upstream-authority-outside-command.v1'
    utc = [DateTime]::UtcNow.ToString('o')
    host_invocations = 1
    host_exit_code = $hostExit
    host_process_elapsed_seconds = $timer.Elapsed.TotalSeconds
    executable = $pythonPath
    arguments = @('-B', '-I', '-S', $hostPath)
    windows_runas_used = $false
    context_file = 'outside-context.json'
    console_file = 'outside-console.txt'
}
Write-ExclusiveUtf8 (Join-Path $privateRoot 'outside-result.json') ($result | ConvertTo-Json -Depth 8)
Write-Output $console
exit $hostExit

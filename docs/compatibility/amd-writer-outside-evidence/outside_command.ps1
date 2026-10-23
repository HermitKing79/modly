# Exactly one unchanged probe; invoke only through the explicit supported
# sandbox_permissions=require_escalated request. This is not Windows RunAs.
$ErrorActionPreference = 'Stop'
$proofRoot = 'C:\Users\wattz\Documents\ChatGPT\modly\proof'
$privateRoot = 'C:\Users\wattz\Documents\ChatGPT\modly\artifacts\amd-writer-outside-e971fbc'
$pythonPath = 'C:\Users\wattz\Documents\Modly\dependencies\venv\Scripts\python.exe'
$probePath = Join-Path $proofRoot 'docs\compatibility\amd-evidence-writer-evidence\writer_probe.py'

function Write-ExclusiveUtf8([string] $path, [string] $text) {
    $stream = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($text)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    } finally { $stream.Dispose() }
}

$expected = Get-Content -LiteralPath (Join-Path $privateRoot 'expected-context.json') -Raw | ConvertFrom-Json
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
    context_limit = 'Token evidence plus externally recorded accepted tool override; the probe has no independent sandbox detector.'
}
Write-ExclusiveUtf8 (Join-Path $privateRoot 'outside-context.json') ($context | ConvertTo-Json -Depth 8)
if ($context.administrator_role_enabled -or $context.is_system -or $context.is_anonymous) { throw 'NOT RUN: normal non-administrator-enabled user token required' }
if ($context.sid -ne $expected.sid -or $context.name -ne $expected.name) { throw 'NOT RUN: original Windows user identity differs' }
if ((Get-FileHash -LiteralPath $pythonPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.python_sha256) { throw 'NOT RUN: Python identity differs' }
if ((Get-FileHash -LiteralPath $probePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.probe_sha256) { throw 'NOT RUN: unchanged probe identity differs' }
if ([IO.Path]::GetPathRoot($privateRoot) -ne 'C:\') { throw 'NOT RUN: original C: volume required' }
if (@(Get-ChildItem -LiteralPath $privateRoot -Filter 'probe-*' -Force).Count -ne 0) { throw 'NOT RUN: fresh comparison root required' }

$timer = [Diagnostics.Stopwatch]::StartNew()
# No RunAs, Start-Process, Job or alternate process route. The API interpreter
# directly inherits this shell's user token. Preserve stderr without a retry.
$ErrorActionPreference = 'Continue'
try {
    $lines = & $pythonPath -B -I -S $probePath --run --private-root $privateRoot 2>&1
    $probeExit = $LASTEXITCODE
} finally { $ErrorActionPreference = 'Stop' }
$timer.Stop()
$console = ($lines | Out-String)
Write-ExclusiveUtf8 (Join-Path $privateRoot 'outside-console.txt') $console
$result = [ordered]@{
    schema = 'modly.writer-outside-command.v1'
    utc = [DateTime]::UtcNow.ToString('o')
    probe_invocations = 1
    probe_exit_code = $probeExit
    probe_process_elapsed_seconds = $timer.Elapsed.TotalSeconds
    executable = $pythonPath
    arguments = @('-B', '-I', '-S', $probePath, '--run', '--private-root', $privateRoot)
    windows_runas_used = $false
    context_file = 'outside-context.json'
    console_file = 'outside-console.txt'
}
Write-ExclusiveUtf8 (Join-Path $privateRoot 'outside-result.json') ($result | ConvertTo-Json -Depth 8)
Write-Output $console
exit $probeExit

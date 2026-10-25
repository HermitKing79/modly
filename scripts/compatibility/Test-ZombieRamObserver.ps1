<# Actual Windows venv redirector regression; only a harmless 64 MiB allocation. #>
[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$path=Join-Path $PSScriptRoot 'Invoke-ZombieRamDiagnostic.ps1'
$source=[IO.File]::ReadAllText($path)
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseInput($source,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Launcher parse failure'}
foreach($function in $ast.FindAll({param($a) $a -is [Management.Automation.Language.FunctionDefinitionAst]},$false)) {
    . ([scriptblock]::Create($function.Extent.Text))
}
$native=[regex]::Match($source,"(?s)Add-Type -TypeDefinition @'\r?\n(.*?)\r?\n'@").Groups[1].Value
Add-Type -TypeDefinition $native
$python='D:\AI\Trellis2AMD\venv\Scripts\python.exe'
$probe=@'
import ctypes, importlib.util, json, os, sys
spec=importlib.util.spec_from_file_location('memory_probe',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
allocation=bytearray(64*1024**2)
print(json.dumps(dict(pid=os.getpid(),image=sys._base_executable,memory=module.windows_memory())),flush=True)
sys.stdin.readline()
'@
$info=New-Object Diagnostics.ProcessStartInfo
$info.FileName=$python
$encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($probe))
$info.Arguments='-B -I -S -c "import base64;exec(base64.b64decode('''+$encoded+'''))" "'+(Join-Path $PSScriptRoot 'amd_zombie_ram_diagnostic.py')+'"'
$info.UseShellExecute=$false;$info.CreateNoWindow=$true
$info.RedirectStandardInput=$true;$info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
$launcher=New-Object Diagnostics.Process;$launcher.StartInfo=$info
$job=[ZombieRamNative]::OwnedJob();$actual=$null
try {
    if(-not $launcher.Start()){throw 'Probe did not start'}
    if(-not [ZombieRamNative]::AssignProcessToJobObject($job,$launcher.Handle)){throw 'Probe job assignment failed'}
    $line=$launcher.StandardOutput.ReadLineAsync()
    if(-not $line.Wait(15000)){throw 'Probe handshake timed out'}
    $identity=$line.GetAwaiter().GetResult()|ConvertFrom-Json
    if($null -eq $identity){throw ('Probe failed: '+$launcher.StandardError.ReadToEnd())}
    $old=Process-Memory $launcher
    # On the old implementation this deliberately measures the redirector and fails.
    if(Get-Command Resolve-WorkerProcess -ErrorAction SilentlyContinue){$actual=Resolve-WorkerProcess $identity $job $launcher}
    else {$actual=[Diagnostics.Process]::GetProcessById($launcher.Id)}
    $measured=Process-Memory $actual
    if($actual.Id -ne $identity.pid -or $actual.Id -eq $launcher.Id){throw 'Observer tracked the venv redirector, not the interpreter'}
    # Both native APIs query the same held allocation. Allow normal small runtime drift.
    if([Math]::Abs($measured.working_set_bytes-$identity.memory.working_set_bytes) -gt 4MB -or
       [Math]::Abs($measured.private_commit_bytes-$identity.memory.private_commit_bytes) -gt 4MB){throw 'Native observer/self memory disagreement'}
    if($measured.working_set_bytes -lt 64MB -or $measured.private_commit_bytes -lt 64MB){throw 'Held allocation was missed'}
    $foreign=[pscustomobject]@{pid=$PID;image=[Diagnostics.Process]::GetCurrentProcess().MainModule.FileName}
    $rejected=$false
    try {$bad=Resolve-WorkerProcess $foreign $job $launcher} catch {$rejected=$true}
    if(-not $rejected){throw 'Foreign process admitted'}
    $wrong=[pscustomobject]@{pid=$identity.pid;image=$python}
    $rejected=$false
    try {$bad=Resolve-WorkerProcess $wrong $job $launcher} catch {$rejected=$true}
    if(-not $rejected){throw 'Mismatched executable admitted'}
    [ordered]@{classification='PASS';launcher_pid=$launcher.Id;worker_pid=$actual.Id;launcher_memory=$old;
        worker_memory=$measured;self_memory=$identity.memory;foreign_pid_rejected=$true;wrong_image_rejected=$true;
        full_CPU_preparation=$false;WSL_launched=$false;neural_forwards=0}|ConvertTo-Json -Depth 5
    $launcher.StandardInput.WriteLine('EXIT');$launcher.StandardInput.Flush()
    if(-not $launcher.WaitForExit(5000) -or $launcher.ExitCode -ne 0){throw 'Probe did not exit naturally'}
} finally {
    [void][ZombieRamNative]::CloseHandle($job)
    if($null -ne $actual){$actual.Dispose()}
    $launcher.Dispose()
}

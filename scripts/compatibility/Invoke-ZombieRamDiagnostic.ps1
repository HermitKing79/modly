<# CPU-only observer. Default is dry-run. Only the user should invoke -Run after closing apps. #>
[CmdletBinding()]
param([switch]$Run, [switch]$SelfTest, [string]$ExpectedCommit)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$script:Floor = [long]23622320128
$script:Preferred = [long]25769803776
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$python = 'D:\AI\Trellis2AMD\venv\Scripts\python.exe'
$worker = Join-Path $PSScriptRoot 'amd_zombie_ram_diagnostic.py'

function Get-Admission {
    param([scriptblock]$Read, [scriptblock]$Sleep, [scriptblock]$Record)
    $rows = @()
    for ($index=1; $index -le 3; $index++) {
        $row = & $Read
        $rows += $row
        & $Record $row
        if ($row.available_bytes -lt $script:Floor) { return @{passed=$false; readings=$rows} }
        if ($index -lt 3) { & $Sleep }
    }
    return @{passed=$true; readings=$rows}
}

function Confirm-PhaseBoundary {
    param([string]$Stage,[scriptblock]$Observe,[scriptblock]$Check,[scriptblock]$Snapshot,[scriptblock]$Continue)
    $row=& $Observe $Stage
    & $Check $row
    & $Snapshot $Stage
    & $Continue
}
function Confirm-ResourceValues {
    param([long]$Available,[double]$Elapsed,[long]$DFree,[long]$EvidenceFree,[string]$Stage)
    if($Available -lt $script:Floor) {throw ('Below 22 GiB at '+$Stage+': '+$Available+' bytes')}
    if($Elapsed -gt 480) {throw 'Original 480-second host deadline reached'}
    if($DFree -lt 50GB) {throw 'Original 50 GiB D: floor failed'}
    if($EvidenceFree -lt 2GB) {throw 'Original 2 GiB evidence-volume floor failed'}
}
function Stop-OwnedDiagnostic {
    param($Process,[IntPtr]$Job,[scriptblock]$CloseJob)
    if($Job -ne [IntPtr]::Zero) { & $CloseJob $Job }
    if(-not $Process.WaitForExit(5000)) {
        $Process.Kill()
        if(-not $Process.WaitForExit(5000)) {throw 'Owned diagnostic did not settle after forced cleanup'}
    }
}
function Confirm-ChildReceipt {
    param([int]$ExitCode,$Receipt)
    if($ExitCode -ne 0 -or $null -eq $Receipt -or $Receipt.classification -ne 'CPU_PREPARATION_COMPLETE' -or
        @($Receipt.forbidden_modules_present).Count -ne 0 -or $Receipt.WSL_launched -ne $false -or
        $Receipt.weights_loaded -ne $false -or $Receipt.neural_attempt_reserved -ne $false -or $Receipt.neural_forwards -ne 0) {
        throw 'Complete CPU-only child receipt missing or invalid'
    }
}
function Read-BoundedGitResult {
    param($Process)
    $output=$Process.StandardOutput.ReadToEndAsync()
    $errorOutput=$Process.StandardError.ReadToEndAsync()
    if(-not $Process.WaitForExit(15000)) {throw 'Read-only Git exceeded its original 15-second bound'}
    if(-not $output.Wait(5000) -or -not $errorOutput.Wait(5000)) {throw 'Read-only Git pipes did not settle'}
    if($Process.ExitCode -ne 0) {throw ('Read-only Git failed: '+$errorOutput.GetAwaiter().GetResult())}
    return $output.GetAwaiter().GetResult()
}
function Invoke-BoundedGit {
    param([string[]]$ReadArguments)
    if(-not (($ReadArguments.Count -eq 2 -and $ReadArguments[0] -eq 'rev-parse' -and $ReadArguments[1] -eq 'HEAD') -or
        ($ReadArguments.Count -eq 2 -and $ReadArguments[0] -eq 'show' -and $ReadArguments[1] -match '^[0-9a-f]{40}:[A-Za-z0-9_./-]+$'))) {
        throw 'Unapproved launcher Git command'
    }
    $info=New-Object Diagnostics.ProcessStartInfo
    $info.FileName=(Get-Command git.exe -CommandType Application | Select-Object -First 1).Source
    $all=@('--no-pager','-C',$repo)+$ReadArguments
    $info.Arguments=($all|ForEach-Object {'"'+$_+'"'}) -join ' '
    $info.UseShellExecute=$false;$info.CreateNoWindow=$true
    $info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
    $process=New-Object Diagnostics.Process; $process.StartInfo=$info
    $owned=[ZombieRamNative]::OwnedJob()
    $started=$false
    try {
        if(-not $process.Start()){throw 'Read-only Git did not start'}
        $started=$true
        if(-not [ZombieRamNative]::AssignProcessToJobObject($owned,$process.Handle)){throw 'Read-only Git containment failed'}
        return Read-BoundedGitResult $process
    } finally {
        [void][ZombieRamNative]::CloseHandle($owned)
        if($started -and -not $process.HasExited) {if(-not $process.WaitForExit(5000)){$process.Kill()}}
        $process.Dispose()
    }
}
function Resolve-SettlementOutcome {
    param([string]$Outcome,$Failure,[object[]]$Errors)
    if($Errors.Count -gt 0) {
        if($null -eq $Failure){$Failure='Required settlement/recovery evidence incomplete'}
        return @{outcome='INCOMPLETE_SETTLEMENT';failure=$Failure}
    }
    return @{outcome=$Outcome;failure=$Failure}
}

if ($SelfTest) {
    if ($Run) { throw 'SelfTest cannot execute the real diagnostic' }
    $script:reads=0; $script:sleeps=0
    $pass=Get-Admission { $script:reads++; @{available_bytes=23622320128} } { $script:sleeps++ } {param($row)}
    if (-not $pass.passed -or $script:reads -ne 3 -or $script:sleeps -ne 2) { throw '22 GiB equality admission test failed' }
    $script:reads=0; $script:sleeps=0
    $fail=Get-Admission { $script:reads++; @{available_bytes=23622320127} } { $script:sleeps++ } {param($row)}
    if ($fail.passed -or $script:reads -ne 1 -or $script:sleeps -ne 0) { throw 'Early rejection test failed' }
    $script:reads=0
    $middle=Get-Admission { $script:reads++; @{available_bytes=$(if($script:reads -eq 2){23622320127}else{25769803776})} } {} {param($row)}
    if ($middle.passed -or $script:reads -ne 2) { throw 'Second-reading rejection test failed' }
    $script:order=@()
    Confirm-PhaseBoundary 'imports/before' {param($stage) $script:order+='observe'; @{available_bytes=23622320128}} {
        param($row) $script:order+='check'; Confirm-ResourceValues $row.available_bytes 1 50GB 2GB 'imports/before'
    } {param($stage) $script:order+='snapshot'} {$script:order+='continue'}
    if(($script:order -join ',') -ne 'observe,check,snapshot,continue'){throw 'Phase permit ordering failed'}
    $script:continued=$false; $rejected=$false
    try {
        Confirm-PhaseBoundary 'checkpoint/before' {param($stage) @{available_bytes=23622320127}} {
            param($row) Confirm-ResourceValues $row.available_bytes 1 50GB 2GB 'checkpoint/before'
        } {param($stage)} {$script:continued=$true}
    } catch {$rejected=$true}
    if(-not $rejected -or $script:continued){throw 'Low RAM permitted another phase'}
    foreach($resource in @(@(23622320128,481,50GB,2GB),@(23622320128,1,(50GB-1),2GB),@(23622320128,1,50GB,(2GB-1)))) {
        $rejected=$false
        try {Confirm-ResourceValues $resource[0] $resource[1] $resource[2] $resource[3] 'synthetic'} catch {$rejected=$true}
        if(-not $rejected){throw 'Resource failure admitted'}
    }
    $fake=New-Object PSObject -Property @{killed=$false;waits=0}
    $fake|Add-Member ScriptMethod WaitForExit {param($ms) $this.waits++; return $this.waits -gt 1}
    $fake|Add-Member ScriptMethod Kill {$this.killed=$true}
    $script:closed=@()
    Stop-OwnedDiagnostic $fake ([IntPtr]123) {param($job) $script:closed+=$job.ToInt64()}
    if(-not $fake.killed -or $fake.waits -ne 2 -or $script:closed.Count -ne 1 -or $script:closed[0] -ne 123){throw 'Owned cleanup test failed'}
    $receipt=[pscustomobject]@{classification='CPU_PREPARATION_COMPLETE';forbidden_modules_present=@();WSL_launched=$false;
        weights_loaded=$false;neural_attempt_reserved=$false;neural_forwards=0}
    Confirm-ChildReceipt 0 $receipt
    $receipt.weights_loaded=$true; $rejected=$false
    try {Confirm-ChildReceipt 0 $receipt} catch {$rejected=$true}
    if(-not $rejected){throw 'Unsafe completion receipt admitted'}
    $script:waitMilliseconds=0
    $task=New-Object PSObject
    $task|Add-Member ScriptMethod Wait {param($ms) return $true}
    $task|Add-Member ScriptMethod GetAwaiter {return $this}
    $task|Add-Member ScriptMethod GetResult {return 'synthetic Git output'}
    $reader=New-Object PSObject -Property @{task=$task}
    $reader|Add-Member ScriptMethod ReadToEndAsync {return $this.task}
    $fakeGit=New-Object PSObject -Property @{StandardOutput=$reader;StandardError=$reader;ExitCode=0;success=$false}
    $fakeGit|Add-Member ScriptMethod WaitForExit {param($ms) $script:waitMilliseconds=$ms; return $this.success}
    $rejected=$false
    try {Read-BoundedGitResult $fakeGit} catch {$rejected=$true}
    if(-not $rejected -or $script:waitMilliseconds -ne 15000){throw 'Git timeout regression failed'}
    $fakeGit.success=$true
    if((Read-BoundedGitResult $fakeGit) -ne 'synthetic Git output'){throw 'Bounded Git output regression failed'}
    $settled=Resolve-SettlementOutcome 'CPU_PREPARATION_COMPLETE' $null @('synthetic recovery failure')
    if($settled.outcome -eq 'CPU_PREPARATION_COMPLETE' -or $null -eq $settled.failure){throw 'Settlement failure incorrectly reported success'}
    Write-Output 'PASS: 14 synthetic admission/permit/resource/owned-cleanup/receipt/Git-timeout/settlement scenarios; no RAM query, child, WSL or model execution.'
    exit 0
}
if (-not $Run) {
    [ordered]@{mode='DRY_RUN';real_diagnostic=$false;python=$python;worker=$worker;
        python_flags=@('-B','-I','-S');admission_bytes=$script:Floor;preferred_only_bytes=$script:Preferred;
        samples=3;sample_spacing_seconds=4;observer_interval_ms=200;host_deadline_seconds=480;
        source_base='28f4f0163833d59d861404719e6b2cb042083f9e';requires_explicit_Run=$true;
        stages=@('python-startup','imports','construct_runtime/constructor-source-lineage','validate_lineage',
            'verify_sources','stream_checkpoint','metadata_checks','child-exit','post-exit-recovery')
    } | ConvertTo-Json -Depth 4
    exit 0
}

# No live APIs, process creation or evidence writes occur above this point.
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ZombieRamNative {
 [StructLayout(LayoutKind.Sequential)] public class Memory {
  public uint length=(uint)Marshal.SizeOf(typeof(Memory)),load;
  public ulong total,available,pageTotal,pageAvailable,virtualTotal,virtualAvailable,extended;
 }
 [DllImport("kernel32.dll",SetLastError=true)] [return:MarshalAs(UnmanagedType.Bool)]
 public static extern bool GlobalMemoryStatusEx([In,Out] Memory m);
 [DllImport("kernel32.dll")] public static extern IntPtr CreateJobObject(IntPtr a,string name);
 [DllImport("kernel32.dll",SetLastError=true)] public static extern bool AssignProcessToJobObject(IntPtr job,IntPtr process);
 [DllImport("kernel32.dll",SetLastError=true)] public static extern bool IsProcessInJob(IntPtr process,IntPtr job,out bool member);
 [DllImport("kernel32.dll",SetLastError=true)] public static extern bool SetInformationJobObject(IntPtr job,int type,IntPtr info,uint length);
 [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr handle);
 [StructLayout(LayoutKind.Sequential)] public struct Basic {
  public long processTime,jobTime; public uint flags; public UIntPtr minWS,maxWS;
  public uint activeLimit; public UIntPtr affinity; public uint priority,scheduling;
 }
 [StructLayout(LayoutKind.Sequential)] public struct Io { public ulong a,b,c,d,e,f; }
 [StructLayout(LayoutKind.Sequential)] public struct Extended {
  public Basic basic; public Io io; public UIntPtr processMemory,jobMemory,peakProcess,peakJob;
 }
 public static IntPtr OwnedJob() {
  IntPtr job=CreateJobObject(IntPtr.Zero,null);
  if(job==IntPtr.Zero) throw new System.ComponentModel.Win32Exception();
  Extended info=new Extended(); info.basic.flags=0x2000; // KILL_ON_JOB_CLOSE
  int size=Marshal.SizeOf(info); IntPtr buffer=Marshal.AllocHGlobal(size);
  try {
   Marshal.StructureToPtr(info,buffer,false);
   if(!SetInformationJobObject(job,9,buffer,(uint)size)) throw new System.ComponentModel.Win32Exception();
  } catch { CloseHandle(job); throw; } finally { Marshal.FreeHGlobal(buffer); }
  return job;
 }
}
'@

function Read-PhysicalMemory {
    $m=New-Object ZombieRamNative+Memory
    if (-not [ZombieRamNative]::GlobalMemoryStatusEx($m)) { throw 'GlobalMemoryStatusEx failed' }
    return [ordered]@{timestamp_utc=[DateTimeOffset]::UtcNow.ToString('o');available_bytes=[long]$m.available;
        total_physical_bytes=[long]$m.total;memory_load_percent=$m.load}
}
function Write-NewJson {
    param([string]$Path,$Value)
    $file=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $writer=New-Object IO.StreamWriter($file,(New-Object Text.UTF8Encoding($false)))
    try { $writer.WriteLine(($Value|ConvertTo-Json -Depth 12)); $writer.Flush(); $file.Flush($true) } finally { $writer.Dispose() }
}
function New-Journal {
    param([string]$Path)
    $file=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $writer=New-Object IO.StreamWriter($file,(New-Object Text.UTF8Encoding($false)))
    $writer.AutoFlush=$true
    return $writer
}
function Process-Memory {
    param($Process)
    if ($null -eq $Process) { return $null }
    try {
        $Process.Refresh()
        if($Process.HasExited){return @{pid=$Process.Id;measurement_status='exited'}}
        return @{pid=$Process.Id;process_name=$Process.ProcessName;start_utc=$Process.StartTime.ToUniversalTime().ToString('o');working_set_bytes=$Process.WorkingSet64;
            private_commit_bytes=$Process.PrivateMemorySize64;os_peak_working_set_bytes=$Process.PeakWorkingSet64}
    } catch { return @{pid=$Process.Id;measurement_error=$_.Exception.Message} }
}
function Resolve-WorkerProcess {
    param($Identity,[IntPtr]$Job,$Launcher)
    if($Job -eq [IntPtr]::Zero -or $Identity.pid -isnot [int] -or $Identity.pid -le 0 -or
        -not [IO.Path]::IsPathRooted($Identity.image)){throw 'Invalid interpreter identity'}
    $process=[Diagnostics.Process]::GetProcessById($Identity.pid)
    try {
        $member=$false
        if(-not [ZombieRamNative]::IsProcessInJob($process.Handle,$Job,[ref]$member) -or -not $member){throw 'Interpreter is outside owned diagnostic Job'}
        if($process.StartTime.ToUniversalTime() -lt $Launcher.StartTime.ToUniversalTime() -or
            $process.MainModule.FileName -ine [IO.Path]::GetFullPath($Identity.image)){throw 'Interpreter start/image identity differs'}
        return $process
    } catch {$process.Dispose();throw}
}
function Process-Snapshot {
    param([string]$Stage)
    $errors=@(); $rows=@()
    foreach($process in [Diagnostics.Process]::GetProcesses()) {
        try {
            $rows+=@{pid=$process.Id;name=$process.ProcessName;start_utc=$process.StartTime.ToUniversalTime().ToString('o');
                working_set_bytes=$process.WorkingSet64;private_commit_bytes=$process.PrivateMemorySize64}
        } catch { $errors+=@{pid=$process.Id;error=$_.Exception.GetType().Name} }
        finally { $process.Dispose() }
    }
    $processJournal.WriteLine((@{timestamp_utc=[DateTimeOffset]::UtcNow.ToString('o');stage=$Stage;processes=$rows;errors=$errors;
        scope='Best-effort visible processes; inaccessible processes and shared pages prevent exclusive physical accounting.'}|ConvertTo-Json -Depth 6 -Compress))
}
function Observe {
    param([string]$Stage,[string]$Kind='periodic')
    $m=Read-PhysicalMemory
    $row=@{timestamp_utc=$m.timestamp_utc;elapsed_seconds=$clock.Elapsed.TotalSeconds;stage=$Stage;kind=$Kind;
        available_bytes=$m.available_bytes;observer=(Process-Memory $observer);child=(Process-Memory $interpreter);
        launcher=(Process-Memory $child)}
    if($null -ne $interpreter -and -not $interpreter.HasExited -and
        ($null -eq $row.child.working_set_bytes -or $null -eq $row.child.private_commit_bytes)) {throw 'Live interpreter memory measurement failed'}
    if($null -ne $script:previousObservation) {
        $row.available_change_since_previous_bytes=$row.available_bytes-$script:previousObservation.available_bytes
    }
    if($null -ne $baseline) { $row.available_change_since_before_python_bytes=$row.available_bytes-$baseline.available_bytes }
    $script:previousObservation=$row
    $telemetry.WriteLine(($row|ConvertTo-Json -Depth 5 -Compress))
    if($m.available_bytes -lt $script:minimum) { $script:minimum=$m.available_bytes }
    if($null -ne $row.child -and $row.child.ContainsKey('working_set_bytes')) {
        $script:peakWS=[Math]::Max($script:peakWS,$row.child.working_set_bytes)
        $script:peakPrivate=[Math]::Max($script:peakPrivate,$row.child.private_commit_bytes)
    }
    return $row
}
function Check-Resources {
    param($Row)
    Confirm-ResourceValues $Row.available_bytes $clock.Elapsed.TotalSeconds (New-Object IO.DriveInfo('D:\')).AvailableFreeSpace (New-Object IO.DriveInfo([IO.Path]::GetPathRoot($evidence))).AvailableFreeSpace $Row.stage
}

$identity=New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if($identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Run as the normal Windows user, without elevation' }
if(-not [IO.File]::Exists($python) -or -not [IO.File]::Exists($worker)) { throw 'Original Python or standalone worker missing' }
if($ExpectedCommit -notmatch '^[0-9a-f]{40}$') {throw '-Run requires the exact published -ExpectedCommit from the handoff'}
$token=[Guid]::NewGuid().ToString('N')
$evidence=Join-Path ([IO.Path]::GetFullPath((Join-Path $repo '..\artifacts'))) ('amd-zombie-standalone-ram-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')+'-'+$token)
foreach($parent in @($repo,(Split-Path $repo), (Split-Path $evidence))) {
    if((Get-Item -LiteralPath $parent -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Redirected repository/evidence parent denied' }
}
New-Item -ItemType Directory -Path $evidence -ErrorAction Stop | Out-Null
Write-Host ('Private evidence: '+$evidence)
$telemetry=New-Journal (Join-Path $evidence 'observer.jsonl')
$processJournal=New-Journal (Join-Path $evidence 'processes-private.jsonl')
$stdout=New-Journal (Join-Path $evidence 'child.stdout.txt')
$observer=[Diagnostics.Process]::GetCurrentProcess(); $child=$null; $interpreter=$null; $job=[IntPtr]::Zero
$clock=[Diagnostics.Stopwatch]::StartNew(); $script:minimum=[long]::MaxValue
$script:previousObservation=$null
$script:peakWS=[long]0; $script:peakPrivate=[long]0
$stage='admission'; $outcome='FAILED'; $failure=$null; $forced=$false; $childExit=$null
$exitObservation=$null; $recovery=@(); $baseline=$null; $stderrTask=$null; $childStarted=$false
$settlementErrors=@()
try {
    $stage='implementation-verification'
    $head=Invoke-BoundedGit @('rev-parse','HEAD')
    if($head.Trim() -ne $ExpectedCommit) {throw 'Checkout differs from the exact published implementation'}
    foreach($relative in @('scripts/compatibility/Invoke-ZombieRamDiagnostic.ps1','scripts/compatibility/amd_zombie_ram_diagnostic.py','scripts/compatibility/amd_zombie_ram_sources.json')) {
        $committed=Invoke-BoundedGit @('show',($ExpectedCommit+':'+$relative))
        $working=[IO.File]::ReadAllText((Join-Path $repo $relative))
        if($committed.Replace("`r`n","`n") -cne $working.Replace("`r`n","`n")) {throw ('Diagnostic implementation differs from published source: '+$relative)}
    }
    $admission=Get-Admission {Read-PhysicalMemory} {Start-Sleep -Seconds 4} {param($row)
        $script:minimum=[Math]::Min($script:minimum,$row.available_bytes)
        $telemetry.WriteLine((@{kind='admission';stage='before-python';sample=$row}|ConvertTo-Json -Depth 4 -Compress))
        Write-Host ('Available RAM: '+[Math]::Round($row.available_bytes/1GB,6)+' GiB')
    }
    Write-NewJson (Join-Path $evidence 'admission.json') @{token=$token;readings=$admission.readings;
        required_available_bytes=$script:Floor;preferred_only_bytes=$script:Preferred;observer_pid=$PID;implementation_commit=$ExpectedCommit}
    Process-Snapshot 'admission'
    if(-not $admission.passed) { $outcome='NO_GO_ADMISSION'; throw 'Fresh three-reading 22 GiB admission failed; no Python child launched' }
    $stage='before-python-start'; $baseline=Observe $stage 'before'; Check-Resources $baseline
    if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14') { throw 'Original Python executable identity changed' }
    $job=[ZombieRamNative]::OwnedJob()
    $info=New-Object Diagnostics.ProcessStartInfo
    $info.FileName=$python
    $info.Arguments='-B -I -S "'+$worker+'" --evidence "'+$evidence+'" --launcher-token '+$token
    $info.WorkingDirectory=$repo; $info.UseShellExecute=$false; $info.CreateNoWindow=$true
    $info.RedirectStandardOutput=$true; $info.RedirectStandardError=$true; $info.RedirectStandardInput=$true
    $child=New-Object Diagnostics.Process; $child.StartInfo=$info
    if(-not $child.Start()) { throw 'Python did not start' }
    $childStarted=$true
    if(-not [ZombieRamNative]::AssignProcessToJobObject($job,$child.Handle)) { throw 'Owned diagnostic containment failed' }
    $stderrTask=$child.StandardError.ReadToEndAsync()
    $lineTask=$child.StandardOutput.ReadLineAsync(); $eof=$false
    $stage='python-startup'; $lastSnapshot=-2.0
    while(-not $child.HasExited -or -not $eof) {
        $row=Observe $stage; Check-Resources $row
        if($clock.Elapsed.TotalSeconds-$lastSnapshot -ge 2) {Process-Snapshot $stage; $lastSnapshot=$clock.Elapsed.TotalSeconds}
        if(-not $eof -and $lineTask.IsCompleted) {
            $line=$lineTask.GetAwaiter().GetResult()
            if($null -eq $line) { $eof=$true } else {
                if($line.Length -gt 65536) {throw 'Oversized worker protocol line'}
                $stdout.WriteLine($line)
                if($line.StartsWith('MODLY_RAM_IDENTITY=')) {
                    if($null -ne $interpreter){throw 'Duplicate interpreter identity'}
                    $interpreter=Resolve-WorkerProcess ($line.Substring(19)|ConvertFrom-Json) $job $child
                }
                if($line.StartsWith('MODLY_RAM_PHASE=')) {
                    if($null -eq $interpreter){throw 'Phase received without owned interpreter identity'}
                    $event=$line.Substring(16)|ConvertFrom-Json
                    if($event.edge -notin @('before','after') -or [string]::IsNullOrWhiteSpace($event.phase)) {throw 'Invalid phase protocol'}
                    $stage=$event.phase+'/'+$event.edge
                    Confirm-PhaseBoundary $stage {param($name) Observe $name 'boundary'} {param($row) Check-Resources $row} {
                        param($name) Process-Snapshot $name
                    } {if(-not $child.HasExited) { $child.StandardInput.WriteLine('CONTINUE'); $child.StandardInput.Flush() }}
                }
                $lineTask=$child.StandardOutput.ReadLineAsync()
            }
        }
        Start-Sleep -Milliseconds 200
    }
    $child.WaitForExit(); $childExit=$child.ExitCode
    $exitObservation=Observe 'child-exit' 'exit'
    if($childExit -ne 0) { throw ('CPU diagnostic child failed with exit code '+$childExit) }
    $childResult=Get-Content -LiteralPath (Join-Path $evidence 'child-result.json') -Raw | ConvertFrom-Json
    Confirm-ChildReceipt $childExit $childResult
    if($null -eq $interpreter){throw 'Complete interpreter identity missing'}
    $outcome='CPU_PREPARATION_COMPLETE'
} catch {
    $failure=$_.Exception.Message
    if($null -ne $child -and $childStarted -and -not $child.HasExited) {
        $forced=$true
        $ownedToClose=$job
        $job=[IntPtr]::Zero
        Stop-OwnedDiagnostic $child $ownedToClose {param($ownedJob) [void][ZombieRamNative]::CloseHandle($ownedJob)}
    }
    if($childStarted -and $child.HasExited) { $childExit=$child.ExitCode; $exitObservation=Observe 'child-exit-after-rejection' 'exit' }
} finally {
    if($job -ne [IntPtr]::Zero) { [void][ZombieRamNative]::CloseHandle($job) }
    if($childStarted) {
        for($index=0;$index -lt 4;$index++) {
            if($index -gt 0) {Start-Sleep -Seconds 4}
            try { $recovery+=Observe 'post-exit-recovery' 'recovery' } catch { $settlementErrors+=$_.Exception.Message }
        }
        try { Process-Snapshot 'post-exit-recovery' } catch { $settlementErrors+=$_.Exception.Message }
    }
    if($null -ne $stderrTask) {
        try {
            if($stderrTask.Wait(5000)) { Write-NewJson (Join-Path $evidence 'child.stderr.json') @{text=$stderrTask.GetAwaiter().GetResult()} }
            else { $settlementErrors+='stderr did not settle within 5 seconds' }
        } catch {$settlementErrors+=$_.Exception.Message}
    }
    $settlement=Resolve-SettlementOutcome $outcome $failure $settlementErrors
    $outcome=$settlement.outcome; $failure=$settlement.failure
    $recovered=$null
    if($recovery.Count -gt 0 -and $null -ne $exitObservation) { $recovered=$recovery[-1].available_bytes-$exitObservation.available_bytes }
    Write-NewJson (Join-Path $evidence 'summary.json') @{classification=$outcome;failure=$failure;failed_or_last_stage=$stage;
        implementation_commit=$ExpectedCommit;settlement_errors=$settlementErrors;
        timestamp_utc=[DateTimeOffset]::UtcNow.ToString('o');child_started=$childStarted;child_exit_code=$childExit;
        forced_owned_diagnostic_cleanup=$forced;minimum_observed_available_bytes=$(if($script:minimum -eq [long]::MaxValue){$null}else{$script:minimum});
        sampled_child_peak_working_set_bytes=$script:peakWS;sampled_child_peak_private_commit_bytes=$script:peakPrivate;
        launcher_pid=$(if($childStarted){$child.Id}else{$null});interpreter_pid=$(if($null -ne $interpreter){$interpreter.Id}else{$null});
        baseline=$baseline;exit_observation=$exitObservation;post_exit_available_change_bytes=$recovered;
        neural_attempt_reserved=$false;WSL_launched=$false;weights_loaded=$false;neural_forwards=0;
        interpretation='Compare observer/child phase deltas and concurrent process snapshots. Shared working sets, private commit, filesystem cache and inaccessible processes prevent exclusive physical attribution. Sampled peaks can miss transients. CPU completion is not neural admission.'}
    $report=@('Standalone zombie CPU preparation diagnostic',('Outcome: '+$outcome),('Failure: '+$failure),
        ('Minimum observed available physical RAM bytes: '+$script:minimum),
        ('Sampled child peak working set bytes: '+$script:peakWS),('Sampled child peak private commit bytes: '+$script:peakPrivate),
        ('Post-exit available RAM change bytes over 12 seconds: '+$recovered),
        'Phase deltas below are after minus before. Nested phases overlap and must not be summed.')
    $phaseFile=Join-Path $evidence 'phases.jsonl'
    if(Test-Path -LiteralPath $phaseFile) {
        foreach($line in Get-Content -LiteralPath $phaseFile) {
            $phase=$line|ConvertFrom-Json
            if($phase.edge -eq 'after') {
                $report+=('{0}: available {1}; working set {2}; private commit {3} bytes' -f $phase.phase,$phase.available_delta_bytes,$phase.working_set_delta_bytes,$phase.private_commit_delta_bytes)
            }
        }
    }
    $report+='Interpretation: temporal association is not exclusive physical attribution. Compare child/observer changes with private concurrent-process snapshots. Working sets share pages; private commit is not physical residency. No new safety margin or neural admission is established by CPU completion.'
    $reportStream=New-Journal (Join-Path $evidence 'report.txt')
    foreach($line in $report) {$reportStream.WriteLine($line)}
    $reportStream.Dispose()
    $telemetry.Dispose();$processJournal.Dispose();$stdout.Dispose();$observer.Dispose()
    if($null -ne $child) { $child.Dispose() }
    if($null -ne $interpreter) { $interpreter.Dispose() }
}
Write-Host ('Outcome: '+$outcome+'; evidence: '+$evidence)
if($outcome -ne 'CPU_PREPARATION_COMPLETE') { Write-Host $failure; exit 2 }
exit 0

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
    Write-Output 'PASS: 3 synthetic admission scenarios; no RAM query, child, WSL or model execution.'
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
        return @{pid=$Process.Id;working_set_bytes=$Process.WorkingSet64;
            private_commit_bytes=$Process.PrivateMemorySize64;os_peak_working_set_bytes=$Process.PeakWorkingSet64}
    } catch { return @{pid=$Process.Id;measurement_error=$_.Exception.Message} }
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
        available_bytes=$m.available_bytes;observer=(Process-Memory $observer);child=(Process-Memory $child)}
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
    if($Row.available_bytes -lt $script:Floor) { throw ('Below 22 GiB at '+$Row.stage+': '+$Row.available_bytes+' bytes') }
    if($clock.Elapsed.TotalSeconds -gt 480) { throw 'Original 480-second host deadline reached' }
    if((New-Object IO.DriveInfo('D:\')).AvailableFreeSpace -lt 50GB) { throw 'Original 50 GiB D: floor failed' }
    if((New-Object IO.DriveInfo([IO.Path]::GetPathRoot($evidence))).AvailableFreeSpace -lt 2GB) { throw 'Original 2 GiB evidence-volume floor failed' }
}

$identity=New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if($identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Run as the normal Windows user, without elevation' }
if(-not [IO.File]::Exists($python) -or -not [IO.File]::Exists($worker)) { throw 'Original Python or standalone worker missing' }
if($ExpectedCommit -notmatch '^[0-9a-f]{40}$') {throw '-Run requires the exact published -ExpectedCommit from the handoff'}
$head=(& git -C $repo rev-parse HEAD).Trim()
if($LASTEXITCODE -ne 0 -or $head -ne $ExpectedCommit) {throw 'Checkout differs from the exact published implementation'}
& git -C $repo diff --quiet $ExpectedCommit -- .
if($LASTEXITCODE -ne 0) {throw 'Tracked source differs from the published implementation'}
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
$observer=[Diagnostics.Process]::GetCurrentProcess(); $child=$null; $job=[IntPtr]::Zero
$clock=[Diagnostics.Stopwatch]::StartNew(); $script:minimum=[long]::MaxValue
$script:previousObservation=$null
$script:peakWS=[long]0; $script:peakPrivate=[long]0
$stage='admission'; $outcome='FAILED'; $failure=$null; $forced=$false; $childExit=$null
$exitObservation=$null; $recovery=@(); $baseline=$null; $stderrTask=$null; $childStarted=$false
$settlementErrors=@()
try {
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
                if($line.StartsWith('MODLY_RAM_PHASE=')) {
                    $event=$line.Substring(16)|ConvertFrom-Json
                    if($event.edge -notin @('before','after') -or [string]::IsNullOrWhiteSpace($event.phase)) {throw 'Invalid phase protocol'}
                    $stage=$event.phase+'/'+$event.edge
                    $boundary=Observe $stage 'boundary'; Check-Resources $boundary
                    Process-Snapshot $stage
                    if(-not $child.HasExited) { $child.StandardInput.WriteLine('CONTINUE'); $child.StandardInput.Flush() }
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
    if($childResult.classification -ne 'CPU_PREPARATION_COMPLETE' -or $childResult.forbidden_modules_present.Count -ne 0) {throw 'Complete CPU-only child receipt missing'}
    $outcome='CPU_PREPARATION_COMPLETE'
} catch {
    $failure=$_.Exception.Message
    if($null -ne $child -and $childStarted -and -not $child.HasExited) {
        $forced=$true
        if($job -ne [IntPtr]::Zero) { [void][ZombieRamNative]::CloseHandle($job); $job=[IntPtr]::Zero }
        if(-not $child.WaitForExit(5000)) { $child.Kill(); [void]$child.WaitForExit(5000) }
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
        if($stderrTask.Wait(5000)) { Write-NewJson (Join-Path $evidence 'child.stderr.json') @{text=$stderrTask.GetAwaiter().GetResult()} }
        else { $settlementErrors+='stderr did not settle within 5 seconds' }
    }
    $recovered=$null
    if($recovery.Count -gt 0 -and $null -ne $exitObservation) { $recovered=$recovery[-1].available_bytes-$exitObservation.available_bytes }
    Write-NewJson (Join-Path $evidence 'summary.json') @{classification=$outcome;failure=$failure;failed_or_last_stage=$stage;
        implementation_commit=$ExpectedCommit;settlement_errors=$settlementErrors;
        timestamp_utc=[DateTimeOffset]::UtcNow.ToString('o');child_started=$childStarted;child_exit_code=$childExit;
        forced_owned_diagnostic_cleanup=$forced;minimum_observed_available_bytes=$script:minimum;
        sampled_child_peak_working_set_bytes=$script:peakWS;sampled_child_peak_private_commit_bytes=$script:peakPrivate;
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
}
Write-Host ('Outcome: '+$outcome+'; evidence: '+$evidence)
if($outcome -ne 'CPU_PREPARATION_COMPLETE') { Write-Host $failure; exit 2 }
exit 0

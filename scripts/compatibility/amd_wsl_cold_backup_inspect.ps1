<# Read-only helper. Inventory never runs a guest; Hash uses .NET independently of Python. #>
[CmdletBinding(DefaultParameterSetName='Inventory')]
param(
    [Parameter(Mandatory,ParameterSetName='Inventory')][switch]$Inventory,
    [Parameter(Mandatory,ParameterSetName='Inventory')][switch]$WaitForOwner,
    [Parameter(Mandatory,ParameterSetName='Hash')][switch]$Hash,
    [Parameter(Mandatory,ParameterSetName='Hash')][string]$LiteralPath
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
if($Hash){
    $file=[IO.FileStream]::new($LiteralPath,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
    $sha=[Security.Cryptography.SHA256]::Create()
    try {
        $size=$file.Length
        $value=([BitConverter]::ToString($sha.ComputeHash($file))).Replace('-','').ToLowerInvariant()
        @{size=$size;sha256=$value;implementation='Windows PowerShell .NET SHA256.ComputeHash'} | ConvertTo-Json -Compress
    } finally {$sha.Dispose();$file.Dispose()}
    exit 0
}
if(-not $WaitForOwner -or [Console]::In.ReadLine() -cne 'MODLY_COLD_BACKUP_INVENTORY_GO' -or [Console]::In.Read() -ne -1){
    throw 'Exact owned inventory GO plus EOF required'
}
$key='HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
$matches=@(Get-ChildItem -LiteralPath $key | Where-Object {$_.GetValue('DistributionName') -eq 'Modly-AMD-Ubuntu-24.04'})
if($matches.Count -ne 1){throw 'Unique registered Modly distribution required'}
$k=$matches[0]
$registration=@{
    guid=$k.PSChildName;name=$k.GetValue('DistributionName');version=$k.GetValue('Version')
    base_path=$k.GetValue('BasePath');vhd_file=$k.GetValue('VhdFileName')
    default_uid=$k.GetValue('DefaultUid');flags=$k.GetValue('Flags');state=$k.GetValue('State')
}
$names=& 'C:\Windows\System32\wsl.exe' --list --running --quiet
if($LASTEXITCODE -ne 0){throw 'Read-only WSL running list failed'}
$running=@($names | ForEach-Object {($_ -replace "`0",'').Trim()} | Where-Object {$_})
foreach($name in $running){if($name -notmatch '^[A-Za-z0-9_.-]+$'){throw 'Ambiguous running inventory'}}
$result=@{
    utc=[DateTime]::UtcNow.ToString('o');registration=$registration;running=$running
    vm_processes=@(Get-Process -Name VmmemWSL,wslhost -ErrorAction SilentlyContinue | ForEach-Object {@{id=$_.Id;name=$_.ProcessName}})
    logical=@(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FileSystem,Size,VolumeSerialNumber)
    disks=@(Get-CimInstance Win32_DiskDrive | Select-Object Index,Model,SerialNumber)
    partitions=@(Get-Partition | Where-Object {$_.DriveLetter -in @('D','E')} | Select-Object DiskNumber,PartitionNumber,DriveLetter,Offset,Size)
    wsl_defaults=Get-ItemProperty -LiteralPath $key | Select-Object DefaultDistribution,DefaultVersion
}
$result | ConvertTo-Json -Depth 6 -Compress

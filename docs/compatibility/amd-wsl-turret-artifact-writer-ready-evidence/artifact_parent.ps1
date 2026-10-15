param([switch]$Prepare,[switch]$AllowExistingLeaf,
      [string]$LeafName='amd-wsl-turret-writer-ready-4870856')
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=New-Object Text.UTF8Encoding($false)
$taskAllowed=@('amd-wsl-turret-writer-ready-4870856','cpu-writer-acceptance-4870856')
if ($LeafName -cnotin $taskAllowed) { throw 'Unapproved artifact leaf' }
$taskRoot='C:\Users\wattz\Documents\ChatGPT\modly'
$taskParent=$taskRoot+'\artifacts'
$taskIdentity=[Security.Principal.WindowsIdentity]::GetCurrent()
$taskSid=$taskIdentity.User.Value
$taskTokenSids=@($taskSid,'S-1-1-0','S-1-5-11')+@($taskIdentity.Groups | ForEach-Object {$_.Value})
$taskRows=@()
function Assert-Directory([string]$Literal,[bool]$UserOwned) {
    $taskItem=Get-Item -LiteralPath $Literal -Force
    $taskResolved=(Resolve-Path -LiteralPath $Literal).ProviderPath
    if (-not $taskItem.PSIsContainer -or ($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        $taskItem.LinkType -or $taskResolved -ine $Literal -or [IO.Path]::GetFullPath($Literal) -ine $Literal) {
        throw 'Unexpected artifact directory redirection or type'
    }
    $taskAcl=Get-Acl -LiteralPath $Literal
    $taskOwner=$taskAcl.GetOwner([Security.Principal.SecurityIdentifier]).Value
    if ($UserOwned -and $taskOwner -ne $taskSid) { throw 'Incorrect artifact directory ownership' }
    if (-not $UserOwned -and $taskOwner -notin @($taskSid,'S-1-5-32-544')) { throw 'Unexpected profile ownership' }
    $taskRules=$taskAcl.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])
    if ($taskRules | Where-Object {$_.AccessControlType -eq 'Deny' -and $_.IdentityReference.Value -in $taskTokenSids}) {
        throw 'Artifact directory has applicable deny permissions'
    }
    if ($UserOwned -and -not ($taskRules | Where-Object {
        $_.IdentityReference.Value -eq $taskSid -and $_.AccessControlType -eq 'Allow' -and
        ($_.FileSystemRights -band [Security.AccessControl.FileSystemRights]::FullControl) -eq [Security.AccessControl.FileSystemRights]::FullControl})) {
        throw 'Artifact owner permissions are insufficient'
    }
    [ordered]@{path=$taskItem.FullName;resolved=$taskResolved;attributes=$taskItem.Attributes.ToString();owner_SID=$taskOwner;reparse=$false}
}
$taskRows+=Assert-Directory 'C:\Users\wattz' $false
foreach ($taskPath in @('C:\Users\wattz\Documents','C:\Users\wattz\Documents\ChatGPT',$taskRoot)) {
    $taskRows+=Assert-Directory $taskPath $true
}
$taskCreated=$false
if (-not (Test-Path -LiteralPath $taskParent)) {
    if (-not $Prepare) { throw 'Exact artifact parent is absent' }
    New-Item -ItemType Directory -Path $taskParent -ErrorAction Stop | Out-Null
    $taskCreated=$true
}
$taskRows+=Assert-Directory $taskParent $true
$taskLeaf=$taskParent+'\'+$LeafName
$taskLeafExists=Test-Path -LiteralPath $taskLeaf
if ($AllowExistingLeaf) {
    if (-not $taskLeafExists) { throw 'Artifact leaf disappeared' }
    $taskRows+=Assert-Directory $taskLeaf $true
} elseif ($taskLeafExists) { throw 'Exclusive artifact leaf already exists' }
$taskFree=([IO.DriveInfo]::new('C:\')).AvailableFreeSpace
if ($taskFree -lt 2GB) { throw 'Artifact output-volume disk admission refused' }
[ordered]@{status='PASS';identity=$taskIdentity.Name;SID=$taskSid;parent=$taskParent;leaf=$taskLeaf;
    parent_created=$taskCreated;leaf_exists=$taskLeafExists;C_free_bytes=$taskFree;directories=$taskRows;
    PowerShell_version=$PSVersionTable.PSVersion.ToString();policy_parent_whitelisted=$false} | ConvertTo-Json -Depth 8 -Compress

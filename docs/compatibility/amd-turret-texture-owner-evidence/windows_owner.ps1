param([Parameter(Mandatory=$true)][string]$Root, [switch]$Files)
$ErrorActionPreference='Stop'
# Match the established observer's process-local Desktop module resolution.
$env:PSModulePath=Join-Path $PSHOME 'Modules'
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Security/Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
$identity=[Security.Principal.WindowsIdentity]::GetCurrent()
$rootItem=Get-Item -Force -LiteralPath $Root
$rows=@()
$item=$rootItem
while ($null -ne $item) {
  $acl=Get-Acl -LiteralPath $item.FullName
  $rules=@($acl.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier]) | ForEach-Object {
    [ordered]@{sid=$_.IdentityReference.Value;rights=[int64]$_.FileSystemRights;rights_text=$_.FileSystemRights.ToString();type=$_.AccessControlType.ToString();inherited=$_.IsInherited;inheritance=$_.InheritanceFlags.ToString();propagation=$_.PropagationFlags.ToString()}
  })
  $rows+=,[ordered]@{path=$item.FullName;owner_sid=$acl.GetOwner([Security.Principal.SecurityIdentifier]).Value;group_sid=$acl.GetGroup([Security.Principal.SecurityIdentifier]).Value;sddl=$acl.Sddl;rules=$rules;attributes=[int64]$item.Attributes;reparse=($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0;link_type=$item.LinkType;target=@($item.Target)}
  $item=$item.Parent
}
$filesOut=@()
if ($Files) {
  foreach ($name in @('texture-field.safetensors','texture-field-metadata.json')) {
    $file=Get-Item -Force -LiteralPath (Join-Path $rootItem.FullName $name)
    $acl=Get-Acl -LiteralPath $file.FullName
    $filesOut+=,[ordered]@{path=$file.FullName;bytes=$file.Length;owner_sid=$acl.GetOwner([Security.Principal.SecurityIdentifier]).Value;sddl=$acl.Sddl;attributes=[int64]$file.Attributes;reparse=($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0;rules=@($acl.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier]) | ForEach-Object {[ordered]@{sid=$_.IdentityReference.Value;rights=[int64]$_.FileSystemRights;type=$_.AccessControlType.ToString();propagation=$_.PropagationFlags.ToString()}})}
  }
}
[ordered]@{windows_user=$identity.Name;current_sid=$identity.User.Value;root=$rootItem.FullName;ancestors=$rows;files=$filesOut;C_filesystem=(Get-Volume -DriveLetter C).FileSystemType.ToString()} | ConvertTo-Json -Depth 10 -Compress

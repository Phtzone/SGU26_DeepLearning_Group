$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class WordWindowInfo {
  public delegate bool EnumProc(IntPtr handle, IntPtr data);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc callback, IntPtr data);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr handle, out uint process);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr handle, StringBuilder text, int max);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr handle, StringBuilder text, int max);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr handle);
}
'@
$wordProcess = Get-Process WINWORD -ErrorAction SilentlyContinue
$wordProcess | Select-Object Id,CPU,Responding,StartTime | ConvertTo-Json
$wordIds = @($wordProcess.Id)
$callback = [WordWindowInfo+EnumProc] {
  param($handle, $data)
  [uint32]$processIdForWindow = 0
  $null = [WordWindowInfo]::GetWindowThreadProcessId($handle, [ref]$processIdForWindow)
  if ($wordIds -contains $processIdForWindow) {
    $titleBuffer = New-Object System.Text.StringBuilder 1024
    $classBuffer = New-Object System.Text.StringBuilder 256
    $null = [WordWindowInfo]::GetWindowText($handle,$titleBuffer,1024)
    $null = [WordWindowInfo]::GetClassName($handle,$classBuffer,256)
    [ordered]@{handle=$handle.ToInt64(); title=$titleBuffer.ToString(); class=$classBuffer.ToString(); visible=[WordWindowInfo]::IsWindowVisible($handle)} | ConvertTo-Json -Compress | Write-Output
  }
  return $true
}
$null = [WordWindowInfo]::EnumWindows($callback,[IntPtr]::Zero)

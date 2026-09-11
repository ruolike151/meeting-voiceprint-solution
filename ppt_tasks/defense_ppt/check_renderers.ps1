try {
  $ppt = New-Object -ComObject PowerPoint.Application
  Write-Output "POWERPOINT_OK $($ppt.Version)"
  $ppt.Quit()
} catch {
  Write-Output "POWERPOINT_MISSING $($_.Exception.Message)"
}

$sofficeCandidates = @(
  "C:\Program Files\LibreOffice\program\soffice.exe",
  "C:\Program Files (x86)\LibreOffice\program\soffice.exe",
  "$env:LOCALAPPDATA\Programs\LibreOffice\program\soffice.exe"
)
$found = $false
foreach ($c in $sofficeCandidates) {
  if (Test-Path -LiteralPath $c) {
    Write-Output "LIBREOFFICE_OK $c"
    $found = $true
  }
}
if (-not $found) {
  Write-Output "LIBREOFFICE_MISSING"
}

$powerpointExe = @(
  "C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE",
  "C:\Program Files (x86)\Microsoft Office\root\Office16\POWERPNT.EXE"
)
foreach ($p in $powerpointExe) {
  if (Test-Path -LiteralPath $p) {
    Write-Output "POWERPNT_EXE_OK $p"
  }
}

# DINOv3 ArcGIS Toolkit v2.15-rs311
# Clone official DINOv3 repo into C:\src\dinov3.
# Run in PowerShell. If git is not recognized after winget install, close and reopen PowerShell.

$ErrorActionPreference = "Stop"
$src = "C:\src"
$repo = "C:\src\dinov3"
$gitCmd = "git"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    $candidate = "C:\Program Files\Git\cmd\git.exe"
    if (Test-Path $candidate) {
        $gitCmd = $candidate
    } else {
        Write-Host "Git not found. Installing Git with winget..."
        winget install --id Git.Git -e --source winget
        Write-Host "Git installed. Close and reopen PowerShell, then run this script again."
        exit 1
    }
}

New-Item -ItemType Directory -Force -Path $src | Out-Null
if (Test-Path "$repo\.git") {
    Write-Host "DINOv3 repo already exists. Updating..."
    & $gitCmd -C $repo pull
} elseif (Test-Path $repo) {
    throw "C:\src\dinov3 exists but is not a git repo. Rename or remove it first."
} else {
    & $gitCmd clone https://github.com/facebookresearch/dinov3.git $repo
}

Write-Host "DINOv3 repo ready: $repo"

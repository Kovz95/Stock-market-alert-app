# Deploy to Docker on remote server (PowerShell version)
# Usage: .\deploy\deploy-docker.ps1 [-Server "root@Stockviz"] [-RemotePath "/opt/stockalert"]

param(
    [string]$Server = "root@45.63.20.126",
    [string]$RemotePath = "/opt/stockalert"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Stock Alert Docker Deployment ===" -ForegroundColor Green
Write-Host "Server: $Server"
Write-Host "Remote path: $RemotePath"
Write-Host ""

# Get app directory (parent of deploy folder)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Split-Path -Parent $ScriptDir

Set-Location $AppDir

# Step 1: Package and copy files to server using scp
Write-Host "[1/4] Packaging and copying files to server..." -ForegroundColor Yellow

# Create temporary archive
$TempArchive = "stockalert-deploy.tar.gz"
Write-Host "Creating archive..." -ForegroundColor Cyan

# Create exclusion file for tar
$ExcludeFile = ".deploy-exclude"
@"
.venv
__pycache__
*.pyc
.git
.env
node_modules
*.log
logs/
.pytest_cache
.mypy_cache
*.egg-info
$TempArchive
$ExcludeFile
"@ | Out-File -FilePath $ExcludeFile -Encoding utf8

# Check if tar is available
$tarAvailable = Get-Command tar -ErrorAction SilentlyContinue

if ($tarAvailable) {
    # Use tar directly (Windows 10+ has built-in tar)
    tar --exclude-from="$ExcludeFile" -czf $TempArchive -C . .
} else {
    # Try using WSL tar
    Write-Host "Using WSL for tar..." -ForegroundColor Cyan
    $wslPath = (wsl wslpath -a ($AppDir -replace '\\', '/'))
    wsl tar --exclude-from=".deploy-exclude" -czf "$TempArchive" -C "$wslPath" .
}

# Copy archive to server using scp
Write-Host "Copying to server..." -ForegroundColor Cyan
scp $TempArchive "${Server}:${RemotePath}/"

# Extract on server
Write-Host "Extracting on server..." -ForegroundColor Cyan
ssh $Server "cd $RemotePath && tar -xzf $TempArchive && rm $TempArchive"

# Clean up local archive
Remove-Item $TempArchive -ErrorAction SilentlyContinue
Remove-Item $ExcludeFile -ErrorAction SilentlyContinue
Write-Host "Cleanup complete." -ForegroundColor Cyan

# Step 2: Copy .env.production to server as .env
Write-Host "[2/4] Deploying production environment file..." -ForegroundColor Yellow

# Check if .env.production exists locally
if (-not (Test-Path ".env.production")) {
    Write-Host "ERROR: .env.production file not found in project root" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please ensure .env.production exists with required variables:"
    Write-Host "  - POSTGRES_PASSWORD"
    Write-Host "  - FMP_API_KEY"
    Write-Host "  - WEBHOOK_URL (Discord)"
    exit 1
}

Write-Host "Copying .env.production to server as .env..." -ForegroundColor Cyan
scp ".env.production" "${Server}:${RemotePath}/.env"
Write-Host "Environment file deployed successfully." -ForegroundColor Green

# Step 3: Build and start containers
Write-Host "[3/4] Building and starting Docker containers..." -ForegroundColor Yellow
ssh $Server "cd $RemotePath && docker-compose down && docker-compose up -d --build"

# Step 4: Show status
Write-Host "[4/4] Checking container status..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
ssh $Server "cd $RemotePath && docker-compose ps"

Write-Host ""
Write-Host "=== Deployment Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  View logs:     ssh $Server 'cd $RemotePath && docker-compose logs -f'"
Write-Host "  View app logs: ssh $Server 'cd $RemotePath && docker-compose logs -f streamlit'"
Write-Host "  Restart:       ssh $Server 'cd $RemotePath && docker-compose restart'"
Write-Host "  Stop:          ssh $Server 'cd $RemotePath && docker-compose down'"
Write-Host ""

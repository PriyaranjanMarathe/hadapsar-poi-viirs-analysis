# Define the root directory for the project
$projectRoot = "C:\Projects\hadapsar-poi-viirs-analysis"

# Define the folder structure
$folders = @(
    "data", # Raw datasets
    "notebooks", # Jupyter notebooks for analysis
    "src", # Python scripts for data processing and analysis
    "outputs"        # Visualizations, models, and reports
)

# Create the root directory
if (!(Test-Path -Path $projectRoot)) {
    New-Item -ItemType Directory -Path $projectRoot
    Write-Host "Created project root directory: $projectRoot"
}
else {
    Write-Host "Project root directory already exists: $projectRoot"
}

# Create subdirectories
foreach ($folder in $folders) {
    $folderPath = Join-Path $projectRoot $folder
    if (!(Test-Path -Path $folderPath)) {
        New-Item -ItemType Directory -Path $folderPath
        Write-Host "Created folder: $folderPath"
    }
    else {
        Write-Host "Folder already exists: $folderPath"
    }
}

# Create empty README.md file if it doesn't exist
$readmePath = Join-Path $projectRoot "README.md"
if (!(Test-Path -Path $readmePath)) {
    New-Item -ItemType File -Path $readmePath
    Write-Host "Created README.md file: $readmePath"
}
else {
    Write-Host "README.md file already exists: $readmePath"
}

# Create .gitignore file
$gitignorePath = Join-Path $projectRoot ".gitignore"
if (!(Test-Path -Path $gitignorePath)) {
    Set-Content -Path $gitignorePath -Value `
        "# Ignore data files
data/
"
    Write-Host "Created .gitignore file: $gitignorePath"
}
else {
    Write-Host ".gitignore file already exists: $gitignorePath"
}

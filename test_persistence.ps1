# Test 1
Write-Host "Run 1: Downloading 1 asset..."
docker compose run asset_downloader --count 1

Write-Host "`nChecking files..."
$count1 = (Get-ChildItem -Path "data\assets\raw" -Recurse -File).Count
Write-Host "Files after run 1: $count1"

# Test 2
Write-Host "`nRun 2: Downloading 2 assets total..."
docker compose run asset_downloader --count 2

Write-Host "`nChecking files..."
$count2 = (Get-ChildItem -Path "data\assets\raw" -Recurse -File).Count
Write-Host "Files after run 2: $count2"

if ($count2 -gt $count1) {
    Write-Host "`n✅ SUCCESS! Data persists between runs!" -ForegroundColor Green
} else {
    Write-Host "`n❌ PROBLEM! Data not persisting!" -ForegroundColor Red
}
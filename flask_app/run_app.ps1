# Run Flask app with UTF-8 encoding (fixes UnicodeDecodeError on Windows)
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot
python app.py

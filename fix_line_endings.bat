@echo off
echo Converting line endings in shell scripts...
powershell -Command "(Get-Content -Raw start.sh) -replace '\r\n', '\n' | Set-Content -NoNewline start.sh"
powershell -Command "(Get-Content -Raw docker_build.sh) -replace '\r\n', '\n' | Set-Content -NoNewline docker_build.sh"
echo Done! Shell scripts now have Unix line endings (LF).

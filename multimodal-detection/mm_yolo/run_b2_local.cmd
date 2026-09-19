@echo off
rem B2 local launcher. It creates b2_depth4_s and never overwrites b1_register_s.
set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%watchdog.ps1" -Only B2

@echo off
rem Launch the unattended A0 -> B0 training watchdog.
rem Keep this file ASCII-only; the encoding-sensitive part lives in watchdog.ps1 (UTF-8 with BOM).
cd /d "C:\Users\35482\Desktop\人工智能精英\2"
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\35482\Desktop\人工智能精英\2\code\mm_yolo\watchdog.ps1" >> "C:\Users\35482\Desktop\人工智能精英\2\code\runs\watchdog_stdout.log" 2>&1

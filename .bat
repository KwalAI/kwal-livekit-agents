@echo off
echo Reading .env file...
for /f "delims== tokens=1,2" %%a in (.env) do (
    echo Setting %%a=%%b...
    set %%a=%%b
)
echo Done.
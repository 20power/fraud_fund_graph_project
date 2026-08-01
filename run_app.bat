@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if exist "%CD%\.venv\Scripts\python.exe" goto use_venv

set "CONDA_BAT="
where conda.bat >nul 2>nul
if errorlevel 1 goto conda_common_paths
set "CONDA_BAT=conda.bat"
goto conda_ready

:conda_common_paths
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\anaconda\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\anaconda\condabin\conda.bat"
if not defined CONDA_BAT goto no_conda

:conda_ready
call "%CONDA_BAT%" activate fraud_fund_graph >nul 2>nul
if not errorlevel 1 goto conda_active
call "%CONDA_BAT%" activate jk_new >nul 2>nul
if errorlevel 1 goto no_environment

:conda_active
set "PYTHON_EXE=python"
goto launch

:use_venv
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
goto launch

:no_conda
echo [ERROR] Project .venv or Conda was not found.
echo Follow the deployment manual under the docs directory.
pause
exit /b 1

:no_environment
echo [ERROR] Conda environment fraud_fund_graph or jk_new was not found.
echo Run: conda env create -f environment.yml
pause
exit /b 1

:launch
set "PYTHONPATH=%CD%\src"
"%PYTHON_EXE%" -c "import streamlit, pandas, sklearn, pyarrow" >nul 2>nul
if errorlevel 1 goto missing_dependencies

echo Open http://127.0.0.1:8501 after the service starts.
"%PYTHON_EXE%" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
exit /b %ERRORLEVEL%

:missing_dependencies
echo [ERROR] Python dependencies are incomplete.
echo Run: python -m pip install -r requirements.txt
pause
exit /b 1

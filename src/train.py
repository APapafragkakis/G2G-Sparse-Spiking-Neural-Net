@echo off
cd /d "%~dp0"
setlocal EnableExtensions EnableDelayedExpansion

REM =========================
REM PATHS
REM =========================
set "PY=py"
set "TRAIN=src\train.py"

if not exist "%TRAIN%" (
  echo [ERROR] Cannot find "%TRAIN%"
  echo Current dir: %CD%
  pause
  exit /b 1
)

REM =========================
REM CONFIG
REM =========================
set "DATASET=cifar10"
set "T=50"
set "BS=256"
set "EPOCHS=20"

set "USE_RESNET=--use_resnet"
set "SPARSEMODE=--sparsity_mode static"

REM We will run a single p_inter value
set "P_INTER=0.90"

REM Capacity-match: keep SAME hidden_dim across dense/index/random/mixer
set "H_CAPMATCH=488"

REM Logs
set "LOGDIR=logs_sparsecapmatch_%DATASET%_T%T%_bs%BS%_e%EPOCHS%_p%P_INTER%_h%H_CAPMATCH%"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo =====================================================
echo RUN (DENSE + INDEX + RANDOM + MIXER) @ p_inter=%P_INTER%
echo dataset=%DATASET% epochs=%EPOCHS% T=%T% batch=%BS%
echo use_resnet=YES sparsity_mode=static
echo hidden_dim(capmatch)=%H_CAPMATCH%
echo Logs: %LOGDIR%
echo =====================================================

REM =========================
REM DENSE
REM =========================
echo [RUN] dense (capmatch)
%PY% "%TRAIN%" --dataset %DATASET% --model dense %USE_RESNET% %SPARSEMODE% ^
  --p_inter %P_INTER% --epochs %EPOCHS% --T %T% --batch_size %BS% --hidden_dim %H_CAPMATCH% ^
  1> "%LOGDIR%\dense_resnet_p%P_INTER%_h%H_CAPMATCH%.log" 2>&1

if errorlevel 1 (
  echo [ERROR] dense failed
  echo Check log: "%LOGDIR%\dense_resnet_p%P_INTER%_h%H_CAPMATCH%.log"
  pause
  exit /b 1
)

REM =========================
REM INDEX
REM =========================
echo [RUN] index
%PY% "%TRAIN%" --dataset %DATASET% --model index %USE_RESNET% %SPARSEMODE% ^
  --p_inter %P_INTER% --epochs %EPOCHS% --T %T% --batch_size %BS% --hidden_dim %H_CAPMATCH% ^
  1> "%LOGDIR%\index_resnet_p%P_INTER%_h%H_CAPMATCH%.log" 2>&1

if errorlevel 1 (
  echo [ERROR] index failed
  echo Check log: "%LOGDIR%\index_resnet_p%P_INTER%_h%H_CAPMATCH%.log"
  pause
  exit /b 1
)

REM =========================
REM RANDOM
REM =========================
echo [RUN] random
%PY% "%TRAIN%" --dataset %DATASET% --model random %USE_RESNET% %SPARSEMODE% ^
  --p_inter %P_INTER% --epochs %EPOCHS% --T %T% --batch_size %BS% --hidden_dim %H_CAPMATCH% ^
  1> "%LOGDIR%\random_resnet_p%P_INTER%_h%H_CAPMATCH%.log" 2>&1

if errorlevel 1 (
  echo [ERROR] random failed
  echo Check log: "%LOGDIR%\random_resnet_p%P_INTER%_h%H_CAPMATCH%.log"
  pause
  exit /b 1
)

REM =========================
REM MIXER
REM =========================
echo [RUN] mixer
%PY% "%TRAIN%" --dataset %DATASET% --model mixer %USE_RESNET% %SPARSEMODE% ^
  --p_inter %P_INTER% --epochs %EPOCHS% --T %T% --batch_size %BS% --hidden_dim %H_CAPMATCH% ^
  1> "%LOGDIR%\mixer_resnet_p%P_INTER%_h%H_CAPMATCH%.log" 2>&1

if errorlevel 1 (
  echo [ERROR] mixer failed
  echo Check log: "%LOGDIR%\mixer_resnet_p%P_INTER%_h%H_CAPMATCH%.log"
  pause
  exit /b 1
)

echo.
echo =====================================================
echo DONE
echo Logs folder: "%LOGDIR%"
echo =====================================================
pause
endlocal

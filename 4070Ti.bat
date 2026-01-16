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
REM GLOBAL CONFIG
REM =========================
set "DATASET=cifar10"
set "T=50"
set "BS=256"
set "EPOCHS=20"

set "USE_RESNET=--use_resnet"
set "SPARSEMODE=--sparsity_mode static"

REM =========================================================
REM ASSUMPTION FOR THIS MAPPING:
REM - ResNet cut_at=layer1
REM - pool_hw=4  -> embedding dim D=256
REM - Capacity match counts ONLY fc1+fc2+fc3 (3 hidden layers)
REM - ResNet params are ignored (frozen and identical)
REM =========================================================

REM Fixed sparse hidden dim (for index/random/mixer)
set "H_SPARSE=512"

REM Fully dense baseline run (once at the end)
set "H_DENSE_FULL=512"

REM p_inter sweep grid
set "P_LIST=0 0.05 0.15 0.25 0.35 0.5 0.75"

REM Logs
set "LOGDIR=logs_resnet_%DATASET%_T%T%_bs%BS%_e%EPOCHS%_static_sparseh%H_SPARSE%_densecapmatch_D256_plus_dense%H_DENSE_FULL%_pgrid"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo =====================================================
echo START SWEEP (CIFAR10)
echo dataset=%DATASET% epochs=%EPOCHS% T=%T% batch=%BS%
echo sparse hidden_dim = %H_SPARSE%
echo dense FULL baseline hidden_dim = %H_DENSE_FULL% (runs once at end)
echo p_inter grid: %P_LIST%
echo Logs: %LOGDIR%
echo =====================================================

REM =====================================================
REM SWEEP: for each p -> dense(capmatch) + sparse(index/random/mixer)
REM =====================================================
for %%P in (%P_LIST%) do (

  REM -------------------------
  REM Dense capmatch mapping (D=256, hidden layers only)
  REM -------------------------
  set "H_DENSE_CAP="
  if "%%P"=="0"     set "H_DENSE_CAP=153"
  if "%%P"=="0.05"  set "H_DENSE_CAP=183"
  if "%%P"=="0.15"  set "H_DENSE_CAP=235"
  if "%%P"=="0.25"  set "H_DENSE_CAP=279"
  if "%%P"=="0.35"  set "H_DENSE_CAP=319"
  if "%%P"=="0.5"   set "H_DENSE_CAP=371"
  if "%%P"=="0.75"  set "H_DENSE_CAP=446"

  if not defined H_DENSE_CAP (
    echo [ERROR] No dense mapping for p_inter=%%P
    pause
    exit /b 1
  )

  set "PSTR=%%P"
  set "PSTR=!PSTR:.=_!"

  echo.
  echo ---- %DATASET% - p_inter=%%P ----
  echo      dense_capmatch_h=!H_DENSE_CAP!, sparse_h=%H_SPARSE%

  REM =========================
  REM DENSE (CAPACITY MATCHED)
  REM =========================
  echo [RUN] dense ^(capmatch^)
  %PY% "%TRAIN%" --dataset %DATASET% --model dense %USE_RESNET% %SPARSEMODE% ^
    --p_inter %%P --epochs %EPOCHS% --T %T% --batch_size %BS% --hidden_dim !H_DENSE_CAP! ^
    1> "%LOGDIR%\dense_capmatch_D256_p!PSTR!_h!H_DENSE_CAP!.log" 2>&1

  if errorlevel 1 (
    echo [ERROR] dense^(capmatch^) failed for dataset=%DATASET% p=%%P
    echo Check log: "%LOGDIR%\dense_capmatch_D256_p!PSTR!_h!H_DENSE_CAP!.log"
    pause
    exit /b 1
  )

  REM =========================
  REM SPARSE RUNS
  REM =========================
  for %%M in (index random mixer) do (
    echo [RUN] %%M ^(sparse^)
    %PY% "%TRAIN%" --dataset %DATASET% --model %%M %USE_RESNET% %SPARSEMODE% ^
      --p_inter %%P --epochs %EPOCHS% --T %T% --batch_size %BS% --hidden_dim %H_SPARSE% ^
      1> "%LOGDIR%\%%M_resnet_p!PSTR!_h%H_SPARSE%.log" 2>&1

    if errorlevel 1 (
      echo [ERROR] %%M failed for dataset=%DATASET% p=%%P
      echo Check log: "%LOGDIR%\%%M_resnet_p!PSTR!_h%H_SPARSE%.log"
      pause
      exit /b 1
    )
  )
)

REM =====================================================
REM FINAL: ONE FULL DENSE BASELINE (hidden_dim=512) ONCE
REM =====================================================
echo.
echo =====================================================
echo [FINAL RUN] dense FULL baseline ^(once^) hidden_dim=%H_DENSE_FULL%
echo =====================================================

%PY% "%TRAIN%" --dataset %DATASET% --model dense %USE_RESNET% %SPARSEMODE% ^
  --epochs %EPOCHS% --T %T% --batch_size %BS% --hidden_dim %H_DENSE_FULL% ^
  1> "%LOGDIR%\dense_full_h%H_DENSE_FULL%_baseline.log" 2>&1

if errorlevel 1 (
  echo [ERROR] dense^(full baseline^) failed
  echo Check log: "%LOGDIR%\dense_full_h%H_DENSE_FULL%_baseline.log"
  pause
  exit /b 1
)

echo.
echo =====================================================
echo ALL RUNS COMPLETED SUCCESSFULLY
echo =====================================================
pause
endlocal

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
set "T=50"
set "BS=256"
set "EPOCHS=20"
set "H_SPARSE=512"

set "USE_RESNET=--use_resnet"
set "SPARSEMODE=--sparsity_mode dynamic"

REM datasets
set "DATASET_LIST=cifar10 cifar100"

REM p' grid (χωρίς 0, 0.75, 0.9)
set "P_LIST=0.05 0.15 0.25 0.35 0.5"

REM DST configs (3 pruning x 2 growing = 6)
set "CP_LIST=set random hebb"
set "CG_LIST=hebb random"

REM Models
set "MODEL_LIST=index random mixer"

REM =====================================================
REM LOOP: dataset -> p -> model -> cp -> cg
REM =====================================================
for %%D in (%DATASET_LIST%) do (

  REM Logs per dataset
  set "LOGDIR=logs_dst_%%D_T%T%_bs%BS%_e%EPOCHS%_h%H_SPARSE%_pgrid_cp3_cg2"
  if not exist "!LOGDIR!" mkdir "!LOGDIR!"

  echo =====================================================
  echo START DST SWEEP for %%D
  echo dataset=%%D epochs=%EPOCHS% T=%T% batch=%BS% hidden_dim=%H_SPARSE%
  echo models=%MODEL_LIST%
  echo p grid=%P_LIST%
  echo cp=%CP_LIST%  ^|  cg=%CG_LIST%
  echo Logs: !LOGDIR!
  echo =====================================================

  for %%P in (%P_LIST%) do (

    set "PSTR=%%P"
    set "PSTR=!PSTR:.=_!"

    echo.
    echo ===============================
    echo Dataset=%%D   p_inter=%%P
    echo ===============================

    for %%M in (%MODEL_LIST%) do (
      for %%C in (%CP_LIST%) do (
        for %%G in (%CG_LIST%) do (

          echo [RUN] %%D %%M p=%%P cp=%%C cg=%%G

          %PY% "%TRAIN%" --dataset %%D --model %%M %USE_RESNET% %SPARSEMODE% ^
            --p_inter %%P --cp %%C --cg %%G ^
            --epochs %EPOCHS% --T %T% --batch_size %BS% --hidden_dim %H_SPARSE% ^
            1> "!LOGDIR!\%%M_dst_p!PSTR!_cp%%C_cg%%G_h%H_SPARSE%.log" 2>&1

          if errorlevel 1 (
            echo [ERROR] FAILED: dataset=%%D model=%%M p=%%P cp=%%C cg=%%G
            echo Check log: "!LOGDIR!\%%M_dst_p!PSTR!_cp%%C_cg%%G_h%H_SPARSE%.log"
            pause
            exit /b 1
          )

        )
      )
    )
  )
)

echo.
echo =====================================================
echo ALL DST RUNS COMPLETED SUCCESSFULLY
echo =====================================================
pause
endlocal

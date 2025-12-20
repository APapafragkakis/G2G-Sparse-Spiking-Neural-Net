@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =========================
REM PATHS
REM =========================
set "TRAIN=src\train.py"
if not exist "%TRAIN%" (
  echo ERROR: cannot find "%TRAIN%" from %CD%
  echo Make sure you run this .bat from the project root.
  exit /b 1
)

REM =========================
REM GLOBAL DEFAULTS
REM =========================
set "EPOCHS=20"
set "T_MAIN=50"
set "BS=256"
set "ENC_SCALE=1.0"
set "ENC_BIAS=0.0"
set "SPARSITY=static"

REM Only rate coding
set "ENC=rate"

REM p' list
set "P_LIST_MAIN=0.00 0.10 0.25 0.35 0.50 0.75"

REM Root output folder (relative to this project folder)
set "ROOT=results"
if not exist "%ROOT%" mkdir "%ROOT%"

REM ============================================================
REM MAIN EXPERIMENT
REM dataset × model × p' (rate coding only)
REM - dense runs ONCE per dataset (p' ignored)
REM - index/random/mixer sweep over p'
REM - skips runs whose output log already exists
REM ============================================================
echo.
echo ===== MAIN EXPERIMENT (rate coding only) =====

set "OUTDIR=%ROOT%\main_rate_T%T_MAIN%"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

for %%D in (fashionmnist cifar10 cifar100) do (
  if not exist "%OUTDIR%\%%D" mkdir "%OUTDIR%\%%D"

  REM -------------------------
  REM DENSE: run once per dataset
  REM -------------------------
  set "OUT=%OUTDIR%\%%D\dense.txt"
  if exist "!OUT!" (
    echo [SKIP] dataset=%%D ^| model=dense ^| exists: !OUT!
  ) else (
    echo [RUN ] dataset=%%D ^| model=dense ^| enc=%ENC% ^| T=%T_MAIN%

    python "%TRAIN%" ^
      --dataset %%D ^
      --model dense ^
      --p_inter 0.00 ^
      --epochs %EPOCHS% ^
      --T %T_MAIN% ^
      --batch_size %BS% ^
      --enc %ENC% ^
      --enc_scale %ENC_SCALE% ^
      --enc_bias %ENC_BIAS% ^
      --sparsity_mode %SPARSITY% ^
      > "!OUT!" 2>&1
  )

  REM -------------------------
  REM SPARSE MODELS: p' sweep
  REM -------------------------
  for %%M in (index random mixer) do (
    for %%P in (%P_LIST_MAIN%) do (
      set "OUT=%OUTDIR%\%%D\%%M_pinter_%%P.txt"

      if exist "!OUT!" (
        echo [SKIP] dataset=%%D ^| model=%%M ^| p'=%%P ^| exists: !OUT!
      ) else (
        echo [RUN ] dataset=%%D ^| model=%%M ^| p'=%%P ^| enc=%ENC% ^| T=%T_MAIN%

        python "%TRAIN%" ^
          --dataset %%D ^
          --model %%M ^
          --p_inter %%P ^
          --epochs %EPOCHS% ^
          --T %T_MAIN% ^
          --batch_size %BS% ^
          --enc %ENC% ^
          --enc_scale %ENC_SCALE% ^
          --enc_bias %ENC_BIAS% ^
          --sparsity_mode %SPARSITY% ^
          > "!OUT!" 2>&1
      )
    )
  )
)

echo.
echo ===== ALL EXPERIMENTS FINISHED =====
endlocal

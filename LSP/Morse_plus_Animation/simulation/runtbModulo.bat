@echo ***********************************************************************
@echo Testbench takes a long time. To terminate its execution, press Ctrl C 
@echo ************************************************************************
@ECHO ON
SETLOCAL
@rem Temporary moving mingw64 to top
@set PATH=C:\msys64\mingw64\bin\;%PATH%
@rem Compile for VHDL-2008
set GHDL_FLAGS=-fsynopsys --std=08
set TBNAME=testbench_LCDgenOnly
@rem Analyse and create object files
ghdl.exe -a %GHDL_FLAGS% ../LCDpackage.vhd ../LCDtestModulo.vhd ../%TBNAME%.vhd 
IF ERRORLEVEL 1 GOTO BAT-END
ghdl.exe -e %GHDL_FLAGS% %TBNAME% 
IF ERRORLEVEL 1 GOTO BAT-END
rem the frame needs 16.6 ms, for 3 frames 60 ms as safety break
ghdl.exe -r %GHDL_FLAGS% %TBNAME% --stop-time=60ms
:BAT-END

@echo ***********************************************************************
@echo Testbench simulates 32 frames. To terminate its execution, press Ctrl C 
@echo ************************************************************************
@ECHO ON
SETLOCAL
@rem Temporary moving mingw64 to top
@set PATH=C:\msys64\mingw64\bin\;%PATH%
@rem Compile for VHDL-2008
set GHDL_FLAGS=-fsynopsys --std=08
set TBNAME=testbench_LCDlogicAnim
@rem Analyse and create object files
ghdl.exe -a %GHDL_FLAGS% ../LCDpackage.vhd ../Consolas24pt_Font.vhd ../imgLSPbmp.vhd ../ZHYTN_bmp.vhd ../LCDlogic0anim.vhd ../%TBNAME%.vhd 
IF ERRORLEVEL 1 GOTO BAT-END
ghdl.exe -e %GHDL_FLAGS% %TBNAME% 
IF ERRORLEVEL 1 GOTO BAT-END
rem 20 ms time is a safety break, the frame needs only 16.6 ms, for more frame we need a higher safety time value
ghdl.exe -r %GHDL_FLAGS% %TBNAME% --stop-time=1000000ms
:BAT-END

@ECHO ON
SETLOCAL
@rem Temporary moving mingw64 to top
@set PATH=C:\msys64\mingw64\bin\;%PATH%
@rem Compile for VHDL-2008
set GHDL_FLAGS=-fsynopsys --std=08
set TBNAME=testbench_LCDlogic
@rem Analyse and create object files
ghdl.exe -a %GHDL_FLAGS% ../LCDpackage.vhd ../LCDlogic0.vhd ../%TBNAME%.vhd 
IF ERRORLEVEL 1 GOTO BAT-END
ghdl.exe -e %GHDL_FLAGS% %TBNAME% 
IF ERRORLEVEL 1 GOTO BAT-END
rem 20 ms time is a safety break, the frame needs only 16.6 ms
ghdl.exe -r %GHDL_FLAGS% %TBNAME% --stop-time=20ms
:BAT-END

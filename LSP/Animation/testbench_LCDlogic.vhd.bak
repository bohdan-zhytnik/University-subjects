-- testbench_LCDlogic version V2.0 (March 16, 2024] adapted for GHDL simulation
-------------------------------------------------------------
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta]
-- Published under GNU General Public License V2.0 2023
-------------------------------------------------------------
-- Note:
-- 1. To find proper places for changes, search for string: ***ToDo !!!!!
-- 2. In Quartus, we can test this code by "Start Analysis & Synthesis" button, 
--    but it always returns the error: Error (10533): VHDL Wait Statement error...
--    It is OK - testbench codes are not synthesizable. They are intended for ModelSim.
-- 3. The result text file, see ***ToDo1, can be opened by LCDTestbenchViewer version 3 and later/
-----------------------------------
library ieee, work;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_textio.all;
use std.textio.all;
use work.LCDpackage.all;

entity testbench_LCDlogic is --testbench entity is always without any inputs/outputs.
end testbench_LCDlogic;

architecture testbench of testbench_LCDlogic is
    ---------------------------------------------------------------------------------------
    -- ***ToDo 1 of 3: Replace by your own component LCDlogic.
    --                 and correct also ***ToDo3 at the end of this file !!!  
    component LCDlogic0
        port(
    xcolumn  : in xy_t:=(others=>'0');  -- x-coordinate of pixel (column index)
    yrow     : in xy_t:=(others=>'0');   -- y-coordinate of pixel (row index)
    XEND_N   : in std_logic:='0';   -- '0' only when max xcolumn, otherwise '1' 
    YEND_N   : in std_logic:='0';   -- '0' only when max yrow
    LCD_DE   : in std_logic:='0';   -- DataEnable control signal of LCD controller, it signals visible part.
    LCD_DCLK : in std_logic:='0';   -- LCD data clock, 33 MHz, see note 2 below
    RGBcolor  : out RGB_t --  color information
    ); 
    end component;
    -------------------------------------------------------------------------------
    -- ***ToDo 2 of 3 - *** Setting of simulations ****
    constant FILE_NAME : string := "C:/SPS/testbenchLCD.txt";
    -- Your filename for testbench results in Unix style, i.e., use no spaces and / instead of \ (backslashes). 
    -- The path must exist!!! The file will be created or overwritten.

    constant ENABLE_COMPRESSION : boolean := TRUE;
    -- The compression reduces file size and accelerates simulation, but it writes only complete lines to disc drive. 
    -- Disable the compression, if you want to see immediate pixels after breakpoints.  

   constant SINGLE_FRAME : boolean := TRUE;
    --  On TRUE, the simulation is terminated after the first frame.
    --  On FALSE, the simulation terminates after storing COUNT_OF_FRAMES. 
    --  You should also set ENABLE_COMPRESSION=TRUE to prevent huge file sizes.
    --  !!! The version 3.0 of LCD Testbench Viewer displays only the first frame. More frames are only planned. !!!
    
    constant COUNT_OF_FRAMES : positive := 10;
    -- The count of stored frames is active only if SINGLE_FRAME=FALSE, otherwise it is ignored.

    --------------------------------------------------------------
    -- Internal signals
    ------------------------------
    signal CLK_33MHz       : std_logic:='0';
    constant CLOCK_PERIOD  : time      := 30 ns; -- 33 MHz
    signal beginWriting    : boolean:=FALSE; -- initialization after power-up

    -- signals that correspond to input ports of VEEK-MT2
    signal LCD_DCLK          : std_logic               := '0'; -- LCD pixel clock, 33 MHz, see note 2 below
    signal LCD_DE            : std_logic               := '0'; -- DataEnable control signal of LCD controller
    signal CLRN              : std_logic               := '0'; -- Power-up initialization
    signal RGBcolor          : RGB_t           := (others=>'0');

    ---------------------------------------------------------------------------------

    -- connection signals
    signal xcolumn, yrow  : xy_t                 := (others => '0');
    constant XYZERO      : unsigned(xy_t'range)  := (others => '0');
    signal XEND_N, YEND_N : std_logic            := '1';

    constant LCD_XCOLUMN_MAX : unsigned(xcolumn'RANGE) := (others => '1'); --1023
    constant LCD_YROW_MAX    : integer                 := 524;

begin

    -- Clock process with 50% duty cycle is generated here.
    clk_process : process
    begin
        CLK_33MHz <= '0';
        wait for CLOCK_PERIOD / 2;      --for 50 % of CLK_33MHz period is'0'.
        CLK_33MHz <= '1';
        wait for CLOCK_PERIOD / 2;      --for next 50% of CLK_33MHz period is '1'.
    end process;

    LCD_DCLK <= CLK_33MHz;

    -- Clear process keeps CLRN='0' for 3 CLOCK_PERIOD.
    clear_process : process
    begin
        beginWriting <= FALSE; CLRN<='0';
        wait for 3*CLOCK_PERIOD;
        beginWriting <= TRUE; CLRN<='1';   
        wait;           -- terminate process.
    end process;

    stimuls : process(CLK_33MHz, beginWriting)
        file outfile        : TEXT;
        variable outline    : line;     --line
        variable fstatus    : FILE_OPEN_STATUS;

        subtype data_t is RGB_t; -- for results
        variable data, lastdata : data_t := (others => '0'); -- RGB

        variable iswrite           : boolean := FALSE;
        constant HEADER            : string    := "## LCD Testbench result - 1024x525 full LCD frame";
        constant NUMHEADER         : string    := "##=";
        variable repeat_counter    : integer   := 0;
        variable counter_of_frames : integer   := 0;

        procedure FLUSH_COMPRESSION(repeatCounter : inout integer) is
        begin
            if repeatCounter > 0 then
                write(outline, '*' & integer'image(repeatCounter) & LF);
                repeatCounter := 0;
            end if;
        end procedure;
        function boolean2std_logic(cond : BOOLEAN) return std_logic is
        begin
            if cond then return '1'; else return '0'; end if;
        end function;
        -- the counters are initialized to position before a frame
        variable horizontal : unsigned(xcolumn'RANGE) := (3=>'0',others => '1'); -- ="1111110111"
        variable vertical   : unsigned(yrow'RANGE)    := to_unsigned(LCD_YROW_MAX, yrow'LENGTH); -- yrow counter
    begin -- process
        ---------- 1st part of LCD generator--------------------------------------------------
        if falling_edge(CLK_33MHz) then
            if horizontal >= LCD_XCOLUMN_MAX then
                if vertical < LCD_YROW_MAX then
                    vertical := vertical + 1;
                else
                    vertical := (others => '0');
                end if;
            end if;
            horizontal := horizontal + 1; -- unsigned counter overflows at its max value 
        end if;

        ------------ Storing results ---------------------------------------------		
        if not beginWriting then
            iswrite        := FALSE;
            repeat_counter := 0;
        elsif rising_edge(CLK_33MHz) then
            -------------------------------------------------------------------------
            if ( not iswrite and LCD_DE = '1') then -- frame is active
                file_open(fstatus, outfile, FILE_NAME, WRITE_MODE); -- file name
                iswrite           := TRUE;
                counter_of_frames := 0;
                repeat_counter    := 0;
                if fstatus = OPEN_OK then
                    write(outline, HEADER);
                    writeline(outfile, outline);
                end if;
            end if;
            if iswrite and fstatus = OPEN_OK then
                if (xcolumn = XYZERO) then
                    FLUSH_COMPRESSION(repeat_counter);
                    write(outline, NUMHEADER); -- we are adding mark for synchronization checks.
                    write(outline, to_integer(xcolumn));
                    write(outline, ',');
                    write(outline, to_integer(yrow));
                    writeline(outfile, outline);
                    lastdata := (others => '0');
                end if;
                -- LCD_DEN, XEND and YEND are not stored, because they are always determined from xcolumn and yrow values
                data := RGBcolor;
                if data = lastdata then
                    if ENABLE_COMPRESSION then
                        repeat_counter := repeat_counter + 1;
                    else
                        write(outline, '*');  -- previous value repeates 1x
                        writeline(outfile, outline);
                    end if;
                else
                    FLUSH_COMPRESSION(repeat_counter);
                    write(outline, integer'image(to_integer(unsigned(data))));
                    lastdata := data;
                    writeline(outfile, outline);
                end if;

                if XEND_N = '0' and YEND_N = '0' then -- the last pixel of frame
                    if repeat_counter > 0 then
                        FLUSH_COMPRESSION(repeat_counter);
                        writeline(outfile, outline);
                    end if;
                    if SINGLE_FRAME then
                        file_close(outfile);
                        iswrite := FALSE;
                        assert false report LF&LF&":-) OK end of SINGLE frame simulation."&LF
                        severity failure;
                      else
                        counter_of_frames := counter_of_frames + 1;
                        if counter_of_frames >= COUNT_OF_FRAMES then
                            assert false
                            report LF&LF&":-) OK end of simulation: " & integer'image(COUNT_OF_FRAMES) & " frames were stored."&LF
                            severity failure;
                        else
                            report "Simulation stored frame "&integer'image(counter_of_frames)&" of "&integer'image(COUNT_OF_FRAMES);
                        end if;
                    end if;
                end if;
                if iswrite and fstatus /= OPEN_OK then
                    assert false report "File status is not OPEN_OK"
                    severity failure;
                    file_close(outfile);
                    iswrite := FALSE;
                end if;
            end if;
            ----------------- 2nd part of LCD_generator  ------------------------	   
            XEND_N  <= boolean2std_logic(horizontal < LCD_XCOLUMN_MAX);
            YEND_N  <= boolean2std_logic(vertical < LCD_YROW_MAX);
            LCD_DE  <= boolean2std_logic((horizontal < LCD_XSCREEN) and (vertical < LCD_YSCREEN));
            xcolumn <= horizontal;
            yrow    <= vertical;
        end if; -- elsif rising_edge(CLK_33MHz)
        ---------------------------------------------------------------
    end process stimuls;

    --------------------------------------------------------------------------------------
    -- ***ToDo 3 of 3: Adjust to your own component
    iLCDLogic : LCDlogic0 port map(
         xcolumn=>xcolumn, yrow=>yrow, 
         XEND_N=>XEND_N, YEND_N=>YEND_N, 
         LCD_DE=>LCD_DE, LCD_DCLK=>LCD_DCLK,
         RGBcolor=>RGBcolor 
    ); 
    -- Note: Here, we preferred much safer VHDL-named associations that do not depend on the order of definitions. 
    -- The port map can also be written by risky positional associations that require exact order.
    -- iLCDLogic : LCDLogic port map(xcolumn, yrow, XEND_N, YEND_N, LCD_DE, LCD_DCLK, RGBcolor ); 
    
end architecture testbench;

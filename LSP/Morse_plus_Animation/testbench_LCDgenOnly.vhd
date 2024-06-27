-- testbench_LCDgenOnly is a simplified version of testbench_LCDlogic.
-- It contains only generator of LCD signal and does not write the result to a file
-------------------------------------------------------------
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta]
-- Published under GNU General Public License V2.1 2024
-------------------------------------------------------------
library ieee, work;
use ieee.std_logic_1164.all; use ieee.numeric_std.all;
use work.LCDpackage.all;

entity testbench_LCDgenOnly is --testbench entity is always without any inputs/outputs.
end entity;

architecture testbench of testbench_LCDgenOnly is
    constant COUNT_OF_FRAMES : positive := 3;
    -- The count of simulated frames. We test for 3 to check stability

    --------------------------------------------------------------
    -- LCDgenerator signals
    constant CLOCK_PERIOD  : time      := 30 ns; -- cca 33 MHz
    signal LCD_DCLK        : std_logic               := '0'; -- LCD pixel clock, 33 MHz
    signal LCD_DE          : std_logic               := '0'; -- DataEnable control signal of LCD controller
    signal CLRN            : std_logic               := '0'; -- Power-up initialization
    constant XYZERO        : unsigned(xy_t'range)  := (others => '0');
    signal xcolumn, yrow   : xy_t                 := XYZERO;
    signal XEND_N, YEND_N  : std_logic            := '1';
---------------------------------------------------------------------------------------------------
--------------------- YOUR TESTBENCH --------------------------------------------------------------
-- signals
begin -- of architecture -------------------------
-- commands
-- *** Single test
--    iDiv: entity work.LCDtestModulo
--        generic map(DIVN=>13)
--        port map(xcolumn=>xcolumn, yrow=>yrow,XEND_N=>XEND_N, YEND_N=>YEND_N,
--                 LCD_DE=>LCD_DE, LCD_DCLK=>LCD_DCLK, RGBcolor=>open);

-- *** The test of more generic values. For each one, we insert its separate instance.                 
-- if we generate 10 instances, then the simulation of 1 frame takes approx. 2 seconds.
-- You can extend range to any positive interval
igen: for i in 33 to 43 generate
    -- for more, see https://dcenet.fel.cvut.cz/edu/fpga/doc/UvodDoVHDL1_concurrent_V20.pdf#page=35              
                iDiv: entity work.LCDtestModulo -- here, iDiv will be automatically changed by compiler 
                generic map(DIVN=>i)
                port map(xcolumn=>xcolumn, yrow=>yrow,XEND_N=>XEND_N, YEND_N=>YEND_N, 
                    LCD_DE=>LCD_DE, LCD_DCLK=>LCD_DCLK,
                    RGBcolor=>open); -- RGBcolor is unconnected
          end generate;

-----------------------END OF YOUR TESTBENCH -------------------------------------------------------
----------------------------------------------------------------------------------------------------

-- ============ LCDgenerator + termination of simulation =============================================== 
    clk_process : process     -- LCD_DCLK process with 50% duty cycle
    begin
        LCD_DCLK <= '0';
        wait for CLOCK_PERIOD / 2;      --for 50 % of CLK_33MHz period is'0'.
        LCD_DCLK <= '1';
        wait for CLOCK_PERIOD / 2;      --for next 50% of CLK_33MHz period is '1'.
    end process;

    clear_process : process     -- Clear process keeps CLRN='0' for 3 CLOCK_PERIOD.
    begin
        CLRN<='0';
        wait for 3*CLOCK_PERIOD;
        CLRN<='1';
        wait;           -- endless wait terminates this process.
    end process;

    iLCDgenerator : process(LCD_DCLK) -- generator of LCD clocks and signals
        -- the boolean to std_logic conversion
        function boolean2std_logic(cond : BOOLEAN) return std_logic is
        begin
            if cond then return '1'; else return '0'; end if;
        end function;
        -- the counters are initialized before the 1st frame after power-up
        variable horizontal : unsigned(xcolumn'RANGE) := (3=>'0',others => '1'); -- ="1111110111"=1015
        variable vertical   : unsigned(yrow'RANGE)    := to_unsigned(YROW_MAX, yrow'LENGTH); -- =524
        variable wasDE : boolean :=FALSE;
        variable counter_of_frames : integer   := 0;
    begin -- process
        ---------- 1st part of LCD generator--------------------------------------------------
        if falling_edge(LCD_DCLK) then
            if horizontal >= XCOLUMN_MAX then
                if vertical < YROW_MAX then  vertical := vertical + 1;
                else vertical := XYZERO;
                end if;
            end if;
            horizontal := horizontal + 1; -- unsigned counter overflows at its max value 
            if LCD_DE='1' then wasDE:=TRUE; end if; -- 1st frame already started
            if wasDE and (XEND_N or YEND_N)='0' then -- the last pixel
                counter_of_frames := counter_of_frames + 1;
                if counter_of_frames >= COUNT_OF_FRAMES then
                    assert false
                    report LF&LF&":-) OK end of simulation: " & integer'image(COUNT_OF_FRAMES) & " frames were done."&LF
                    severity failure;
                else
                    report "Simulation frame "&integer'image(counter_of_frames)&" of "&integer'image(COUNT_OF_FRAMES);
                end if;
            end if;
        end if; -- falling_edge(LCD_DCLK)
        ----------------- 2nd part of LCD_generator  ------------------------       
        if rising_edge(LCD_DCLK) then
            XEND_N  <= boolean2std_logic(horizontal < XCOLUMN_MAX);
            YEND_N  <= boolean2std_logic(vertical < YROW_MAX);
            LCD_DE  <= boolean2std_logic((horizontal < LCD_XSCREEN) and (vertical < LCD_YSCREEN));
            xcolumn <= horizontal;
            yrow    <= vertical;
        end if; -- rising_edge(CLK_33MHz)
        ---------------------------------------------------------------
    end process iLCDgenerator;
end architecture testbench;

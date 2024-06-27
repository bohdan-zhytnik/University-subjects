-- Start Stop Finite State Machine -----------------------------------------------------------------------
-----------------------------------------------------------------------------------------
-- Library for Logic Systems and Processors, CTU-FFE Prague, Dept. of Control Eng., Richard Susta
-- Published under GNU General Public License
-------------------------------------------------------------

library ieee; use ieee.std_logic_1164.all; 
entity StartStop_FSM is
      port( STOP  : in std_logic; -- set RUN to 0
		      START : in std_logic; -- set RUN to 1, if not STOP and not RESET
				CLK :in std_logic;    -- input clock any higher frequency
            RUN: out std_logic);  -- the signalization of the Running state
end entity;

architecture rtl of StartStop_FSM is
begin -- architecture
iStartStop: process(CLK)
variable Running : std_logic:='0';
begin 
	if rising_edge(CLK) then
		if STOP then Running:='0';
		elsif START='1' then Running:='1';  
		end if;
	end if;
	RUN<=Running;
end process;
end architecture; 

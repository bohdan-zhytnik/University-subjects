-- Shift register of 18 LEDR -----------------------------------------------------------------------
-----------------------------------------------------------------------------------------
library ieee; use ieee.std_logic_1164.all; 
entity  ShiftLeft18 is
   port( SI :in std_logic;  -- Serial Input
	      Load :in std_logic; -- Load Data value on Rising edge of CLK
			Data : in std_logic_vector(17 downto 0); -- Data to be loaded
         CLK :in std_logic;  -- clock input
         LEDR: out std_logic_vector(17 downto 0));
end entity;
architecture rtl of ShiftLeft18 is
begin -- architecture
pshift: process(CLK)
			variable rg : std_logic_vector(LEDR'RANGE):=(others=>'0');
			begin 
			  if rising_edge(CLK) then  
			     if LOAD then rg:=Data;
				  else rg := rg(LEDR'LEFT-1 downto 0) & SI;
				  end if;
			  end if;
			LEDR<=rg;
        end process;
end architecture; 

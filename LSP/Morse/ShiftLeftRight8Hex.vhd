-- Shift register for HEX0 to HEX7 pins ----------------------------------------------------------------------
-----------------------------------------------------------------------------------------
library ieee; use ieee.std_logic_1164.all; 
entity  ShiftLeftRight8Hex is
   port( SI :in std_logic_vector(6 downto 0);  -- Serial Input
	      RShift : in std_logic; -- Shift Right
			RESET  : in std_logic;
         CLK :in std_logic;  -- clock input
         HEX7,HEX6, HEX5, HEX4, HEX3, HEX2, HEX1, HEX0: out std_logic_vector(6 downto 0));
end entity;

architecture rtl of ShiftLeftRight8Hex is
type HShift_t is array(7 downto 0) of std_logic_vector(6 downto 0);
constant ONES : HShift_t :=(others=>(others=>'1')); -- HEX are off on '1'
begin -- architecture
pshift: process(CLK)
			variable rg : HShift_t:=ONES;
			begin 
			  if rising_edge(CLK) then  
			      if RESET then rg:=ONES;
					elsif RShift then rg := SI & rg(rg'LEFT downto 1);
				   else rg := rg(rg'LEFT-1 downto 0) & SI;
				   end if;
			  end if;
			HEX7<=rg(7); HEX6<=rg(6); HEX5<=rg(5); HEX4<=rg(4); 
			HEX3<=rg(3); HEX2<=rg(2); HEX1<=rg(1); HEX0<=rg(0);
        end process;
end architecture; 

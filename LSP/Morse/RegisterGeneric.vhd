-- Data Register -----------------------------------------------------------------------
-----------------------------------------------------------------------------------------
library ieee; use ieee.std_logic_1164.all; 
entity  RegisterGeneric is
   generic(WBus:integer:=8);  -- Width of Data bus
   port( Data : in std_logic_vector(WBus-1 downto 0); -- data to be loaded
         RESET :in std_logic; -- on rising edge of clk clear Q
        	Enable :in std_logic; -- If ENABLE='1' and RESET='0' then the Data are stored in Q
			CLK :in std_logic;  -- clock input
         Q: out std_logic_vector(WBus-1 downto 0));
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert WBus>0 report "Excepted the width of buses WBus>0" severity ERROR;	
end entity;

architecture rtl of RegisterGeneric is
begin -- architecture
pshift: process(CLK)
			variable rg : std_logic_vector(Q'RANGE):=(others=>'0');
			begin 
			  if rising_edge(CLK) then  
			     if RESET then rg:=(others=>'0'); -- assign '0' to all bits
				  elsif Enable then rg:=Data;
			     end if;
			  end if;
			  Q<=rg;
        end process;
end architecture; 

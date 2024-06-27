-- Counter with width of bit width defined by generic parameter--------------------------
-----------------------------------------------------------------------------------------
library ieee; use ieee.std_logic_1164.all; use ieee.numeric_std.all;

entity CounterUpGeneric is
   generic(WBus:integer:=8 --Width (Length) of Q bus
	        );
   port(RESET, ENABLE, CLK : in std_logic;
        Q : out std_logic_vector(WBus-1 downto 0));
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert WBus>0 report "Excepted the width of bus WBus>0" severity ERROR;	
end entity;

architecture rtl of CounterUpGeneric is
begin
  process (CLK)   
  variable  cnt : unsigned(Q'RANGE);
  begin
	  if rising_edge(CLK) then
		  if RESET then cnt:=(others=>'0'); 
		  elsif ENABLE then cnt := cnt + 1; 
		  end if;
	  end if;
	  Q<=std_logic_vector(cnt);
  end process;
end architecture;
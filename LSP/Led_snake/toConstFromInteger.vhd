-- Multiplexer of 2 buses -----------------------------------------------------------------------
-------------------------------------------------------------
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta]
-- Published under GNU General Public License
-------------------------------------------------------------
library ieee;use ieee.std_logic_1164.all;use ieee.numeric_std.all;


entity toConstFromInteger is
	generic -- definition of constants that can be parameterized in instances of circuits
	(	N : integer:= 165;  
		WBus : natural := 8); -- the width of bus
	port (	Q : out std_logic_vector(Wbus-1 downto 0));
	
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert WBus>0 report "Excepted the width of Q bus WBus>0" severity ERROR;	
end entity;

architecture behavorial of toConstFromInteger is
begin
 	Q<=std_logic_vector(to_unsigned(N,Q'LENGTH)); 
end architecture;

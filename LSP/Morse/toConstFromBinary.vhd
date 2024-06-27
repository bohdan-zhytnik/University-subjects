-- Multiplexer of 2 buses -----------------------------------------------------------------------
-------------------------------------------------------------
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta]
-- Published under GNU General Public License
-------------------------------------------------------------
library ieee;use ieee.std_logic_1164.all;use ieee.numeric_std.all;


entity toConstFromBinary is
	generic -- definition of constants that can be parameterized in instances of circuits
	(	N : string:= "1011000";  
		WBus : natural := 8); -- the width of bus
	port (	Q : out std_logic_vector(Wbus-1 downto 0));
	
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert WBus>0 report "Excepted the width of Q bus WBus>0" severity ERROR;	
		tloop: for i in N'RANGE generate
		       assert N(i)='1' or N(i)='0' report "Expected string N of '0' or '1' characters!"
		       severity error;
	          end generate;			 
end entity;

architecture behavorial of toConstFromBinary is
signal r:std_logic_vector(N'LOW to N'HIGH);
signal x:integer range 0 to 2**WBus-1;
begin
  iloop: for i in r'RANGE generate
	  r(i) <= '1' when N(i)='1' else '0';
	end generate;
	-- we adapt possible different lengths by the double conversion
	x<=to_integer(unsigned(r));
 	Q<=std_logic_vector(to_unsigned(x,Q'LENGTH)); 
end architecture;

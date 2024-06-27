-- Multiplexer of 2 buses 
-----------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;


entity Multiplexer2x1bus is

	generic ( WBUS : natural := 16 ); -- WIDTH (LENGTH) of data Bus

	port -- inputs and outputs
	(	Bus0	    : in std_logic_vector((WBUS-1) downto 0);
		Bus1 	 : in std_logic_vector((WBUS-1) downto 0);
		sel    : in std_logic;
		Y      : out std_logic_vector((WBUS-1) downto 0));
	
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert WBus>0 report "Excepted the width of buses WBus>0" severity ERROR;	
end entity;

architecture behavorial of Multiplexer2x1bus is
begin
	y <= Bus0 when sel='0' else Bus1; 
end architecture;

-- Multiplexer of 2 buses -----------------------------------------------------------------------
-----------------------------------------------------------------------------------------
library ieee;use ieee.std_logic_1164.all;


entity Multiplexer4x1bus is

	generic -- definition of constants that can be parameterized in instances of circuits
	(	WBus : natural := 16 ); -- The width of input buses

	port -- inputs and outputs
	(
		Bus0, Bus1, Bus2, Bus3 : in std_logic_vector((WBUS-1) downto 0);
		sel    : in std_logic_vector(1 downto 0);
		Y      : out std_logic_vector((WBUS-1) downto 0));
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert WBus>0 report "Excepted the width of buses WBus>0" severity ERROR;	
end entity;

architecture behavorial of Multiplexer4x1bus is
begin
	with sel select
 	   y <= Bus0 when "00",
	        Bus1 when "01",
			  Bus2 when "10",
			  Bus3 when others; 
end architecture;

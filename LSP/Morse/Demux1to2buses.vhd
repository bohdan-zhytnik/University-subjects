-- Demux aka Demultiplexer 1:2 
-------------------------------------------
library ieee;use ieee.std_logic_1164.all;

entity Demux1to2buses is
	generic ( WBUS : natural := 16 ); -- WIDTH (LENGTH) of data Buses
	port -- inputs and outputs
	(	Data   : in std_logic_vector((WBUS-1) downto 0); -- data input
	   sel    : in std_logic;
		Bus0	 : out std_logic_vector((WBUS-1) downto 0);
		Bus1 	 : out std_logic_vector((WBUS-1) downto 0));
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert WBus>0 report "Excepted the width of buses WBus>0" severity ERROR;	
end entity;

architecture behavorial of Demux1to2buses is
constant ZERO : std_logic_vector((WBUS-1) downto 0):=(others=>'0');
begin
	Bus0 <= data when sel='0' else ZERO; 
	Bus1 <= data when sel='1' else ZERO; 
end architecture;
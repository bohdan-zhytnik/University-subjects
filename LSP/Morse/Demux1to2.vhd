-- Demux aka Demultiplexer 1:2 
-------------------------------------------
library ieee;use ieee.std_logic_1164.all;

entity Demux1to2 is
	port -- inputs and outputs
	(  Data   : in std_logic; -- data input
		sel    : in std_logic; -- selection of output
		X0	    : out std_logic;
		X1 	 : out std_logic);
end entity;

architecture behavorial of Demux1to2 is
begin
	X0 <= data when sel='0' else '0'; 
	X1 <= data when sel='1' else '0'; 
end architecture;
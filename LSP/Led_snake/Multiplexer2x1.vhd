-- Multiplexer of 2 inputs 
-------------------------------------------
library ieee;use ieee.std_logic_1164.all;

entity Multiplexer2x1 is
	port -- inputs and outputs
	(  X0	    : in std_logic;
		X1 	 : in std_logic;
		sel    : in std_logic;
		Y      : out std_logic);
end entity;

architecture behavorial of Multiplexer2x1 is
begin
	Y <= X0 when sel='0' else X1; 
end architecture;
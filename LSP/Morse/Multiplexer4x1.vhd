-- Multiplexer of 4 inputs 
-------------------------------------------
library ieee;use ieee.std_logic_1164.all;


entity Multiplexer4x1 is
	port -- inputs and outputs
	(
		X0	    : in std_logic;
		X1 	 : in std_logic;
		X2	    : in std_logic;
		X3 	 : in std_logic;
		sel    : in std_logic_vector(1 downto 0);
		Y      : out std_logic);
end entity;

architecture behavorial of Multiplexer4x1 is
begin
	with sel select
 	   y <= X0 when "00",
	        X1 when "01",
			  X2 when "10",
			  X3 when others; 
end architecture;
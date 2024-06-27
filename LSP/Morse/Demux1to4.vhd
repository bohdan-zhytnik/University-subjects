-- Demux aka Demultiplexer 1:2 
-------------------------------------------
library ieee;use ieee.std_logic_1164.all;

entity Demux1to4 is
	port -- inputs and outputs
	(  Data   : in std_logic; -- data input
	   sel    : in std_logic_vector(1 downto 0); -- selection of output
		X0, X1, X2, X3 : out std_logic);
end entity;

architecture behavorial of Demux1to4 is
begin
	X0 <= data when sel="00" else '0'; 
	X1 <= data when sel="01" else '0'; 
	X2 <= data when sel="10" else '0'; 
	X3 <= data when sel="11" else '0'; 
end architecture;
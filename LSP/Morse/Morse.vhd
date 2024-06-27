library ieee; use ieee.std_logic_1164.all;
entity Morse is
	port
	(		X    : in std_logic_vector(3 downto 0);	
		STOP, Y : out std_logic );
	end entity;
	
	architecture behavioral of Morse is
	begin
		Y <= (X(0) or X(3)) and (not X(1) or X(2));
		STOP <= X(3) and X(1) and X(0);
	end architecture;
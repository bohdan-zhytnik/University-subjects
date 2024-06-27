-- 2 input bases are joined to 1 -----------------------------------------------------------------------
library ieee; use ieee.std_logic_1164.all;
entity toBusFrom2Buses is
	generic -- definition of constants that can be parameterized in instances of circuits
	(W2 : natural := 8; -- the width of bus2
	 W1 : natural := 8); -- the width of bus1

	port -- inputs and outputs
	(	Bus2 : in std_logic_vector((W2-1) downto 0);
	   Bus1 : in std_logic_vector((W1-1) downto 0);
		Y : out std_logic_vector(((W2+W1)-1) downto 0));
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert W1>0 and W2>0 report "Excepted the width of buses W1>0 and W2>0" severity ERROR;	
end entity;

architecture behavorial of toBusFrom2Buses is
constant M:integer:=Y'LEFT-(W2-1);  
begin
	Y(Y'LEFT downto M)<=Bus2;
	Y(M-1 downto 0)<=Bus1;
end architecture;

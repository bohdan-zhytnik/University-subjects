-- 2 input bases are joined to 1 -----------------------------------------------------------------------
library ieee; use ieee.std_logic_1164.all;
entity toBuses2From1Bus is
	generic -- definition of constants that can be parameterized in instances of circuits
	(W2 : natural := 8; -- the width of bus2 (upper)
	 W1 : natural := 8); -- the width of bus1 (lower)

	port -- inputs and outputs
	(	X : in std_logic_vector(((W2+W1)-1) downto 0);
	   Bus2 : out std_logic_vector((W2-1) downto 0);
	   Bus1 : out std_logic_vector((W1-1) downto 0));
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert W1>0 and W2>0 report "Excepted the width of buses W1>0 and W2>0" severity ERROR;	
end entity;

architecture behavorial of toBuses2From1Bus is
constant M:integer:=X'LEFT-(W2-1);  
begin
	Bus2<=X(X'LEFT downto M);
	Bus1<=X(M-1 downto 0);
end architecture;

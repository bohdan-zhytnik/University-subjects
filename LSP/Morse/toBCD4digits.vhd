-- The entity convert bin input to 2 BCD digits
-------------------------------------------------------------
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta]
-- Published under GNU General Public License
-------------------------------------------------------------

library ieee;use ieee.std_logic_1164.all;use ieee.numeric_std.all;

entity toBCD4digits is
generic(WBus:integer:=12); -- Left Bound of input range 
port(  bin : in std_logic_vector((Wbus-1) downto 0);
       bcd3, bcd2, bcd1, bcd0 : out std_logic_vector(3 downto 0));
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert WBus>0 report "Excepted the width of bin bus WBus>0" severity ERROR;	
end entity;

architecture rtl of toBCD4digits is

component toBCDgeneric IS 
   generic(N_LENGHT:integer:=20; BCD_digits:integer:=7);
	port(N : in std_logic_vector(N_LENGHT-1 downto 0);
		BCD : out std_logic_vector(BCD_digits*4-1 downto 0));
end component;
signal bcdout : std_logic_vector(15 downto 0);
begin
 ibcd : ToBCDgeneric generic map(N_LENGHT=>WBus, BCD_digits=>4) port map(N=>bin, BCD=>bcdout);
 bcd3<=bcdout(15 downto 12); 
 bcd2<=bcdout(11 downto 8); 
 bcd1<=bcdout(7 downto 4); 
 bcd0<=bcdout(3 downto 0);				  
end architecture;
					  
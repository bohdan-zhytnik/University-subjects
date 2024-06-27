-- Compare bus value with unsigned constant  -----------------------------------------------------------------------
-----------------------------------------------------------------------------------------
-- Library for Logic Systems and Processors, CTU-FFE Prague, Dept. of Control Eng., Richard Susta
-- Published under GNU General Public License
-------------------------------------------------------------
library ieee;use ieee.std_logic_1164.all;use ieee.numeric_std.all;

entity CompareUnsigned is
	generic -- definition of constants that can be parameterized in instances of circuits
	(	WBus : natural := 8); -- the Width of output bus
	port -- inputs and outputs
	(
		X : in std_logic_vector(WBus-1 downto 0);
		Y : in std_logic_vector(WBus-1 downto 0);
		xGTy : out std_logic;  -- X>Y: X is Great Than Y
		xEQy  : out std_logic; -- X=Y  X is Equal to Y
		xLTy : out std_logic); -- X>Y: X is Less Y
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert WBus>0 report "Excepted the width of buses WBus>0" severity ERROR;	
end entity;

architecture behavorial of CompareUnsigned is
signal gt,lt : boolean;
signal xi, yi : integer range 0 to 2**WBus-1;
begin
   xi <= to_integer(unsigned(X)); yi <= to_integer(unsigned(Y));
	gt <= (xi > yi); lt<=(xi < yi);
	
 	xGTy <= '1'  when gt else '0';
 	xLTy <= '1'  when lt else '0';
	xEQy  <= '1' when not (lt or gt) else '0';
end architecture;

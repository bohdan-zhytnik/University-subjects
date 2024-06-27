-- Compare bus value with unsigned constant  -----------------------------------------------------------------------
-----------------------------------------------------------------------------------------
-- Library for Logic Systems and Processors, CTU-FFE Prague, Dept. of Control Eng., Richard Susta
-- Published under GNU General Public License
-------------------------------------------------------------
library ieee;use ieee.std_logic_1164.all;use ieee.numeric_std.all;

entity CompareWithUnsigned is
	generic -- definition of constants that can be parameterized in instances of circuits
	(	WBus : natural := 8; -- the Width of output bus
	   unsignedConstant : natural:= 165); -- i.e. X"A5" 

	port -- inputs and outputs
	(	X : in std_logic_vector(WBus-1 downto 0);
		xGTu : out std_logic;  -- x>uConst: x is Great Than unsignedConstant
		xEQu  : out std_logic; -- x=uConst x is Equal to unsignedConstant
		xLTu : out std_logic); -- x>uConst: x is Less Than unsignedConstant
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert WBus>0 report "Excepted width of bus WBus>0" severity ERROR;	
		assert unsignedConstant>=0
		report "Expected width of output bus WBus>0"
		severity error;
end entity;

architecture behavorial of CompareWithUnsigned is
signal gt,lt : boolean;
signal xi : integer range 0 to 2**WBus-1;
begin
   xi <= to_integer(unsigned(X));
	gt <= xi > unsignedConstant; lt<=xi<unsignedConstant;
	
 	xGTu <= '1' when gt else '0';
 	xLTu <= '1' when lt else '0';
	xEQu  <= '1' when not (lt or gt) else '0';
end architecture;

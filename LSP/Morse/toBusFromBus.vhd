-- It interconnects buses with different widths.
-- All possible upper unused outputs are set to '0's. 
-----------------------------------------------------------------------
library ieee; use ieee.std_logic_1164.all;use ieee.numeric_std.all;
entity toBusFromBus is
	generic -- Definitions of constants that can be parameterized in instances of the circuit
	(Win : natural := 12;-- the width of input bus
	 Wout : natural := 8); -- the width of output bus

	port -- inputs and outputs
	(	BusIn : in std_logic_vector((Win-1) downto 0);
	   BusOut : out std_logic_vector((Wout-1) downto 0)
	);
-- The following optional part of the entity is called the passive process 
-- and checks at compile time whether the values of the generic parameters are meaningful.
   begin 
		assert Win>0 and Wout>0 report "Excepted the width of buses Win>0 and Wout>0" severity ERROR;	
end entity;

architecture behavorial of toBusFromBus is

-- VHDL2008 knows maximum function, but Quartus compiler does not support its usage yet.
function maximum(x,y:integer) return integer is
begin
-- Note: The >= comparission is easier implemented in hardware than > 
    if x>=y then return x; else return y; end if; 
end function;
-- In VHDL, constants can be initialized by functions, unlike in C language.
constant M:integer:=maximum(Win,Wout); 

begin
   -- We need a process because it allows multiple assignments of values
	process(BusIn)
   variable R : std_logic_vector(M-1 downto 0);
	begin 
	  R:=(others=>'0'); -- we clear all values
	  R(BusIn'RANGE):=BusIn; -- we fill all input bits
	  BusOut<=R(BusOut'RANGE); -- we assign all output bits
	end process;
end architecture;

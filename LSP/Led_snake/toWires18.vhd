-- Convert bus to 18 wires
library ieee; use ieee.std_logic_1164.all; use ieee.numeric_std.all;
entity toWires18 is
 port 
  (   Y : in std_logic_vector(17 downto 0);
	  X17,X16,X15,X14,X13,X12,X11,X10,X9,X8,X7,X6,X5,X4,X3,X2,X1,X0:out std_logic);
end entity;

architecture rtl of toWires18 is

begin -- architecture
-- Note: The result of concurrent asignments does not depend on their order.
   X0<=Y(0);  X1<=Y(1);  X2<=Y(2);  X3<=Y(3);  X4<=Y(4);  X5<=Y(5);
	X6<=Y(6);  X7<=Y(7);  X8<=Y(8);  X9<=Y(9);  X10<=Y(10);X11<=Y(11);
	X12<=Y(12);X13<=Y(13);X14<=Y(14);X15<=Y(15);X16<=Y(16);X17<=Y(17);
end architecture;


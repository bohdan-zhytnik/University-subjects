library ieee; use ieee.std_logic_1164.all; use ieee.numeric_std.all;
entity toBus18 is
 port 
  (  X17,X16,X15,X14,X13,X12,X11,X10,X9,X8,X7,X6,X5,X4,X3,X2,X1,X0:in std_logic;
     Y : out std_logic_vector(17 downto 0));
end entity;

architecture rtl of toBus18 is

begin -- architecture
   Y(0)<=X0;Y(1)<=X1;Y(2)<=X2;Y(3)<=X3;Y(4)<=X4;Y(5)<=X5;
	Y(6)<=X6;Y(7)<=X7;Y(8)<=X8; Y(9)<=X9;Y(10)<=X10;Y(11)<=X11;
	Y(12)<=X12;Y(13)<=X13;Y(14)<=X14;Y(15)<=X15;Y(16)<=X16;Y(17)<=X17;
end architecture;

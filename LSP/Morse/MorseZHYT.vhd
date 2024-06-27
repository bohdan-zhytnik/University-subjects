library ieee; use ieee.std_logic_1164.all; 
entity MorseZHYT is 
port
(X:in std_logic_vector(5 downto 0);
	STOP, Y : out std_logic);
end entity;

architecture behavioral of MorseZHYT is
signal a,b,c,d,e,f:std_logic;
signal Y00,Y01,Y10,Y11 : std_logic;
begin
a<=X(5); b<=X(4); c <= X(3); d <= X(2); e <= X(1); f <= X(0);
Y00 <= (e and not c) or (e and f) or (f and not c) or (not d and f);
Y01 <=(e and c and not d) or (f and not e) or (f and e and not c and not d) or (e and f and c and d);
Y10 <= (not c and not e)  or (f and not d) or (e and c and(not d or f));
Y11 <=(not e and not c and not d) or (f and not c and not d);

with X(5 downto 4) select
Y <= Y00 when "00",Y01 when "01",Y10 when "10",Y11 when others;
Stop <= a and b and f and d	;
end architecture;
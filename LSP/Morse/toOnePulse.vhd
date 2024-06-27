-- When X is in '1' for DIGITAL_FILTER_LENGTH
-- then Y output is '1' only for 1 rising edge of CLK clock
library ieee; use ieee.std_logic_1164.all;
entity toOnePulse is
 generic(DIGITAL_FILTER_LENGTH:natural:=4);
 port ( X : in std_logic;  -- input signal
  CLK : in std_logic;      -- clock
  Y : out std_logic);
begin 
  assert DIGITAL_FILTER_LENGTH>=0 report "Expected DIGITAL_FILTER_LENGTH>=0" severity ERROR;   
  assert DIGITAL_FILTER_LENGTH>0 report "Input X is not filted due to DIGITAL_FILTER_LENGTH=0" severity WARNING;   
end entity;

architecture behavioral of toOnePulse is
begin
 ifilter : process(CLK)
  constant ZERO : std_logic_vector(0 to DIGITAL_FILTER_LENGTH-1):=(others=>'0');
  constant ONES : std_logic_vector(ZERO'RANGE):=(others=>'1');
  variable memory : std_logic_vector(ZERO'RANGE):=ZERO;
  variable qfilter, qfilterLast: std_logic:='0';
  begin
     if rising_edge(CLK) then 
	   -- we implement hysteresis, we need DIGITAL_FILTER_LENGTH '1's to go to '1'
      -- from which we return to '0' on DIGITAL_FILTER_LENGTH '0's
      if DIGITAL_FILTER_LENGTH>0 then
	      if memory=ONES then qfilter:='1'; 
		   elsif memory=ZERO then qfilter:='0';
		   end if; 
		   memory:=X & memory(0 to memory'RIGHT-1); -- shift
 	   else qfilter:=X;
	   end if;
      if qfilter='1' and qfilterLast='0' then Y<='1';else Y<='0';end if; -- rising edge detection
		qfilterLast:=qfilter;
    end if;
  end process;
end architecture;
-- The entity converts N input to the vector BCD containing BCD digits.
-------------------------------------------------------------------------
-- Its algorithm is explained in the 6.4 chapter of Richard Susta's textbook:
-- Eng: Logic Circuits On FPGAs  |  Cz: Logicke obvody na FPGA
-- available at https://dcenet.fel.cvut.cz/edu/fpga/guides.aspx 
-- Switch the page to CZ for its Czech version.
-------------------------------------------------------------
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta]
-- Published under GNU General Public License
-------------------------------------------------------------
library ieee; use ieee.std_logic_1164.all; use ieee.numeric_std.all;

entity toBCDgeneric IS 
   generic(N_LENGHT:integer:=20; -- the length of input number N
	        BCD_digits:integer:=7); -- the count of output BCD digits, each has 4 bits
	port(N : in std_logic_vector(N_LENGHT-1 downto 0);
		BCD : out std_logic_vector(BCD_digits*4-1 downto 0));
	begin 
	assert N_LENGHT>=4 and N_LENGHT<=32 
	report "Required N_LENGHT from 4 to 32" severity failure;
	assert BCD_digits>0 and BCD_digits<=10 
	report "Required BCD_digits from 1 to 10" severity failure;
end entity;

architecture rtl of toBCDgeneric is
begin

 show : process(N)  
 
	subtype bcd_digit_type is std_logic_vector(3 downto 0);
   function adjust(x:bcd_digit_type) return bcd_digit_type is
     begin
        if unsigned(x)>"0100" then return bcd_digit_type(unsigned(x)+3);
	     else return x;
	     end if;
    end function;

    -- convert unsigned integer to BCD digits
	function binary2bcd(number : std_logic_vector) return std_logic_vector is
    variable hex_src : std_logic_vector (N'RANGE);
    variable bcd : std_logic_vector(BCD'RANGE) ;
    begin
        hex_src := number;
        bcd     := (others => '0') ;
    iloop:    for i in 0 to hex_src'LENGTH-1 loop
		    jloop:  for j in 0 to BCD'LENGTH/4-1 loop
                      bcd(4*j+3 downto 4*j):=adjust(bcd(4*j+3 downto 4*j));
                  end loop jloop;
				  
            bcd := bcd(bcd'LEFT-1 downto 0) & hex_src(hex_src'LEFT) ; -- shift bcd + 1 new entry
            hex_src := hex_src(hex_src'LEFT-1 downto 0) & '0' ; -- shift src + pad with 0
        end loop iloop;
        return std_logic_vector(bcd);
    end function;
 begin
   
   BCD<=binary2bcd(N); -- unsigned to BCD
	
  end process;
end architecture;
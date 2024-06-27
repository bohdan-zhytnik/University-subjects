-------------------------------------------------------------
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta], Published under GNU General Public License
-------------------------------------------------------------
-- The file is intended for testbench only. It is not synthesizable!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
library ieee, work;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for integer and unsigned types
use work.LCDpackage.all;

entity LCDtestModulo  is
    generic(DIVN:integer:=13);  -- we have moved constant DIVN to the generic section
    port(xcolumn  : in  xy_t      := XY_ZERO; -- x-coordinate of pixel (column index)
         yrow     : in  xy_t      := XY_ZERO; -- y-coordinate of pixel (row index)
         XEND_N   : in  std_logic := '0'; -- '0' only when max xcolumn, otherwise '1' 
         YEND_N   : in  std_logic := '0'; -- '0' only when max yrow
         LCD_DE   : in  std_logic := '0'; -- DataEnable control signal of LCD controller
         LCD_DCLK : in  std_logic := '0'; -- LCD data clock, 33 MHz
         RGBcolor : out RGB_t);         --  color data type RGB_t = std_logic_vector(23 downto 0), defined in LCDpackage
end entity;

architecture behavioral of LCDtestModulo is
    subtype modN_t is integer range 0 to DIVN-1;
    subtype divN_t is integer range 0 to XCOLUMN_MAX/DIVN+1;
    signal xmN:modN_t:=0;
    signal xdN:divN_t:=0;

begin  -- architecture 
    RGBcolor <= BLACK;
    
imodulo: process(LCD_DCLK)
			variable cntr : modN_t:=0; 
			variable ix   : divN_t:=0;
			begin
				if falling_edge(LCD_DCLK) then
					if (YEND_N and XEND_N)='1' then -- no row/column end signals
						if cntr<DIVN-1 then cntr := cntr+1; else cntr:= 0; ix:=ix+1; end if;
					else cntr:= 0; ix:=0; 
					end if;
				end if;
				if rising_edge(LCD_DCLK) then xmN<= cntr; xdN<=ix; 
				end if;
			end process;
			
     -- process compares the results    
   iTest : process(LCD_DCLK)
         variable x, y: integer;
         function ii(n:integer) return string is begin return integer'Image(n); end function;
	    begin
		     if falling_edge(LCD_DCLK) then
		         if LCD_DE='1' then -- visible part
   	              x := to_integer(xcolumn); y := to_integer(yrow); -- converting unsigned to integers
					  -- we are testing compliance. In simulation, we can divide by any number.
                 assert (x mod DIVN)=xmN and (x / DIVN)=xdN
				     report "In IDIV="&ii(DIVN)&": Wrong mod="&ii(xmN) &" div="&ii(xdN)&" for x="&ii(x)
				     severity Error;
				 end if;
			   end if;
		end process;	

end architecture;


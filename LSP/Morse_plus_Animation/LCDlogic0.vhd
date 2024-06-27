-------------------------------------------------------------
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta], Published under GNU General Public License
-------------------------------------------------------------
-- The original LCDlogic0
library ieee, work; use ieee.std_logic_1164.all; use ieee.numeric_std.all; -- for integer and unsigned types
use work.LCDpackage.all;

entity LCDlogic0 is
    port( xcolumn  : in  xy_t      := XY_ZERO; -- x-coordinate of pixel (column index)
         yrow     : in  xy_t          := XY_ZERO; -- y-coordinate of pixel (row index)
         XEND_N   : in  std_logic := '0'; -- '0' only when max xcolumn, otherwise '1' 
         YEND_N   : in  std_logic := '0'; -- '0' only when max yrow
         LCD_DE   : in  std_logic := '0'; -- DataEnable control signal of LCD controller
         LCD_DCLK : in  std_logic := '0'; -- LCD data clock, 33 MHz, see note 2 below
         RGBcolor : out RGB_t     ); --  color data type RGB_t = std_logic_vector(23 downto 0), defined in LCDpackage
end;

architecture behavioral of LCDlogic0 is
    -- We append additional colors to the  Windows 16 color palette defined in the LCDpackage.vhd.
    constant DARKBLUE   : RGB_t := ToRGB(0, 0, 139);
    constant GOLDENROD  : RGB_t := X"DAA520";
begin -- architecture 
    -- The sensitive list of the process specifies signals; after their changes, the outputs can change.  
    LSPimage : process(xcolumn, yrow, LCD_DE)
        variable RGB    :RGB_t:=BLACK; -- the color of pixel [xcolumn, yrow], initialized to 0.
        variable xmod32 :integer range 0 to 31 :=0; -- reminder after 32 division
        variable meo    :integer range 0 to 15 :=0;
        variable x, y   : integer:=0; -- integer xcolumn and yrow
        -- We added ranges to xmod32 and meo as hints to the compiler.
        --Note: In processes, all variable initializations must be performed inside our code, i.e., after its 'begin' keyword.
        --The initializations after variables are only for simulations. They are ignored in synthesis.
    begin -- process
        x := to_integer(xcolumn); y := to_integer(yrow); -- convert unsigned to integers
        xmod32 := x mod 32; --  -- powers of 2 are only allowed in modulo
        if xcolumn(5)='1' then meo:=(31-xmod32)/2; else meo:=xmod32/2; end if; -- odd, even line equations
        ------------------------------------------------------------
        if LCD_DE = '0' then  RGB := LIGTHGRAY;   --outside of the visible frame
        elsif y>=(240-meo) and y<=(240+meo) then RGB := YELLOW;
        -- Our LCD screen has sizes W*H=800*480, i.e., the ratio 5:3, 
        -- its half sizes have GCD (greatest common divisor) = 80 (400/80=5, 240/80=3) 
        -- we short (240*240)*(x-400)**2 + (400*400)*(y-240)**2 = (240*240*400*400)
        -- ellipse equation by dividing 80*80 to the simpler form
        elsif (3*3)*((x-400)**2)+(5*5)*((y-240)**2) < (3*3)*(400*400) then
            RGB := GOLDENROD;
        else -- 'if -elsif'  chain must terminate by 'else' option to prevent forbidden latches
            RGB := DARKBLUE;
        end if;
        RGBcolor <=RGB;  -- Inside processes, we always prefer variables. Signals are only connections from/to outside.
    end process;
end architecture;
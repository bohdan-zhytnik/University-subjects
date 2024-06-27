-------------------------------------------------------------
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta], Published under GNU General Public License
-------------------------------------------------------------
-- LCDlogic0 with nimation of its middle chain
library ieee, work; use ieee.std_logic_1164.all; use ieee.numeric_std.all; -- for integer and unsigned types
use work.LCDpackage.all;

entity LCDlogic0anim is
    -- approx. equation: ANIMSPEED = 0.26 * pps (pixels per second), exactly 1792/6875 * pps :-)
    generic(ANIMSPEEDX:integer:=33; --movement of the chain is 128 pixels per second,  
            ANIMSPEEDY: integer:=62); -- vertical speed = 240 pps 
    port(xcolumn  : in  xy_t      := XY_ZERO; -- x-coordinate of pixel (column index)
         yrow     : in  xy_t          := XY_ZERO; -- y-coordinate of pixel (row index)
         XEND_N   : in  std_logic := '0'; -- '0' only when max xcolumn, otherwise '1', f=32.2 kHz = 33 MHz/1024 
         YEND_N   : in  std_logic := '0'; -- '0' only when max yrow, f=61.4 Hz =33 MHz/(1024*525)
         LCD_DE   : in  std_logic := '0'; -- DataEnable control signal of LCD controller
         LCD_DCLK : in  std_logic := '0'; -- LCD data clock, 33 MHz
         RGBcolor : out RGB_t     ); --  color data type RGB_t = std_logic_vector(23 downto 0), defined in LCDpackage
end;

architecture behavioral of LCDlogic0anim is
    -- We append additional colors to the  Windows 16 color palette defined in the LCDpackage.vhd.
    constant DARKBLUE   : RGB_t := ToRGB(0, 0, 139);
    constant GOLDENROD  : RGB_t := X"DAA520";
	constant FPBASE : integer:=2**4; -- base for fixpoint, required power of 2
    constant MODCHAIN:integer:=32; -- a power of 2 required 
    signal   xanim:integer range 0 to 2*MODCHAIN:=0;
    signal   yanim:integer range 0 to LCD_YSCREEN+2*MODCHAIN:=0;
    
begin -- architecture 
    LSPimage : process(xcolumn, yrow, LCD_DE)
        variable RGB    :RGB_t:=BLACK; -- the color of pixel [xcolumn, yrow], initialized to 0.
        variable xmod32 :integer range 0 to 31 :=0; -- reminder after 32 division
        variable meo    :integer range 0 to 15 :=0;
        variable x, y, xorg, ychain : integer:=0; -- integer xcolumn and yrow
    begin -- process
        x := to_integer(xcolumn); y := to_integer(yrow); -- convert unsigned to integers
        xorg:=x+xanim; -- we added animation offset that is cyclic, so we only add mod
        xmod32 := xorg mod MODCHAIN; 
        if (xorg/MODCHAIN) mod 2=1 then meo:=(MODCHAIN-1-xmod32)/2;  -- odd, even line equations 
        else meo:=xmod32/2;
        end if;
        ychain:=y+MODCHAIN; -- translate tranformation to delay chain appearence
        ------------------------------------------------------------
        if LCD_DE = '0' then  RGB := LIGTHGRAY;   --outside of the visible frame
        elsif ychain>=(yanim-meo) and ychain<=(yanim+meo) then RGB := YELLOW; -- the middle chain links
        elsif (3*3)*((x-400)**2)+(5*5)*((y-240)**2) < (3*3)*(400*400) then -- ellipse
             RGB := GOLDENROD;
        else RGB := DARKBLUE;
        end if;
        RGBcolor <=RGB;  
    end process;
 ------------------------------------------------------------------------------ 
 -- Inside processes, we always prefer variables. Signals are only connections from/to outside.  
-- The process counts xamin from 0 to 2*MODCHAIN-1
ianim1: process(YEND_N)
        constant LIMITX:integer := 2*MODCHAIN*FPBASE; -- chain has link length with modulo 64
        variable cntx:integer range 0 to LIMITX-1:=0;
        constant LIMITY:integer := (LCD_YSCREEN+2*MODCHAIN)*FPBASE; -- chain has link length with modulo 64
        variable cnty:integer range 0 to LIMITY-1:=0;
        variable resetx:boolean:=FALSE;
        begin
           if falling_edge(YEND_N) then -- the last row begins
               resetx:=FALSE;
               if cnty<LIMITY-ANIMSPEEDY then cnty:=cnty+ANIMSPEEDY;
               else cnty:=0; resetx:=TRUE;-- we wil start with invisible chain, so no continuity is required
               end if;
               if resetx then cntx:=0;
               elsif cntx<LIMITX-ANIMSPEEDX then cntx:=cntx+ANIMSPEEDX;  
               else cntx:=cntx+ANIMSPEEDX-LIMITX; -- in LCD visible part, we keep continuos movement
               end if;
           end if;
           xanim<=(cntx+FPBASE/2)/FPBASE;
            -- we convert from fixpoint with rounding 
           yanim<=(cnty+FPBASE/2)/FPBASE; 
        end process;
end architecture;
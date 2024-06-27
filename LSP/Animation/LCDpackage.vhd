-- LCDpackage version V2.0 (March 16, 2024] adapted for GHDL simulation
-------------------------------------------------------------
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta]
-- Published under GNU General Public License
-------------------------------------------------------------
-- Package is explained:
--cz: kapitola 7 v  https://dcenet.fel.cvut.cz/edu/fpga/doc/UvodDoVHDL1_concurrent_V20.pdf
--eng: Chapter 7 in https://dcenet.fel.cvut.cz/edu/fpga/doc/CircuitDesignWithVHDL_dataflow_and_structural_eng_V10.pdf 
---------------------------------------------------------------
library ieee; use ieee.std_logic_1164.all; use ieee.numeric_std.all; 

package LCDpackage is

   constant LCD_XSCREEN : integer := 800;  -- the visible columns of LCD screen, the width of screen 
   constant LCD_YSCREEN : integer := 480;  -- the visible rows of LCD screen, the height of screen 
   constant XCOLUMN_MAX : integer :=1023;   -- max. xcolumn lies in invisible part)
   constant YROW_MAX    : integer := 524;  -- max. yrow lies in invisible part)

   -- for subtype, there exists conversion from/to original type and all its subtypes
   subtype  xy_t is unsigned(9 downto 0);  --xcolumn and yrow data sent by VGAgenerator
   constant XY_ZERO : xy_t := (others=>'0'); 

   subtype RGB_t is std_logic_vector(23 downto 0); -- R G B color, R:23..16, G:15..8, B:7..0 

	-- 16 named web colors aka (also known as) 16 Windows colors 	
	constant AQUA      : RGB_t:=X"00FFFF"; --  
    constant BLACK     : RGB_t:=X"000000";
    constant BLUE      : RGB_t:=X"0000FF";
    constant GRAY      : RGB_t:=X"808080";
    constant GREEN     : RGB_t:=X"008000";
    constant FUCHSIA   : RGB_t:=X"FF00FF"; -- aka VIOLET
    constant LIGTHGRAY : RGB_t:=X"C0C0C0"; -- aka SILVER
    constant LIME      : RGB_t:=X"008000";
    constant OLIVE     : RGB_t:=X"808000";
    constant MAROON    : RGB_t:=X"800000";
    constant NAVY      : RGB_t:=X"000080";
    constant PURPLE    : RGB_t:=X"800080";
    constant RED       : RGB_t:=X"FF00FF";
    constant SILVER    : RGB_t:=X"C0C0C0";  -- aka LIGHTGRAY
    constant TEAL      : RGB_t:=X"008080";
    constant VIOLET    : RGB_t:=X"FF00FF";  -- aka FUCHSIA
    constant WHITE     : RGB_t:=X"FFFFFF";
    constant YELLOW    : RGB_t:=X"FFFF00"; 

   subtype color_t is std_logic_vector(7 downto 0); -- color information
	-- Type is unconverteble to others types. Records types are VHDL equivalents of C structures
	type RGB_record is -- RGB colors in components for individual manipulation 
		record 
			R, G, B : color_t;
		end record;

-- Conversion functions -------------------------------------------

 -- it creates the color from integer R G B values 0 to 255
	function ToRGB(r, g, b:natural) return RGB_t;
-- it does nothing, but it exist if a user accidently utilizes it
	function ToRGB(rgb:RGB_t) return RGB_t;

	
 -- it converts positive integer to color byte in range 0 to 255
	function ToByte(n:natural) return color_t; 
-- it joins RGB_record into 24 bit color vector RGB_t
    function ToRGB(rgb:RGB_record) return RGB_t;
 -- it splits the color from 24 bit std_logic_vector to individual components
	function RGB2record(rgb_hex:RGB_t) return RGB_record;

 ------------------------------- color model HSV ---------------------------
 -- !!! We recommend the usage of HsvToRGB only for initialization of constants !!!
 -- The function creates a RGB color defined by HSV palette. 
 -- All its parameters are in range 0 to 255 to keep byte sizes.
 -- H - Hue (H=0:0 degrees, H=255 corresponds to 360 degrees !!!), 
 -- S - Saturation (S=0:0 %,H=255:100 %),
 -- V - Value (V=0:0 %,V=255:100 %),, 
 --- see https://en.wikipedia.org/wiki/HSL_and_HSV
  function HsvToRGB(H,S,V:natural) return RGB_t;

end package LCDpackage;
----------------------------------------------------------------------------------------------------
package body LCDpackage is
	
	function ToRGB(rgb:RGB_record) return RGB_t is
    begin
       return RGB_t'(rgb.R & rgb.G & rgb.B); -- RGB_t' type hint for compiler
    end function;

	 -- it does nothing, but it exist if a user accidently utilizes it
	function ToRGB(rgb:RGB_t) return RGB_t is
   begin
       return rgb; -- RGB_t' type hint for compiler
    end function;

 	function ToByte(n:natural) return color_t is
	begin return std_logic_vector(to_unsigned(n,color_t'LENGTH));
	end;

	function ToRGB(r,g,b:natural) return RGB_t is
	variable tmp:RGB_record;
	begin  tmp.R:=ToByte(r); tmp.G:=ToByte(g); tmp.B:=ToByte(b);
       return ToRGB(tmp);
	end;

	function RGB2record(rgb_hex:RGB_t) return RGB_record is
	variable tmp:RGB_record;
	begin  tmp.R:=rgb_hex(23 downto 16); tmp.G:=rgb_hex(15 downto 8); tmp.B:=rgb_hex(7 downto 0);
       return tmp;
	end;
	
	function limit(x:natural) return integer is
	begin
	   	   if x <= 255 then return x; else return 255; end if; 
	end function;

	function HsvToRGB(H,S,V:natural) return RGB_t is
	variable Hv,Sv,Vv, Rv, Gv, Bv, p,q,t : integer range 0 to 255;
	variable Region : integer range 0 to 5;
	variable Remainder : integer range 0 to 6*42;
	variable rgbRecord:RGB_record;
	
	begin
	  Hv:=limit(H); Sv:=limit(S); Vv:=limit(V);
	  -- range tests replace the division by 43 
	  if Hv<3*43 then 
	     if Hv<43 then Region:=0;
	     elsif Hv<2*43 then Region:=1;
	     else Region:=2; end if;
	  else 
	     if Hv<4*43 then Region:=3;
	     elsif Hv<5*43 then Region:=4;
	     else Region:=5;
		  end if;
	  end if;
	  
	  Remainder:= (Hv-Region*43)*6;
     p := (Vv * (255 - Sv))/256;
     q := (Vv * (255 - ((Sv * Remainder)/256))) / 256;
     t := (Vv * (255 - ((Sv * (255 - Remainder))/256)))/256;
	  
	  if Sv=0 then Rv:=0; Gv:=0; Bv:=0;
	  else
		  case Region is
			 when 0 => Rv:=Vv; Gv:=t;  Bv:=p;
			 when 1 => Rv:=q;  Gv:=Vv; Bv:=p;
			 when 2 => Rv:=p;  Gv:=Vv; Bv:=t;
			 when 3 => Rv:=p;  Gv:=q;  Bv:=Vv;
			 when 4 => Rv:=t;  Gv:=p;  Bv:=Vv;
			 when others => Rv:=Vv; Gv:=p; Bv:=q;
		  end case; 
     end if;
	  rgbRecord.R:=std_logic_vector(to_unsigned(Rv,rgbRecord.R'LENGTH));
	  rgbRecord.G:=std_logic_vector(to_unsigned(Gv,rgbRecord.G'LENGTH));
	  rgbRecord.B:=std_logic_vector(to_unsigned(Bv,rgbRecord.B'LENGTH));
	  return ToRGB(rgbRecord);
end function;	
  
end package body LCDpackage;


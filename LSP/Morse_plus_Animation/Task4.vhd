-------------------------------------------------------------
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta], Published under GNU General Public License
-------------------------------------------------------------
-- LCDlogic0 with nimation of its middle chain


library ieee, work; use ieee.std_logic_1164.all; use ieee.numeric_std.all; -- for integer and unsigned types
use work.LCDpackage.all;

entity LCDlogic0anim is
    generic(ANIMSPEEDX:integer:=33;           -- movement of the chain is 128 pixels per second,  
            ANIMSPEEDY: integer:=62);         -- vertical speed = 240 pps 
    port(SW       : in std_logic_vector(5 downto 0) := "000111";  -- 8 switches input
			xcolumn  : in  xy_t      := XY_ZERO; -- x-coordinate of pixel (column index)
         yrow     : in  xy_t      := XY_ZERO; -- y-coordinate of pixel (row index)
         XEND_N   : in  std_logic := '0';     -- '0' only when max xcolumn, otherwise '1', f=32.2 kHz = 33 MHz/1024 
         YEND_N   : in  std_logic := '0';     -- '0' only when max yrow, f=61.4 Hz =33 MHz/(1024*525)
         LCD_DE   : in  std_logic := '0';     -- DataEnable control signal of LCD controller
         LCD_DCLK : in  std_logic := '0';     -- LCD data clock, 33 MHz
         RGBcolor : out RGB_t     ); --  color data type RGB_t = std_logic_vector(23 downto 0), defined in LCDpackage
end;

architecture behavioral of LCDlogic0anim is

    constant add_to_counter : integer := 75; -- This value is added to the second counter (with LCD_DCLK).
	 constant FPBASE      : integer := 2**10; 
    signal xMorseStart      : integer range 0 to LCD_XSCREEN := LCD_XSCREEN;
    signal xMorseStart_X_FPBASE  : integer range 0 to LCD_XSCREEN*FPBASE := LCD_XSCREEN*FPBASE; 
    signal Morse_expand_FPBASE   : integer range 0 to 4*LCD_XSCREEN*add_to_counter := 0;--value of the second counter 
	 
    constant DARKBLUE    : RGB_t   := ToRGB(0, 0, 139);
    constant GOLDENROD   : RGB_t   := X"DAA520";
	 
    signal addr     : std_logic_vector(13 downto 0) := (others => '0');
    signal q0, q    : std_logic_vector(1 downto 0) := (others => '0');
	 signal addr_ZHYTN     : std_logic_vector(13 downto 0) := (others => '0');--addr for vhd file, where the image of the Morse letters is located
	 signal q0_ZHYTN, q_ZHYTN    : std_logic_vector(0 downto 0) := (others => '0');

function MorseZHYTN(X: std_logic_vector(5 downto 0)) return std_logic is
    variable a, b, c, d, e, f   : std_logic;
    variable Y00, Y01, Y10, Y11 : std_logic;
    variable Y                  : std_logic;
begin
    -- Assign individual bits of X to variables
    a := X(5); b := X(4); c := X(3); d := X(2); e := X(1); f := X(0);

    -- Compute the intermediate variables
    Y00 := (e and not c) or (e and f) or (f and not c) or (not d and f);
    Y01 := (e and c and not d) or (f and not e) or (f and e and not c and not d) or (e and f and c and d);
    Y10 := (not c and not e) or (f and not d) or (e and c and (not d or f));
    Y11 := (not e and not c and not d) or (f and not c and not d);

    -- Select output based on the high bits of X
    case X(5 downto 4) is
        when "00" =>
            Y := Y00;
        when "01" =>
            Y := Y01;
        when "10" =>
            Y := Y10;
        when others =>
            Y := Y11;
    end case;

    return Y;
end function;


begin -- architecture 

	 imem : entity work.imgLSPbmp port map(addr, LCD_DCLK, q0); --read the code of the stars 
	 q <= q0 when rising_edge(LCD_DCLK); -- output register
	 
	 ZHYTN : entity work.ZHYTN_bmp port map(addr_ZHYTN, LCD_DCLK, q0_ZHYTN);--read the code of the "ZHYTN" 
	 q_ZHYTN <= q0_ZHYTN when rising_edge(LCD_DCLK); -- output register	
	 
-- draws all on the screen 
   LSPimage : process(xcolumn, yrow, LCD_DE, LCD_DCLK, SW, YEND_N,q,q_ZHYTN, xMOrseStart )
	constant MW : integer := 127;
	constant MH : integer := 129;
	constant MX1 : integer :=  MW/2 ;
	constant MX2 : integer := LCD_XSCREEN/2 + MW ;
	constant MY1 : integer := MH/4 ;
	constant MY2 : integer := LCD_YSCREEN/2 + MH / 2;
   constant triangle_position : integer := 752; 
	
    type palette_t is array (natural range <>) of RGB_t;        -- unconstrained array suitable for all !! constant !! palletes 
	 constant palette1 : palette_t := (GOLDENROD, GRAY, MAROON); -- default range 0 to 2 for 1. sun image
	 constant palette2 : palette_t := (MAROON, GRAY, GOLDENROD); -- default range 0 to 2 for 2. sun image
	
    variable ixp        : integer range 0 to palette1'HIGH := 0;  -- We cannot reffer to pallete_t here, it is unconstrained 
    variable inImg      : integer range 0 to 2 := 0;
	 
    variable Morse_shift : integer range 0 to 163840 := 1024;--xMorseStart is shifted by this value every frame 
	 variable Morse_shift_Level :integer range 1 to 8 := 1; --number of columns that indicate speed level
	 variable Morse_value : std_logic := '0';	--output of the function MorseZHYTN
    variable triangle_value   : std_logic := '0'; -- this value regulates sums' pallete
    variable Pause   : integer range 0 to 1200 := 2;
	 variable Pause_FOR_DRAW   : integer range 0 to 1200 := 2;-- It is used to draw indicator of pause time 
	 
    variable RGB    :RGB_t:=BLACK;              -- the color of pixel [xcolumn, yrow], initialized to 0.
    variable x, y : integer:=0;   -- integer xcolumn and yrow
    
    begin -- process 
            x := to_integer(xcolumn); y := to_integer(yrow); -- convert unsigned to integers
            inImg := 0;           
				
				case SW (2 downto 0) is
                when "000" => Morse_shift := 64;  
					 					Morse_shift_Level:=1;
                when "001" => Morse_shift := 128; 
					 					Morse_shift_Level:=2;
                when "010" => Morse_shift := 256;  
					 					Morse_shift_Level:=3;
                when "011" => Morse_shift := 512;  
					 					Morse_shift_Level:=4;
                when "100" => Morse_shift := 1024;  
					 					Morse_shift_Level:=5;
                when "101" => Morse_shift := 2048;  
					 					Morse_shift_Level:=6;
                when "110" => Morse_shift := 3072;  
					 					Morse_shift_Level:=7;
                when "111" => Morse_shift := 4096;  
					 					Morse_shift_Level:=8;
                when others      => Morse_shift := 4096;  
            end case;

				 if falling_edge(YEND_N) then 
				 
					  if (triangle_position < xMorseStart ) or (Pause/=0) then triangle_value := '0'; --this 'if' sets triangle_value 
					  else triangle_value := MorseZHYTN(std_logic_vector(to_unsigned((triangle_position - xMorseStart)*add_to_counter/FPBASE, 6)));
					  end if;
					  
					  if (xMorseStart_X_FPBASE - Morse_shift>= 0) or (Pause /=0)  then  --this 'if' shifts xMorseStart and set value of the Pause 
						  if Pause = 0 then
							 xMorseStart_X_FPBASE <= xMorseStart_X_FPBASE - Morse_shift;	
							 xMorseStart <= xMorseStart_X_FPBASE / FPBASE;		
						  else 
								if Pause = 1 then
									xMorseStart_X_FPBASE <= LCD_XSCREEN*FPBASE;
									xMorseStart <= LCD_XSCREEN;
								end if;						 
								Pause := Pause - 1;					 
						  end if;  
							
					  else 	

							case SW (4 downto 3) is
								 when "00" => Pause := 64; Pause_FOR_DRAW := 12; 
								 when "01" => Pause := 128; Pause_FOR_DRAW := 6;
								 when "10" => Pause := 256; Pause_FOR_DRAW := 3;
								 when "11" => Pause := 512;  Pause_FOR_DRAW := 1;
								 when others      => Pause := 512; Pause_FOR_DRAW := 1;
							end case;
							
							--Pause_FOR_DRAW:=800/Pause;
					  end if;     
				 end if;


				if rising_edge(LCD_DCLK) then --this 'if' is used for draw morse code
				
					 if x < xMorseStart then
						Morse_expand_FPBASE <= 0;
					 else Morse_expand_FPBASE <= Morse_expand_FPBASE + add_to_counter;
					 end if;
					 
					 Morse_value := MorseZHYTN(std_logic_vector(to_unsigned(Morse_expand_FPBASE/FPBASE, 6)));
				end if;
	
	if x >= MX1 and x < MX1 + MW and  y >= MY1 and y < MY1 + MH then inImg := 1;   
	elsif x >= MX2 and x < MX2 + MW and  y >= MY2 and y < MY2 + MH then inImg := 2; 
	end if;
	
	if LCD_DE = '0' then  RGB := LIGTHGRAY;   --outside of the visible frame
	elsif Pause /=0 and  x >= 352 and x < 480 and y >= 420+ Pause_FOR_DRAW*Pause/16 and y<470  then  --it draws a rising bar that indicates a pause 
		 RGB := WHITE;
	elsif x>=352 and x < 480 and((y>=470 and y<=480) or (y>=410 and y<420))	 then --bar limit
		RGB := BLACK;	
		
	elsif x>=354 and x<475 and  ((x mod 16 = 1) or (x mod 16 = 0) or (x mod 16 = 15)) and y >= 10 and y<60  then --draws columns of the speed level
		 RGB := DARKBLUE;
	elsif x >= 352 and x < 352+Morse_shift_Level*16 and y >= 10 and y<60  then
		 RGB := WHITE;
	elsif x>=352 and x < 480 and((y>=60 and y<70) or (y>=0 and y<10))	 then
		RGB := BLACK;
		
	elsif inImg > 0 and q /= "11" then 
		 ixp := to_integer(unsigned(q)); 
		 if triangle_value = '1' and SW(5) = '1' then 
			  RGB := palette2(ixp); 
		 else RGB := palette1(ixp);
		end if;
		 
	elsif x>736 and x<=752 and y>= 175 and y < 175 + (x mod 32) then RGB := LIME; --draws the triangle
	elsif x<768 and x>=752 and y>= 175 and y < 175 + (32 - x mod 32) then RGB := LIME;

	elsif y >= -(x-800)*1228/2048	  and  (x-286)**2 + (y+315)**2 <367000 then RGB := RED;
	elsif y <= -(x-800)*1228/2048	  and  ((x-517)**2+(y-795)**2)<367000 then RGB := GOLDENROD;
   else RGB := DARKBLUE; 
	end if;
	
	
	if  y > 195 and y < 205 and Morse_value = '1' then --draws morse code
	 RGB := not RGB; end if;
	 
	if q_ZHYTN /= "1"  and x>=xMOrseStart and y>=210 and y<234 then  --draws the letters
		 RGB := not RGB; end if;
	
	case inImg is
    when 0 => addr <= (others => '0');	
    when 1 => addr <= std_logic_vector(to_unsigned((y - MY1) * MW + ( MW+MX1-1-x), addr'LENGTH));  	
	 when 2 => addr <= std_logic_vector(to_unsigned((y - MY2) * MW + x - MX2, addr'LENGTH));
  end case;
  
  if x >= xMOrseStart and x - xMOrseStart<=662 and y>=210 and y<242 then 
		addr_ZHYTN <= std_logic_vector(to_unsigned((y - 210) * 662 + x-xMOrseStart, addr_ZHYTN'LENGTH));
  else 
		addr_ZHYTN <= std_logic_vector(to_unsigned(0, addr_ZHYTN'LENGTH));
  end if;
  
  RGBcolor <= RGB;    
    end process;
 ------------------------------------------------------------------------------ 
end architecture;
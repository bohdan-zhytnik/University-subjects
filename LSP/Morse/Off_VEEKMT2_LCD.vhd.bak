-- Eng: If you do not use the rear LCD display, turn it off to reduce the heating of the board.
-- Cz: Pokud nepouzivame zadni LCD display, vypneme ho, cimz redukujeme ohrivani desky.
------------------------------------------------------------------------------------------
library ieee; use ieee.std_logic_1164.all; use ieee.numeric_std.all;
entity Off_VEEKMT2_LCD is
 port( LCD_DIM : out std_logic; -- backlight on
	    LCD_POWER_CTL: out std_logic); --power on/off
end entity;

architecture beh of Off_VEEKMT2_LCD is
begin -- architecture
	LCD_POWER_CTL<='0'; -- set LCD power off
   LCD_DIM <= '0';     -- set LCD backlight off
end architecture;


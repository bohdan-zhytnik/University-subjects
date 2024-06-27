-------------------------------------------------------------
-- Module Buzzer750Hz generates sound with a frequency of 750 Hz
-- ACLRN = 1 SoundON=1 enable sound; otherwise sound is disabled
-- ACLRN = 0 SoundOn=X  initialize
-- Other pins correspond to VEEK-MT2 assignments 
-- Its pins can be created in Block Diagram editor by right-mouse on Buzzer750Hz 
-- and selection "Generates Pins for Symbol Ports"
---------------------------------------------------------------------------------------------------
-- The Buzzer utilizes Phase-locked Loop so its CLOCK_50 input must be connected directly to pin CLOCK_50
--------------------------------------------------------------------------------------------------
-- Library for Logic Systems and Processors, CTU-FFE Prague, Dept. of Control Eng., Richard Susta
-- Published under GNU General Public License
--------------------------------------------------------------------------------------------
library ieee; use ieee.std_logic_1164.all; 

entity Buzzer750Hz is 
	port(	SoundON, CLOCK_50, ACLRN, AUD_ADCDAT : in  std_logic;
		I2C_SDAT :  inout  std_logic;
		AUD_ADCLRCK, AUD_DACLRCK, AUD_DACDAT, AUD_XCK, AUD_BCLK, I2C_SCLK :  out  std_logic);
end entity;

ARCHITECTURE structural OF Buzzer750Hz IS 

COMPONENT Buzzer IS 
	PORT
	(
		SoundON :  IN  STD_LOGIC;
		CLOCK_50 :  IN  STD_LOGIC; 
		ACLRN :  IN  STD_LOGIC;    -- connect to KEY[0] pin
		AUD_ADCDAT :  IN  STD_LOGIC; 
		I2C_SDAT :  INOUT  STD_LOGIC;
		Divider1500Hz :  IN  STD_LOGIC_VECTOR(3 DOWNTO 0);
		AUD_ADCLRCK :  OUT  STD_LOGIC;
		AUD_DACLRCK :  OUT  STD_LOGIC;
		AUD_DACDAT :  OUT  STD_LOGIC;
		AUD_XCK :  OUT  STD_LOGIC;
		AUD_BCLK :  OUT  STD_LOGIC;
		I2C_SCLK :  OUT  STD_LOGIC
	);
END COMPONENT;


begin

inst_Buzzer : Buzzer
PORT MAP(SoundON, 	CLOCK_50, ACLRN, AUD_ADCDAT,I2C_SDAT, "0001",
			AUD_ADCLRCK,AUD_DACLRCK,AUD_DACDAT,AUD_XCK,AUD_BCLK,I2C_SCLK);

END ARCHITECTURE;
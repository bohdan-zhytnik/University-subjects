-- The Buzzer generates sound with selectable frequency
-- ACLRN = 1 SoundON=1 enable sound, otherwise sound is disabled
-- ACLRN = 0 SoundOn=X  initilize
-- Divider1500Hz - divider of base frequency 1500Hz - output sound is 1500/(Divider1500Hz_input_value+1)
-- /15=94Hz, /14=100 Hz, /13=107Hz... /6=214Hz, /5=250Hz,  /4=300Hz, /3=375Hz, /2=500Hz, /1=750Hz, /0=1500 Hz
-- Its pins can be created in Block Diagram editor by right-mouse on Buzzer750Hz 
-- and selection "Generates Pins for Symbol Ports"
---------------------------------------------------------------------------------------------------
-- !!!!! The Buzzer utilizes Phase-locked Loop so its CLOCK_50 input must be connected directly to pin CLOCK_50
--------------------------------------------------------------------------------------------------
-- Library for Logic Systems and Processors, CTU-FFE Prague, Dept. of Control Eng., Richard Susta
-- Published under GNU General Public License
-----------------------------------------------------------------------------------------------------------------
-- The All-In-One structure of Buzzer.vhd is explained in chapter 7.2 of Richard Susta's textbooks:
-- ENG: Circuit Design With VHDL dataflow and structural 
-- CZ: Uvod do navrhu obvodu v jazyce VHDL I.
-- available at https://dcenet.fel.cvut.cz/edu/fpga/guides.aspx 
-- Switch the page to CZ for its Czech version.
----------------------------------------------------------------------------------------------------------------------

LIBRARY ieee; USE ieee.std_logic_1164.all; 

LIBRARY work;

ENTITY Buzzer IS 
	PORT
	(	SoundON :  IN  STD_LOGIC;
		CLOCK_50 :  IN  STD_LOGIC;
		ACLRN :  IN  STD_LOGIC;      -- connect to KEY[0] pin
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
END Buzzer;

ARCHITECTURE bdf_type OF Buzzer IS 

COMPONENT Buzzer_AdcDac
	PORT(ACLRN : IN STD_LOGIC;
		 selectAdcData : IN STD_LOGIC;
		 audioClock : IN STD_LOGIC;
		 adcData : IN STD_LOGIC;
		 leftDataIn : IN STD_LOGIC_VECTOR(3 DOWNTO 0);
		 rightDataIn : IN STD_LOGIC_VECTOR(3 DOWNTO 0);
		 bitClock : OUT STD_LOGIC;
		 adcLRSelect : OUT STD_LOGIC;
		 dacLRSelect : OUT STD_LOGIC;
		 dacData : OUT STD_LOGIC;
		 dataInClock : OUT STD_LOGIC
	);
END COMPONENT;

COMPONENT Buzzer_AudioCodec
	PORT(AClrn : IN STD_LOGIC;
		 audioClock : IN STD_LOGIC;
		 sda : INOUT STD_LOGIC;
		 scl : OUT STD_LOGIC;
		 stateOut : OUT STD_LOGIC_VECTOR(2 DOWNTO 0)
	);
END COMPONENT;

COMPONENT Buzzer_ClockGen
	PORT(ACLRN : IN STD_LOGIC;
		 Clock_50 : IN STD_LOGIC;
		 AudioClock : OUT STD_LOGIC;
		 Delayed_Clrn : OUT STD_LOGIC
	);
END COMPONENT;

COMPONENT Buzzer_SinGen
	PORT(CLK : IN STD_LOGIC;
		 ACLRN : IN STD_LOGIC;
		 FREQ : IN STD_LOGIC_VECTOR(3 DOWNTO 0);
		 Q : OUT STD_LOGIC_VECTOR(3 DOWNTO 0)
	);
END COMPONENT;

SIGNAL	aclrn_signal :  STD_LOGIC;
SIGNAL	delayedAclrn :  STD_LOGIC;
SIGNAL	f18_432MHz :  STD_LOGIC;
SIGNAL	f48kHz :  STD_LOGIC;
SIGNAL	sinGenOutput :  STD_LOGIC_VECTOR(3 DOWNTO 0);
SIGNAL	soundOnNeg, soundOnNeg0 :  STD_LOGIC;


BEGIN 

process(f18_432MHz)
begin  -- Synchronizer for crossing time domain
  if rising_edge(f18_432MHz) then
      soundOnNeg <= soundOnNeg0;
      soundOnNeg0 <= not SoundON;
  end if;
end process;

b2v_inst_adcDacController : Buzzer_AdcDac
PORT MAP(ACLRN => delayedAclrn,
		 selectAdcData => soundOnNeg,
		 audioClock => f18_432MHz,
		 adcData => AUD_ADCDAT,
		 leftDataIn => sinGenOutput,
		 rightDataIn => sinGenOutput,
		 bitClock => AUD_BCLK,
		 adcLRSelect => AUD_ADCLRCK,
		 dacLRSelect => AUD_DACLRCK,
		 dacData => AUD_DACDAT,
		 dataInClock => f48kHz);


b2v_inst_audioCodecController : Buzzer_AudioCodec
PORT MAP(AClrn => aclrn_signal,
		 audioClock => f18_432MHz,
		 sda => I2C_SDAT,
		 scl => I2C_SCLK);


b2v_inst_clockGenrator : Buzzer_ClockGen
PORT MAP(ACLRN => aclrn_signal,
		 Clock_50 => CLOCK_50,
		 AudioClock => f18_432MHz,
		 Delayed_Clrn => delayedAclrn);


b2v_inst_sinGen : Buzzer_SinGen
PORT MAP(CLK => f48kHz,
		 ACLRN => aclrn_signal,
		 FREQ => Divider1500Hz,
		 Q => sinGenOutput);

aclrn_signal <= ACLRN;
AUD_XCK <= f18_432MHz;

END bdf_type;

------------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity Buzzer_AudioCodec is port(
	AClrn : in std_logic;
	audioClock : in std_logic;
	scl : out std_logic;
	sda : inout std_logic;
	stateOut : out integer range 0 to 7);
end Buzzer_AudioCodec;

architecture behavioral of Buzzer_AudioCodec is
	
	subtype	i2cDataType  is std_logic_vector(15 downto 0);
	constant LENGTH_OF_CODEC : integer := 11;
	type codecArray is array(0 to LENGTH_OF_CODEC) of i2cDataType;
 
    signal codec : codecArray := 
    ( (others=>'0'), -- 0 dummy data
	  (X"001F"),     -- 1 Left input volume is maximum
	  (X"021F"),     -- 2 Right input volume is maximum
	  (X"0479"),     -- 3 Left output volume is high
	  (X"0679"),      --4 -- Right output volume is high
	  (X"0810"),      --5, -- No sidetone, DAC: on, disable mic, line input to ADC: on
	  (X"0A06"),      --6, -- deemphasis to 48 KHz, enable high pass filter on ADC
	  (X"0C00"),      --7, -- no power down mode
	  (X"0E01"),      --8, -- MSB first, left-justified, slave mode
	  (X"1002"),      --9, -- 384 fs oversampling
	  (X"1201"),      --10, -- activate					  
	  (others=>'0')   --11 -- it should never occur
	); -- should never occur,

	signal muxSelect : integer range 0 to LENGTH_OF_CODEC+1;
	signal i2cData : i2cDataType := X"0000";
	signal i2cClock20KHz : std_logic := '0';
	constant DIV18m4to20khz : integer := 921; -- 18.421 Mhz to 20 kHz	
	signal i2cClockCounter : integer range 0 to DIV18m4to20khz := 0;
	
	signal i2cControllerData : std_logic_vector(23 downto 0);
	signal i2cRun,done,ack : std_logic;	
	
	signal incrementMuxSelect : std_logic := '0';
		
	type states is (resetState,transmit,checkAcknowledge,turnOffi2cControl,incrementMuxSelectBits,stop);
	signal currentState,nextState : states;	

COMPONENT I2C_Controller
	PORT
	(
		CLOCK		:	 IN STD_LOGIC;
		I2C_SCLK		:	 OUT STD_LOGIC;
		I2C_SDAT		:	 INOUT STD_LOGIC;
		I2C_DATA		:	 IN STD_LOGIC_VECTOR(23 DOWNTO 0);
		reset		:	 IN STD_LOGIC;
		start		:	 IN STD_LOGIC;
		done		:	 OUT STD_LOGIC;
		readWriteEnable		:	 IN STD_LOGIC;
		ACK		:	 OUT STD_LOGIC;
		SD_COUNTER		:	 OUT STD_LOGIC_VECTOR(5 DOWNTO 0);
		SDO		:	 OUT STD_LOGIC
	);
END COMPONENT;

begin
    
		-- 20 KHz i2c clock 
		process(audioClock,AClrn)
		begin
			if AClrn = '0' then
				i2cClockCounter <= 0;
				i2cClock20KHz <= '0';
			else
				if audioClock'event and audioClock = '1' then
						if i2cClockCounter < DIV18m4to20khz/2 then -- 1249 then
							i2cClock20KHz <= '0';
							i2cClockCounter <= i2cClockCounter + 1;
						elsif i2cClockCounter < DIV18m4to20khz then --2499 then
							i2cClock20KHz <= '1';
							i2cClockCounter <= i2cClockCounter + 1;
						else
							i2cClockCounter <= 0;
							i2cClock20KHz <= '0';
						end if;
				end if;
			end if;
		end process;
		
		-- mini FSM to send out right data to audio codec via i2c
		process(i2cClock20KHz)
		begin
			if i2cClock20KHz'event and i2cClock20Khz = '1' then
				currentState <= nextState;
			end if;
		end process;
				
		process(currentState,AClrn,muxSelect,done,ack)
		begin
			case currentState is
				when resetState =>										
					if AClrn = '0' then
						nextState <= resetState;
					else
						nextState <= transmit;
					end if;
					incrementMuxSelect <= '0';
					i2cRun <= '0';
					 
				when transmit =>
					if muxSelect >= LENGTH_OF_CODEC then					
						i2cRun <= '0';
						nextState <= stop;	
					else
						i2cRun <= '1';
						nextState <= checkAcknowledge;
					end if;		
					incrementMuxSelect <= '0';
					 
				when checkAcknowledge =>					
					if done = '1' then
						if ack = '0' then -- all the ACKs from codec better be low
							i2cRun <= '0';
							nextState <= turnOffi2cControl;
						else
							i2cRun <= '0';
							nextState <= transmit;
						end if;
					else					
						nextState <= checkAcknowledge;
					end if;					
					i2cRun <= '1';
					incrementMuxSelect <= '0';
					
				when turnOffi2cControl =>
					incrementMuxSelect <= '0';
					nextState <= incrementMuxSelectBits; 
					i2cRun <= '0';
 
				when incrementMuxSelectBits =>
					incrementMuxSelect <= '1';
					nextState <= transmit; 
					i2cRun <= '0';
 
				when stop =>
					nextState <= stop; -- don't need an others clause since all states have been accounted for
					i2cRun <= '0';
					incrementMuxSelect <= '0';					
 
			end case;
		end process;
		
		process(incrementMuxSelect,AClrn)
		begin
			if AClrn = '0' then
				muxSelect <= 0;
			else
				if incrementMuxSelect'event and incrementMuxSelect='1' then
					muxSelect <= muxSelect + 1;
				end if;				
			end if;
		end process;
		
		-- data to be sent to audio code obtained via a MUX
		-- the select bits for the MUX are obtained by the mini FSM above
		-- the 16-bit value for each setting can be found
		-- in table 29 and 30 on pp. 46-50 of the audio codec datasheet (on the DE1 system CD)

		i2cData <= codec(muxSelect);
					
		-- 0x34 is the base address of your device
		-- Refer to page 43 of the audio codec datasheet and the schematic
		-- on p. 38 of DE1 User's manual.  CSB is tied to ground so the 8-bit base address is
		-- b00110100 = 0x34.  		
		i2cControllerData <= X"34"&i2cData; 		
		
		-- instantiate i2c controller
		i2cController : i2c_controller port map (i2cClock20KHz,scl,sda,i2cControllerData,AClrn,i2cRun,done,'0',ack);
		
		-- User I/O
		with currentState select
			stateOut <= 0 when resetState,
						   1 when transmit,
							2 when checkAcknowledge,
							3 when turnOffi2cControl,
							4 when incrementMuxSelectBits,
							5 when stop;							
		
end behavioral;

----------------------------------------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity Buzzer_AdcDac is port (
		ACLRN : in std_logic;
		selectAdcData : in std_logic; -- connected to SW(9), default (down) means sine on lineout.  up is ADC-DAC loopback
		audioClock : in std_logic; -- 18.432 MHz sample clock
		bitClock : out std_logic;
		adcLRSelect : out std_logic;
		dacLRSelect : out std_logic;
		adcData : in std_logic;
		dacData : out std_logic;
		leftDataIn:in std_logic_vector(3 downto 0);
		rightDataIn:in std_logic_vector(3 downto 0);
		dataInClock:out std_logic);
		
end entity;

architecture behavioral of Buzzer_AdcDac is
	signal internalBitClock : std_logic := '0';
	signal bitClockCounter : integer range 0 to 255;
	
	signal internalLRSelect : std_logic := '0';
	signal LRCounter : integer range 0 to 31; 
	signal leftOutCounter : integer range 0 to 15 :=15;
	signal rightOutCounter : integer range 0 to 15 :=15;
	
	-- ADC,DAC data registers
	signal adcDataLeftChannelRegister, adcDataRightChannelRegister: std_logic_vector(15 downto 0);
	signal dacDataLeftChannelRegister ,dacDataRightChannelRegister : std_logic_vector(15 downto 0);
			
	
	
begin
	-- generate bit clock
	-- we have an 18.432 MHz reference clock (refer to audio codec datasheet, this is the required frequency)
	-- we need to shift out 16 bits, 2 channels at 48 KHz.  Hence, the count value for flipping the clock bit is
	-- count = 18.432e6/(48000*16*2) - 1 = 11 (approx)
	
	process(audioClock,ACLRN)
	begin
		if ACLRN = '0' then
			bitClockCounter <= 0;
			internalBitClock <= '0';
		else
			if audioClock'event and audioClock = '1' then
				if bitClockCounter < 5 then					
					internalBitClock <= '0'; 
					bitClockCounter <= bitClockCounter + 1;
				elsif bitClockCounter >= 5 and bitClockCounter < 11 then
					internalBitClock <= '1'; 
					bitClockCounter <= bitClockCounter + 1;
				else
					internalBitClock <= '0'; 
					bitClockCounter <= 0;
				end if;
			end if;
		end if;
	end process;
	bitClock <= internalBitClock;
	

	
	-- generate LeftRight select signals 
	-- flip every 16 bits, starting on NEGATIVE edge
	process(internalBitClock,ACLRN)
	begin
		if ACLRN = '0' then					
			dacDataLeftChannelRegister <= X"0000";
			dacDataRightChannelRegister <= X"0000";
			LRCounter <= 0;
			internalLRSelect <= '0'; -- should start at low, fig. 26 on p. 33 of audio codec datasheet
			leftOutCounter <= 15;
			rightOutCounter <= 15;
		else
			if internalBitClock'event and internalBitClock = '0' then -- flip on negative edge								
				if LRCounter < 16 then	
					internalLRSelect <= '1';
					LRCounter <= LRCounter + 1;
					leftOutCounter <= leftOutCounter - 1;
					rightOutCounter <= 15;
			--		dataCount <= '0';
			        dataInClock<='0';					
				elsif LRCounter >= 16 and LRCounter < 32 then
					internalLRSelect <= '0';
					LRCounter <= LRCounter + 1;
			--		dataCount <= '0';
					leftOutCounter <= 15;
					rightOutCounter <= rightOutCounter - 1;
			        dataInClock<='1';					
					if LRCounter = 31 then
						LRCounter <= 0;
				        if leftDataIn(3)='1' then dacDataLeftChannelRegister <= (leftDataIn & X"FFF"); 
					       else dacDataLeftChannelRegister <=  (leftDataIn & X"000");
					    end if;
				        if rightDataIn(3)='1' then dacDataRightChannelRegister <= (rightDataIn & X"FFF"); 
					       else dacDataRightChannelRegister <=  (rightDataIn & X"000");
					    end if;
					end if;									
				end if;
			end if;
		end if;
	end process;

	adcLRSelect <= internalLRSelect;
	dacLRSelect <= internalLRSelect;
	
	-- sample adc data
	process(internalBitClock,ACLRN,internalLRSelect)
	begin
		if ACLRN = '0' then
			adcDataLeftChannelRegister <= X"0000";
			adcDataRightChannelRegister <= X"0000";
		else
			if internalBitClock'event and internalBitClock = '1' then
				if internalLRSelect = '1' then
					adcDataLeftChannelRegister(15 downto 0) <= adcDataLeftChannelRegister(14 downto 0) & adcData;
				else
					adcDataRightChannelRegister(15 downto 0) <= adcDataRightChannelRegister(14 downto 0) & adcData;
				end if;
			end if;
		end if;
	end process;
	
	
	-- dac data output
	process(internalBitClock,ACLRN,internalLRSelect)
	begin
		if ACLRN = '0' then
			 dacData <= '0';			 
		else
			-- start on falling edge of bit clock
			if internalBitClock'event and internalBitClock = '0' then 
				if internalLRSelect = '1' then		
					if selectAdcData = '1' then
						-- remember, you need to send MSb first.  So, we start at bit 15
						dacData <= adcDataLeftChannelRegister(leftOutCounter);
					else
						dacData <= dacDataRightChannelRegister(leftOutCounter);															
					end if;
				else
					if selectAdcData = '1' then
						dacData <= adcDataLeftChannelRegister(rightOutCounter);
					else
						dacData <= dacDataRightChannelRegister(rightOutCounter);														
					end if;
				end if;
			end if;
		end if;
	 end process;
	 
end behavioral;

-----------------------------------------------------------------------------------------------
-- DE1 ADC DAC interface
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity Buzzer_ClockGen is port (
		ACLRN : in std_logic;  -- KEY[0]
		Clock_50 : in std_logic;  -- 50 MHz
		AudioClock : out std_logic;     -- 18.432 MHz
		Delayed_Clrn : out std_logic);
end entity;

architecture clockGeneratorInside of Buzzer_ClockGen is
   constant  WAITCOUNT : integer := 16#C0000#; --42.67 ms, we keep simple comparission
	-- olriginal value 16#200000# for CLOCK_50; -- 41.94 ms - it has simple comparission
	signal clockGeneratorInternalCount : integer range 0 to WAITCOUNT; -- 32-bit counter
   signal locked, c0:std_logic:='0';

	component  pllAudioClock is
	port(areset		: IN std_logic  := '0';
		  inclk0		: IN std_logic  := '0';
		  c0		: OUT std_logic ;
		  locked		: OUT std_logic);
   end component;

	begin
	audioPllClockGen : pllAudioClock 
	   port map (areset=>not AClrn, inclk0=>Clock_50, c0=>c0, locked=>locked);
	AudioClock<=c0;
	
	process(c0)
	begin
		if (AClrn and locked)='0' then 
		   clockGeneratorInternalCount <= 0; Delayed_Clrn <= '0'; -- reset ADC is active low
		elsif rising_edge(c0) then
		   if clockGeneratorInternalCount < WAITCOUNT then
					Delayed_Clrn <= '0';	clockGeneratorInternalCount <= clockGeneratorInternalCount + 1;
			else
					Delayed_Clrn <= '1';
			end if;
		end if;
	end process;
	
end architecture clockGeneratorInside;
	

-----------------------------------------------------------------------------------------------
-- Library for A0B35SPS - Structures of Computers System
-- CTU-FFE Prague, Dept. of Control Eng. [Richard Susta]
-- Published under GNU General Public License

LIBRARY ieee;
USE ieee.std_logic_1164.all;
USE ieee.numeric_std.all;

LIBRARY work;

ENTITY Buzzer_SinGen IS 
	PORT
	(   CLK :  IN  STD_LOGIC;
		ACLRN :  IN  STD_LOGIC;
		FREQ: IN STD_LOGIC_VECTOR(3 DOWNTO 0);
		Q :  OUT  STD_LOGIC_VECTOR(3 DOWNTO 0)
	);
END Buzzer_SinGen;

ARCHITECTURE rtl OF Buzzer_SinGen IS 

signal  q8 : std_logic_vector(7 downto 0);
    signal waveCounter : unsigned(4 downto 0);
   signal waveIX : integer range 0 to 15;
   signal sineWave0 : integer range -7 to 7;
   signal clk2 : std_logic;

BEGIN

process(clk, FREQ, ACLRN)

variable   cnt : integer range 0 to 31;
variable   f : integer range 0 to 31;
BEGIN 
		if ACLRN = '0' then
				-- Reset the counter to 0
				cnt := 0;
	    else
             if (rising_edge(CLK)) then
                    f:=to_integer(unsigned(FREQ))+1;             
			        if cnt>=f then cnt:=0; clk2<='1';
                    else cnt:=cnt+1; clk2<='0';
			        end if;
			 end if;
		end if;

END process;

	-- square wave address generator
	process(CLK2, ACLRN)
  	begin
		if ACLRN = '0' then
			waveCounter <= (others=>'0');
		else
			if CLK2'event and CLK2 = '1' then
				waveCounter <= waveCounter + 1;
		    end if;
		end if;
        waveIX <= to_integer(waveCounter(3 downto 0));
	end process;

with waveIX select  
  sineWave0  <= 0 when 0,
				1 when 1,
				3 when 2,
				4 when 3,
				5 when 4,
				6 when 5,
				7 when 6,
				7 when 7,
				7 when 8,
				7 when 9,
				7 when 10,
				6 when 11,
				5 when 12,
				4 when 13,
				3 when 14,
				1 when 15;
Q <= std_logic_vector(to_signed(sineWave0,4)) when wavecounter(4)='0' 
     else std_logic_vector(to_signed(-sineWave0,4)); 

END rtl;
 
-------------------------------------------------------------------------------------

-- megafunction wizard: %ALTPLL%
-- GENERATION: STANDARD
-- VERSION: WM1.0
-- MODULE: altpll 

-- ============================================================
-- File Name: pllAudioClock.vhd
-- Megafunction Name(s):
-- 			altpll
--
-- Simulation Library Files(s):
-- 			altera_mf
-- ============================================================
-- ************************************************************
-- THIS IS A WIZARD-GENERATED FILE. DO NOT EDIT THIS FILE!
--
-- 13.0.1 Build 232 06/12/2013 SP 1 SJ Web Edition
-- ************************************************************


--Copyright (C) 1991-2013 Altera Corporation
--Your use of Altera Corporation's design tools, logic functions 
--and other software and tools, and its AMPP partner logic 
--functions, and any output files from any of the foregoing 
--(including device programming or simulation files), and any 
--associated documentation or information are expressly subject 
--to the terms and conditions of the Altera Program License 
--Subscription Agreement, Altera MegaCore Function License 
--Agreement, or other applicable license agreement, including, 
--without limitation, that your use is for the sole purpose of 
--programming logic devices manufactured by Altera and sold by 
--Altera or its authorized distributors.  Please refer to the 
--applicable agreement for further details.


LIBRARY ieee;
USE ieee.std_logic_1164.all;

LIBRARY altera_mf;
USE altera_mf.all;

ENTITY pllAudioClock IS
	PORT
	(
		areset		: IN STD_LOGIC  := '0';
		inclk0		: IN STD_LOGIC  := '0';
		c0		: OUT STD_LOGIC ;
		locked		: OUT STD_LOGIC 
	);
END pllAudioClock;


ARCHITECTURE SYN OF pllaudioclock IS

	SIGNAL sub_wire0	: STD_LOGIC ;
	SIGNAL sub_wire1	: STD_LOGIC_VECTOR (4 DOWNTO 0);
	SIGNAL sub_wire2	: STD_LOGIC ;
	SIGNAL sub_wire3	: STD_LOGIC ;
	SIGNAL sub_wire4	: STD_LOGIC_VECTOR (1 DOWNTO 0);
	SIGNAL sub_wire5_bv	: BIT_VECTOR (0 DOWNTO 0);
	SIGNAL sub_wire5	: STD_LOGIC_VECTOR (0 DOWNTO 0);



	COMPONENT altpll
	GENERIC (
		bandwidth_type		: STRING;
		clk0_divide_by		: NATURAL;
		clk0_duty_cycle		: NATURAL;
		clk0_multiply_by		: NATURAL;
		clk0_phase_shift		: STRING;
		compensate_clock		: STRING;
		inclk0_input_frequency		: NATURAL;
		intended_device_family		: STRING;
		lpm_hint		: STRING;
		lpm_type		: STRING;
		operation_mode		: STRING;
		pll_type		: STRING;
		port_activeclock		: STRING;
		port_areset		: STRING;
		port_clkbad0		: STRING;
		port_clkbad1		: STRING;
		port_clkloss		: STRING;
		port_clkswitch		: STRING;
		port_configupdate		: STRING;
		port_fbin		: STRING;
		port_inclk0		: STRING;
		port_inclk1		: STRING;
		port_locked		: STRING;
		port_pfdena		: STRING;
		port_phasecounterselect		: STRING;
		port_phasedone		: STRING;
		port_phasestep		: STRING;
		port_phaseupdown		: STRING;
		port_pllena		: STRING;
		port_scanaclr		: STRING;
		port_scanclk		: STRING;
		port_scanclkena		: STRING;
		port_scandata		: STRING;
		port_scandataout		: STRING;
		port_scandone		: STRING;
		port_scanread		: STRING;
		port_scanwrite		: STRING;
		port_clk0		: STRING;
		port_clk1		: STRING;
		port_clk2		: STRING;
		port_clk3		: STRING;
		port_clk4		: STRING;
		port_clk5		: STRING;
		port_clkena0		: STRING;
		port_clkena1		: STRING;
		port_clkena2		: STRING;
		port_clkena3		: STRING;
		port_clkena4		: STRING;
		port_clkena5		: STRING;
		port_extclk0		: STRING;
		port_extclk1		: STRING;
		port_extclk2		: STRING;
		port_extclk3		: STRING;
		self_reset_on_loss_lock		: STRING;
		width_clock		: NATURAL
	);
	PORT (
			areset	: IN STD_LOGIC ;
			clk	: OUT STD_LOGIC_VECTOR (4 DOWNTO 0);
			inclk	: IN STD_LOGIC_VECTOR (1 DOWNTO 0);
			locked	: OUT STD_LOGIC 
	);
	END COMPONENT;

BEGIN
	sub_wire5_bv(0 DOWNTO 0) <= "0";
	sub_wire5    <= To_stdlogicvector(sub_wire5_bv);
	locked    <= sub_wire0;
	sub_wire2    <= sub_wire1(0);
	c0    <= sub_wire2;
	sub_wire3    <= inclk0;
	sub_wire4    <= sub_wire5(0 DOWNTO 0) & sub_wire3;

	altpll_component : altpll
	GENERIC MAP (
		bandwidth_type => "AUTO",
		clk0_divide_by => 3125,
		clk0_duty_cycle => 50,
		clk0_multiply_by => 1152,
		clk0_phase_shift => "0",
		compensate_clock => "CLK0",
		inclk0_input_frequency => 20000,
		intended_device_family => "Cyclone IV E",
		lpm_hint => "CBX_MODULE_PREFIX=pllAudioClock",
		lpm_type => "altpll",
		operation_mode => "NORMAL",
		pll_type => "AUTO",
		port_activeclock => "PORT_UNUSED",
		port_areset => "PORT_USED",
		port_clkbad0 => "PORT_UNUSED",
		port_clkbad1 => "PORT_UNUSED",
		port_clkloss => "PORT_UNUSED",
		port_clkswitch => "PORT_UNUSED",
		port_configupdate => "PORT_UNUSED",
		port_fbin => "PORT_UNUSED",
		port_inclk0 => "PORT_USED",
		port_inclk1 => "PORT_UNUSED",
		port_locked => "PORT_USED",
		port_pfdena => "PORT_UNUSED",
		port_phasecounterselect => "PORT_UNUSED",
		port_phasedone => "PORT_UNUSED",
		port_phasestep => "PORT_UNUSED",
		port_phaseupdown => "PORT_UNUSED",
		port_pllena => "PORT_UNUSED",
		port_scanaclr => "PORT_UNUSED",
		port_scanclk => "PORT_UNUSED",
		port_scanclkena => "PORT_UNUSED",
		port_scandata => "PORT_UNUSED",
		port_scandataout => "PORT_UNUSED",
		port_scandone => "PORT_UNUSED",
		port_scanread => "PORT_UNUSED",
		port_scanwrite => "PORT_UNUSED",
		port_clk0 => "PORT_USED",
		port_clk1 => "PORT_UNUSED",
		port_clk2 => "PORT_UNUSED",
		port_clk3 => "PORT_UNUSED",
		port_clk4 => "PORT_UNUSED",
		port_clk5 => "PORT_UNUSED",
		port_clkena0 => "PORT_UNUSED",
		port_clkena1 => "PORT_UNUSED",
		port_clkena2 => "PORT_UNUSED",
		port_clkena3 => "PORT_UNUSED",
		port_clkena4 => "PORT_UNUSED",
		port_clkena5 => "PORT_UNUSED",
		port_extclk0 => "PORT_UNUSED",
		port_extclk1 => "PORT_UNUSED",
		port_extclk2 => "PORT_UNUSED",
		port_extclk3 => "PORT_UNUSED",
		self_reset_on_loss_lock => "OFF",
		width_clock => 5
	)
	PORT MAP (
		areset => areset,
		inclk => sub_wire4,
		locked => sub_wire0,
		clk => sub_wire1
	);



END SYN;

-- ============================================================
-- CNX file retrieval info
-- ============================================================
-- Retrieval info: PRIVATE: ACTIVECLK_CHECK STRING "0"
-- Retrieval info: PRIVATE: BANDWIDTH STRING "1.000"
-- Retrieval info: PRIVATE: BANDWIDTH_FEATURE_ENABLED STRING "1"
-- Retrieval info: PRIVATE: BANDWIDTH_FREQ_UNIT STRING "MHz"
-- Retrieval info: PRIVATE: BANDWIDTH_PRESET STRING "Low"
-- Retrieval info: PRIVATE: BANDWIDTH_USE_AUTO STRING "1"
-- Retrieval info: PRIVATE: BANDWIDTH_USE_PRESET STRING "0"
-- Retrieval info: PRIVATE: CLKBAD_SWITCHOVER_CHECK STRING "0"
-- Retrieval info: PRIVATE: CLKLOSS_CHECK STRING "0"
-- Retrieval info: PRIVATE: CLKSWITCH_CHECK STRING "0"
-- Retrieval info: PRIVATE: CNX_NO_COMPENSATE_RADIO STRING "0"
-- Retrieval info: PRIVATE: CREATE_CLKBAD_CHECK STRING "0"
-- Retrieval info: PRIVATE: CREATE_INCLK1_CHECK STRING "0"
-- Retrieval info: PRIVATE: CUR_DEDICATED_CLK STRING "c0"
-- Retrieval info: PRIVATE: CUR_FBIN_CLK STRING "c0"
-- Retrieval info: PRIVATE: DEVICE_SPEED_GRADE STRING "7"
-- Retrieval info: PRIVATE: DIV_FACTOR0 NUMERIC "1"
-- Retrieval info: PRIVATE: DUTY_CYCLE0 STRING "50.00000000"
-- Retrieval info: PRIVATE: EFF_OUTPUT_FREQ_VALUE0 STRING "18.431999"
-- Retrieval info: PRIVATE: EXPLICIT_SWITCHOVER_COUNTER STRING "0"
-- Retrieval info: PRIVATE: EXT_FEEDBACK_RADIO STRING "0"
-- Retrieval info: PRIVATE: GLOCKED_COUNTER_EDIT_CHANGED STRING "1"
-- Retrieval info: PRIVATE: GLOCKED_FEATURE_ENABLED STRING "0"
-- Retrieval info: PRIVATE: GLOCKED_MODE_CHECK STRING "0"
-- Retrieval info: PRIVATE: GLOCK_COUNTER_EDIT NUMERIC "1048575"
-- Retrieval info: PRIVATE: HAS_MANUAL_SWITCHOVER STRING "1"
-- Retrieval info: PRIVATE: INCLK0_FREQ_EDIT STRING "50.000"
-- Retrieval info: PRIVATE: INCLK0_FREQ_UNIT_COMBO STRING "MHz"
-- Retrieval info: PRIVATE: INCLK1_FREQ_EDIT STRING "100.000"
-- Retrieval info: PRIVATE: INCLK1_FREQ_EDIT_CHANGED STRING "1"
-- Retrieval info: PRIVATE: INCLK1_FREQ_UNIT_CHANGED STRING "1"
-- Retrieval info: PRIVATE: INCLK1_FREQ_UNIT_COMBO STRING "MHz"
-- Retrieval info: PRIVATE: INTENDED_DEVICE_FAMILY STRING "Cyclone IV E"
-- Retrieval info: PRIVATE: INT_FEEDBACK__MODE_RADIO STRING "1"
-- Retrieval info: PRIVATE: LOCKED_OUTPUT_CHECK STRING "1"
-- Retrieval info: PRIVATE: LONG_SCAN_RADIO STRING "1"
-- Retrieval info: PRIVATE: LVDS_MODE_DATA_RATE STRING "Not Available"
-- Retrieval info: PRIVATE: LVDS_MODE_DATA_RATE_DIRTY NUMERIC "0"
-- Retrieval info: PRIVATE: LVDS_PHASE_SHIFT_UNIT0 STRING "deg"
-- Retrieval info: PRIVATE: MIG_DEVICE_SPEED_GRADE STRING "Any"
-- Retrieval info: PRIVATE: MIRROR_CLK0 STRING "0"
-- Retrieval info: PRIVATE: MULT_FACTOR0 NUMERIC "1"
-- Retrieval info: PRIVATE: NORMAL_MODE_RADIO STRING "1"
-- Retrieval info: PRIVATE: OUTPUT_FREQ0 STRING "18.43200000"
-- Retrieval info: PRIVATE: OUTPUT_FREQ_MODE0 STRING "1"
-- Retrieval info: PRIVATE: OUTPUT_FREQ_UNIT0 STRING "MHz"
-- Retrieval info: PRIVATE: PHASE_RECONFIG_FEATURE_ENABLED STRING "1"
-- Retrieval info: PRIVATE: PHASE_RECONFIG_INPUTS_CHECK STRING "0"
-- Retrieval info: PRIVATE: PHASE_SHIFT0 STRING "0.00000000"
-- Retrieval info: PRIVATE: PHASE_SHIFT_STEP_ENABLED_CHECK STRING "0"
-- Retrieval info: PRIVATE: PHASE_SHIFT_UNIT0 STRING "deg"
-- Retrieval info: PRIVATE: PLL_ADVANCED_PARAM_CHECK STRING "0"
-- Retrieval info: PRIVATE: PLL_ARESET_CHECK STRING "1"
-- Retrieval info: PRIVATE: PLL_AUTOPLL_CHECK NUMERIC "1"
-- Retrieval info: PRIVATE: PLL_ENHPLL_CHECK NUMERIC "0"
-- Retrieval info: PRIVATE: PLL_FASTPLL_CHECK NUMERIC "0"
-- Retrieval info: PRIVATE: PLL_FBMIMIC_CHECK STRING "0"
-- Retrieval info: PRIVATE: PLL_LVDS_PLL_CHECK NUMERIC "0"
-- Retrieval info: PRIVATE: PLL_PFDENA_CHECK STRING "0"
-- Retrieval info: PRIVATE: PLL_TARGET_HARCOPY_CHECK NUMERIC "0"
-- Retrieval info: PRIVATE: PRIMARY_CLK_COMBO STRING "inclk0"
-- Retrieval info: PRIVATE: RECONFIG_FILE STRING "pllAudioClock.mif"
-- Retrieval info: PRIVATE: SACN_INPUTS_CHECK STRING "0"
-- Retrieval info: PRIVATE: SCAN_FEATURE_ENABLED STRING "1"
-- Retrieval info: PRIVATE: SELF_RESET_LOCK_LOSS STRING "0"
-- Retrieval info: PRIVATE: SHORT_SCAN_RADIO STRING "0"
-- Retrieval info: PRIVATE: SPREAD_FEATURE_ENABLED STRING "0"
-- Retrieval info: PRIVATE: SPREAD_FREQ STRING "50.000"
-- Retrieval info: PRIVATE: SPREAD_FREQ_UNIT STRING "KHz"
-- Retrieval info: PRIVATE: SPREAD_PERCENT STRING "0.500"
-- Retrieval info: PRIVATE: SPREAD_USE STRING "0"
-- Retrieval info: PRIVATE: SRC_SYNCH_COMP_RADIO STRING "0"
-- Retrieval info: PRIVATE: STICKY_CLK0 STRING "1"
-- Retrieval info: PRIVATE: SWITCHOVER_COUNT_EDIT NUMERIC "1"
-- Retrieval info: PRIVATE: SWITCHOVER_FEATURE_ENABLED STRING "1"
-- Retrieval info: PRIVATE: SYNTH_WRAPPER_GEN_POSTFIX STRING "0"
-- Retrieval info: PRIVATE: USE_CLK0 STRING "1"
-- Retrieval info: PRIVATE: USE_CLKENA0 STRING "0"
-- Retrieval info: PRIVATE: USE_MIL_SPEED_GRADE NUMERIC "0"
-- Retrieval info: PRIVATE: ZERO_DELAY_RADIO STRING "0"
-- Retrieval info: LIBRARY: altera_mf altera_mf.altera_mf_components.all
-- Retrieval info: CONSTANT: BANDWIDTH_TYPE STRING "AUTO"
-- Retrieval info: CONSTANT: CLK0_DIVIDE_BY NUMERIC "3125"
-- Retrieval info: CONSTANT: CLK0_DUTY_CYCLE NUMERIC "50"
-- Retrieval info: CONSTANT: CLK0_MULTIPLY_BY NUMERIC "1152"
-- Retrieval info: CONSTANT: CLK0_PHASE_SHIFT STRING "0"
-- Retrieval info: CONSTANT: COMPENSATE_CLOCK STRING "CLK0"
-- Retrieval info: CONSTANT: INCLK0_INPUT_FREQUENCY NUMERIC "20000"
-- Retrieval info: CONSTANT: INTENDED_DEVICE_FAMILY STRING "Cyclone IV E"
-- Retrieval info: CONSTANT: LPM_TYPE STRING "altpll"
-- Retrieval info: CONSTANT: OPERATION_MODE STRING "NORMAL"
-- Retrieval info: CONSTANT: PLL_TYPE STRING "AUTO"
-- Retrieval info: CONSTANT: PORT_ACTIVECLOCK STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_ARESET STRING "PORT_USED"
-- Retrieval info: CONSTANT: PORT_CLKBAD0 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_CLKBAD1 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_CLKLOSS STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_CLKSWITCH STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_CONFIGUPDATE STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_FBIN STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_INCLK0 STRING "PORT_USED"
-- Retrieval info: CONSTANT: PORT_INCLK1 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_LOCKED STRING "PORT_USED"
-- Retrieval info: CONSTANT: PORT_PFDENA STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_PHASECOUNTERSELECT STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_PHASEDONE STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_PHASESTEP STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_PHASEUPDOWN STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_PLLENA STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_SCANACLR STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_SCANCLK STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_SCANCLKENA STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_SCANDATA STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_SCANDATAOUT STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_SCANDONE STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_SCANREAD STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_SCANWRITE STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_clk0 STRING "PORT_USED"
-- Retrieval info: CONSTANT: PORT_clk1 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_clk2 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_clk3 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_clk4 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_clk5 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_clkena0 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_clkena1 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_clkena2 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_clkena3 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_clkena4 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_clkena5 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_extclk0 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_extclk1 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_extclk2 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: PORT_extclk3 STRING "PORT_UNUSED"
-- Retrieval info: CONSTANT: SELF_RESET_ON_LOSS_LOCK STRING "OFF"
-- Retrieval info: CONSTANT: WIDTH_CLOCK NUMERIC "5"
-- Retrieval info: USED_PORT: @clk 0 0 5 0 OUTPUT_CLK_EXT VCC "@clk[4..0]"
-- Retrieval info: USED_PORT: @inclk 0 0 2 0 INPUT_CLK_EXT VCC "@inclk[1..0]"
-- Retrieval info: USED_PORT: areset 0 0 0 0 INPUT GND "areset"
-- Retrieval info: USED_PORT: c0 0 0 0 0 OUTPUT_CLK_EXT VCC "c0"
-- Retrieval info: USED_PORT: inclk0 0 0 0 0 INPUT_CLK_EXT GND "inclk0"
-- Retrieval info: USED_PORT: locked 0 0 0 0 OUTPUT GND "locked"
-- Retrieval info: CONNECT: @areset 0 0 0 0 areset 0 0 0 0
-- Retrieval info: CONNECT: @inclk 0 0 1 1 GND 0 0 0 0
-- Retrieval info: CONNECT: @inclk 0 0 1 0 inclk0 0 0 0 0
-- Retrieval info: CONNECT: c0 0 0 0 0 @clk 0 0 1 0
-- Retrieval info: CONNECT: locked 0 0 0 0 @locked 0 0 0 0
-- Retrieval info: GEN_FILE: TYPE_NORMAL pllAudioClock.vhd TRUE
-- Retrieval info: GEN_FILE: TYPE_NORMAL pllAudioClock.ppf TRUE
-- Retrieval info: GEN_FILE: TYPE_NORMAL pllAudioClock.inc FALSE
-- Retrieval info: GEN_FILE: TYPE_NORMAL pllAudioClock.cmp TRUE
-- Retrieval info: GEN_FILE: TYPE_NORMAL pllAudioClock.bsf FALSE
-- Retrieval info: GEN_FILE: TYPE_NORMAL pllAudioClock_inst.vhd FALSE
-- Retrieval info: LIB_FILE: altera_mf
-- Retrieval info: CBX_MODULE_PREFIX: ON
 
library verilog;
use verilog.vl_types.all;
entity MorseZHYT is
    port(
        X               : in     vl_logic_vector(5 downto 0);
        STOP            : out    vl_logic;
        Y               : out    vl_logic
    );
end MorseZHYT;

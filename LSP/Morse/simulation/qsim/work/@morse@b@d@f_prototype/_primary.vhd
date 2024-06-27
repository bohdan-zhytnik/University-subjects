library verilog;
use verilog.vl_types.all;
entity MorseBDF_prototype is
    port(
        Y               : out    vl_logic;
        X               : in     vl_logic_vector(5 downto 0);
        STOP            : out    vl_logic
    );
end MorseBDF_prototype;

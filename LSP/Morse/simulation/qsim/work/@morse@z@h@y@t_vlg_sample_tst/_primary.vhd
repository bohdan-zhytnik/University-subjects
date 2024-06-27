library verilog;
use verilog.vl_types.all;
entity MorseZHYT_vlg_sample_tst is
    port(
        X               : in     vl_logic_vector(5 downto 0);
        sampler_tx      : out    vl_logic
    );
end MorseZHYT_vlg_sample_tst;

library verilog;
use verilog.vl_types.all;
entity MorseBDF_prototype_vlg_sample_tst is
    port(
        X               : in     vl_logic_vector(5 downto 0);
        sampler_tx      : out    vl_logic
    );
end MorseBDF_prototype_vlg_sample_tst;

library verilog;
use verilog.vl_types.all;
entity Morse_vlg_sample_tst is
    port(
        X               : in     vl_logic_vector(3 downto 0);
        sampler_tx      : out    vl_logic
    );
end Morse_vlg_sample_tst;

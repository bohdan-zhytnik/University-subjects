library verilog;
use verilog.vl_types.all;
entity Morse_vlg_check_tst is
    port(
        STOP            : in     vl_logic;
        Y               : in     vl_logic;
        sampler_rx      : in     vl_logic
    );
end Morse_vlg_check_tst;

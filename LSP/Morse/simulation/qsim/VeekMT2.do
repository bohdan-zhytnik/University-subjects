onerror {quit -f}
vlib work
vlog -work work VeekMT2.vo
vlog -work work VeekMT2.vt
vsim -novopt -c -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cycloneive_ver work.MorseBDF_vlg_vec_tst
vcd file -direction VeekMT2.msim.vcd
vcd add -internal MorseBDF_vlg_vec_tst/*
vcd add -internal MorseBDF_vlg_vec_tst/i1/*
add wave /*
run -all

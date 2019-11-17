open_project systolic
set_top systolic
add_files systolic.cpp
add_files systolic.h
add_files systolic_testbench.cpp -tb
open_solution "solution1"
set_part {xc7z020clg400-1} -tool vivado
create_clock -period 10 -name default

# source "./solution1/directives.tcl"
#csim_design
csynth_design
#cosim_design -trace_level all -tool xsim
export_design -format ip_catalog

exit

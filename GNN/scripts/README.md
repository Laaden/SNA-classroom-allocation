# README Scripts

Directory contains scripts for compiling the GNN cluster program for portable distribution. Note that build.sh should be run from the root of the GNN folder. Ensure your cd is pointing at GNN and not scripts, or the root of the git repo.

- `build.sh`: bash script for creating the precompilation script, and calling the create_app script
  - this will also create a temporary compile_trace file that is used for making the executable AOT instead of JIT
- `test_input.json`: test data used to create the compile_trace.jl

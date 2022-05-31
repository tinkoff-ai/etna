mprof run -o profile_results/old_logic.dat old_logic.py
mprof run -o profile_results/new_logic_inplace.dat new_logic_inplace.py
mprof run -o profile_results/new_logic_copy.dat new_logic_copy.py
mprof plot -o \
  profile_results/results.png \
  profile_results/old_logic.dat \
  profile_results/new_logic_inplace.dat \
  profile_results/new_logic_copy.dat

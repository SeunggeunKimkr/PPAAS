python PPAAS/train.py --algo HER --env_name COMP --project_name comp --seed 50 --concat_all_specs False --SoF True --goal_in_obs False \
    --name test  --n_warmup 8 --obj_style softmax --temp 5.0 --conservative True --alpha 0.0  --tt_threshold -1.0 --tt_threshold_2 -3.0 \
    --lookup_style tanh --learning_starts 120 --n_goal 1 --lr 0.003 --gamma 0.8 --goal future \
    --total_timesteps 8000 --eval_freq 8000 --num_eval 100 --pareto_freq 1 \
    --yaml ./eval_engines/ngspice/ngspice_inputs/yaml_files/comparator_gf180.yaml \
    --eval_yaml ./eval_engines/ngspice/ngspice_inputs/yaml_files/comparator_gf180.yaml \
    --eval_spec_path ngspice_specs_gen_comp_gf180_test_100
 

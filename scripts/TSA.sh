entity=seunggeun_kim_kr-austin # change this to your entity name
python PPAAS/train.py --algo HER --env_name TSA --project_name tsa --seed 7 --concat_all_specs False --SoF True --goal_in_obs False \
    --name test  --entity $entity --n_warmup 4 --obj_style softmax --temp 5.0 --conservative True --alpha 0.0  --tt_threshold -1.0 --tt_threshold_2 -3.0 \
    --lookup_style tanh --learning_starts 120 --n_goal 1 --lr 0.003 --gamma 0.8 --goal future \
    --total_timesteps 12000 --eval_freq 3000 --num_eval 10 --pareto_freq 1 \
    --yaml ./eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp_gf180.yaml \
    --eval_yaml ./eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp_gf180.yaml \
    --eval_spec_path ngspice_specs_gen_tsa_gf180_tran_test_200
    
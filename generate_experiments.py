
import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
exp_name = str(sys.argv[4])
base_call = "python main.py"
output_file = open(generated_name, "w")
seeds = 2
condition = "initial_pendulum_experiments"
env_name = "pendulum"
planner = "cem"
plan_horizons= [5,10,20,30,50,100]
num_candidates = [100,200,500,1000,2000]
top_candidates = [10,50,100,200]
action_stds = [0.1,0.5,1,2]
for plan_horizon in plan_horizons:
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --env_name " + str(env_name) + " --planner " + str(planner) + " --plan_horizon " + str(plan_horizon)
        print(final_call)
        print(final_call, file=output_file)

for n_cand in num_candidates:
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --env_name " + str(env_name) + " --planner " + str(planner) +  " --num_candidates " + str(n_cand)
    
        print(final_call)
        print(final_call, file=output_file)

for top_cand in top_candidates:
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --env_name " + str(env_name) + " --planner " + str(planner) + " --top_candidates " + str(top_cand)
        print(final_call)
        print(final_call, file=output_file)

for action_std in action_stds:
    for s in range(seeds):
        lpath = log_path + "/"+str(exp_name) +"_"+condition + "/" + str(s)
        spath = save_path + "/" + str(exp_name) +"_"+condition + "/" + str(s)
        final_call = base_call + " --logdir " + str(lpath) + " --savedir " + str(spath) + " --env_name " + str(env_name) + " --planner " + str(planner) + " --action_std " + str(action_std)
        print(final_call)
        print(final_call, file=output_file)
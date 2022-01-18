#!/bin/bash

#hiperparameters
LEARNING_RATE=("1e-3" "1e-4")           #default 1e-3
DISCOUNT=("0.99" "0.95" "0.8")          #default 0.99

# BATCH_SIZE=16                         #default 16         HARDCODED
# SAMPLES_PER_INSERT="None"             #default None       HARDCODED

#only for Impala
ENTROPY_COST=("0.01" "0.1")             #default 0.01
MAX_ABS_REWARD=("100.0" "None")         #default np.inf (== None)


NUM_STEPS = 60000   #~30 min


all_exp=$((${#LEARNING_RATE[@]} * ${#DISCOUNT[@]} * ${#ENTROPY_LOST[@]} * ${#MAX_ABS_REWARD[@]}))  #24 experiments

STARTING_EXP_ID=0  #starting point

curr_exp_id=0
for lr in "${LEARNING_RATE[@]}"; do
    for discount in "${DISCOUNT[@]}"; do
        for entropy_cost in "${ENTROPY_COST[@]}"; do
            for max_abs_reward in "${MAX_ABS_REWARD[@]}"; do
                ((curr_exp_id++))
                if [ ${curr_exp_id} -lt ${STARTING_EXP_ID} ]; then
                    continue
                fi
                echo -e "Starting experiment $((curr_exp_id)) / ${all_exps}"
                echo -e "Model:IMPALA, lr:${lr}, discount:${discount}, entropy_cost:${entropy_cost}, max_abs_reward:${max_abs_reward}\n"

                python src/main.py \
                    --num_steps ${NUM_STEPS} \
                    --alg impala \
                    --save_video 0
                    --save_csv 1
                    --lr ${lr} \
                    --discount ${discount} \
                    --entropy_cost ${entropy_cost} \
                    --max_abs_reward ${max_abs_reward}

                retVal=$?
                if [ $retVal -ne 0 ]; then
                    echo -e "\n\n Last run experiment $((curr_exp_id)) / ${all_exps}"
                    echo -e "Model:IMPALA, lr:${lr}, discount:${discount}, entropy_cost:${entropy_cost}, max_abs_reward:${max_abs_reward}\n"
                    exit $retVal
                fi
            done
        done
    done
done
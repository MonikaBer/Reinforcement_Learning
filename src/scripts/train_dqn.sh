#!/bin/bash

#hiperparameters
LEARNING_RATE=("1e-3" "1e-4")           #default 1e-3
DISCOUNT=("0.99" "0.95" "0.8")          #default 0.99
TARGET_UPDATE_PERIOD=("75" "400")       #default 100


all_exps=$((${#LEARNING_RATE[@]} * ${#DISCOUNT[@]} * ${#TARGET_UPDATE_PERIOD[@]}))  #12 experiments

FIRST_EXP_ID=1  #start point
LAST_EXP_ID=12   #end point

curr_exp_id=0
for lr in "${LEARNING_RATE[@]}"; do
    for discount in "${DISCOUNT[@]}"; do
        for target_update_period in "${TARGET_UPDATE_PERIOD[@]}"; do
            ((curr_exp_id++))
            if [ ${curr_exp_id} -lt ${FIRST_EXP_ID} ]; then
                continue
            fi
            echo -e "Starting experiment $((curr_exp_id)) / ${all_exps}"
            echo -e "Model:DQN, lr:${lr}, discount:${discount}, target_update_period:${target_update_period}\n"

            python src/main.py \
                --num_episodes $1 \
                --alg dqn \
                --save_video 1 \
                --save_csv 1 \
                --max_steps $2 \
                --collect_frames $3 \
                --lr ${lr} \
                --discount ${discount} \
                --target_update_period ${target_update_period}

            retVal=$?
            if [ $retVal -ne 0 ]; then
                echo -e "\n\n Last run experiment $((curr_exp_id)) / ${all_exps}"
                echo -e "Model:DQN, lr:${lr}, discount:${discount}, target_update_period:${target_update_period}\n"
                exit $retVal
            fi

            if [ ${curr_exp_id} -ge ${LAST_EXP_ID} ]; then
                exit $retVal
            fi
        done
    done
done

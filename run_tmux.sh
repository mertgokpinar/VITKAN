#!/bin/bash

SESSION_NAME="vitkan"
tmux send-keys -t $SESSION_NAME:0 'conda activate patchgcn3; CUDA_VISIBLE_DEVICES=0 wandb agent --count 125 (your_sweep_cmd_here)' C-m
tmux new-window -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME:1 'conda activate patchgcn3; CUDA_VISIBLE_DEVICES=0 wandb agent --count 125 (your_sweep_cmd_here)' C-m
tmux new-window -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME:2 'conda activate patchgcn3; CUDA_VISIBLE_DEVICES=0 wandb agent --count 125 (your_sweep_cmd_here)' C-m
tmux new-window -t $SESSION_NAME
tmux send-keys -t $SESSION_NAME:3 'conda activate patchgcn3; CUDA_VISIBLE_DEVICES=0 wandb agent --count 125 (your_sweep_cmd_here)' C-m

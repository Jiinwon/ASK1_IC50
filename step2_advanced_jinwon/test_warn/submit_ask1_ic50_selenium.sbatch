#!/usr/bin/env bash
#SBATCH -p gpu1
#SBATCH --gres=gpu:1
#SBATCH -J ask1-ic50-sel
#SBATCH -o /home1/won0316/DACON/JUMP_AI_2025_EST/step2_advanced_jinwon/test_warn/selenium_logs/sbatch_%j.out
#SBATCH -t 06:00:00
#SBATCH --mem=16G

# (선택) 프록시가 필요하면 활성화
# export HTTPS_PROXY="http://proxy.host:3128"
# export HTTP_PROXY="http://proxy.host:3128"

# Conda 초기화 및 환경
source ~/.bashrc
conda activate toxcast_env

# 로그 디렉터리
mkdir -p /home1/won0316/DACON/JUMP_AI_2025_EST/step2_advanced_jinwon/test_warn/selenium_logs

python /gpfs/home1/won0316/DACON/JUMP_AI_2025_EST/step2_advanced_jinwon/test_warn/ask1_ic50_selenium.py

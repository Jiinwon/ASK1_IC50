#!/usr/bin/env bash
# 1시간마다 git add, commit, push를 수행하는 자동 스크립트

# --- 사용자 설정 ---
REPO_DIR="/home1/won0316/DACON/JUMP_AI_2025_EST/step2_advanced_jinwon"
LOGFILE="$REPO_DIR/auto_git_sync.log"

# 로그 디렉터리 및 파일 초기화
mkdir -p "$(dirname "$LOGFILE")"
# exec으로 모든 출력(표준/에러)을 로그 파일로 리다이렉트
exec >>"$LOGFILE" 2>&1

# 무한 루프 시작
while true; do
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$TIMESTAMP] Starting sync cycle."

  cd "$REPO_DIR" || {
      echo "[$TIMESTAMP] ERROR: Cannot cd to $REPO_DIR";
      exit 1;
  }

  # 변경사항이 있을 때만 커밋
  if [ -n "$(git status --porcelain)" ]; then
    echo "[$TIMESTAMP] Changes detected, staging files..."
    git add .

    echo "[$TIMESTAMP] Committing changes..."
    git commit -m "자동 커밋: $TIMESTAMP"

    echo "[$TIMESTAMP] Pushing to origin main..."
    git push origin main

    echo "[$TIMESTAMP] Sync complete."
  else
    echo "[$TIMESTAMP] No changes to commit."
  fi

  # 1시간 대기
  sleep 3600

done
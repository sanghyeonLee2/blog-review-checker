#!/bin/bash

# 사용법: ./commit.sh "커밋 메시지" "2024-05-08T15:00:00"

MESSAGE=$1
DATE=$2

if [ -z "$MESSAGE" ] || [ -z "$DATE" ]; then
  echo "❗ 사용법: ./commit.sh \"커밋 메시지\" \"YYYY-MM-DDTHH:MM:SS\""
  exit 1
fi

git add .
GIT_COMMITTER_DATE="$DATE" git commit --date="$DATE" -m "$MESSAGE"

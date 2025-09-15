#!/bin/bash
set -e  # 에러 발생 시 스크립트 종료
if [ -f .env ]; then
  echo "Loading environment variables from .env..."
  set -a
  source .env
  set +a
else
  echo ".env file not found! Exiting..."
  exit 1
fi
echo "Setting executable permissions for Python scripts..."
find . -name "*.py" -type f -exec chmod +x {} \;
find . -name "*.cpp" -type f -exec chmod +x {} \;
find . -name "*.launch" -type f -exec chmod +x {} \;
sleep 3
roslaunch interaction_manager interaction_manager.launch
wait
#!/bin/bash

# Python 스크립트 실행 권한 설정
echo "Setting executable permissions for Python scripts..."
find /home/$USER/CMU-VLA-Challenge -name "*.py" -type f -exec chmod +x {} \;
find /home/$USER/CMU-VLA-Challenge -name "*.cpp" -type f -exec chmod +x {} \;
find /home/$USER/CMU-VLA-Challenge -name "*.launch" -type f -exec chmod +x {} \;

sleep 3
# interaction_manager와 color_scan_generation을 함께 실행
roslaunch interaction_manager interaction_manager.launch 
# 모든 백그라운드 프로세스가 완료될 때까지 대기
wait
# CMU-VLA-Challenge Exploration Manager

ROS Noetic 기반의 자율 탐색 시스템입니다.

## 🐳 Docker 이미지 사용법

### 1. 도커 이미지 다운로드
```bash
docker pull your_username/exploration_robot:latest
```

### 2. 프로젝트 코드 다운로드
```bash
git clone https://github.com/your_username/CMU-VLA-Challenge.git
cd CMU-VLA-Challenge
```

### 3. 컨테이너 실행
```bash
docker run -it --rm \
  -v $(pwd):/home/aailab/cwh316/CMU-VLA-Challenge:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  --network host \
  your_username/exploration_robot:latest
```

### 4. 프로젝트 빌드 및 실행
```bash
# 컨테이너 내부에서
cd /home/aailab/cwh316/CMU-VLA-Challenge/ai_module
catkin_make
source devel/setup.bash

# Graph Builder 실행
rosrun exploration_manager graph_builder_node.py

# Exploration Executor 실행
rosrun exploration_manager exploration_executor_node.py

# 또는 launch 파일로 실행
roslaunch exploration_manager exploration_manager.launch
```

## 📁 프로젝트 구조

```
ai_module/
├── src/
│   └── exploration_manager/
│       ├── src/
│       │   ├── graph_builder_node.py      # 그래프 생성 노드
│       │   ├── exploration_executor_node.py # 탐색 실행 노드
│       │   └── graph_visualizer_node.py   # 그래프 시각화 노드
│       ├── launch/
│       │   └── exploration_manager.launch
│       ├── config/
│       │   └── exploration.rviz
│       └── package.xml
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

## 🔧 필요한 토픽

### 입력 토픽
- `/state_estimation` (Odometry): 로봇 위치 정보
- `/traversable_area` (PointCloud2): 통행 가능 영역

### 출력 토픽
- `/exploration/occupancy_grid` (OccupancyGrid): 기본 그리드
- `/exploration/inflated_grid` (OccupancyGrid): inflation 적용 그리드
- `/exploration/traversable_graph` (OccupancyGrid): 그래프 노드
- `/waypoint_with_heading` (PoseStamped): 로봇 이동 목표점

## 📊 저장되는 파일

- `/tmp/graph_data.json`: 그래프 데이터
- `/tmp/exploration_results.json`: 탐색 결과

## 🚀 빠른 시작

```bash
# 1. 이미지 다운로드
docker pull your_username/exploration_robot:latest

# 2. 코드 다운로드
git clone https://github.com/your_username/CMU-VLA-Challenge.git

# 3. 실행
docker run -it --rm \
  -v $(pwd)/CMU-VLA-Challenge:/home/aailab/cwh316/CMU-VLA-Challenge:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  --network host \
  your_username/exploration_robot:latest
```

## 📝 라이센스

MIT License 
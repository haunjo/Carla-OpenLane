Hard Scenario Mode (악천후 + 근접 차량 혼잡도)

본 프로젝트에는 기존 Carla-OpenLane 데이터셋의 다양성과 난이도를 높이기 위해 Hard Scenario 모드를 추가하였다. 이 모드는 일반적인 CARLA 기본 환경이 아닌, 실제 도로에서 자주 발생하는 악천후 상황과 ego 차량 주변의 높은 교통 혼잡도를 인위적으로 생성하여 보다 현실적인 학습 데이터를 수집하는 것을 목표로 한다.

Hard Scenario 모드는 run_bh.sh에서 다음 옵션이 활성화되면 자동으로 적용된다.

--hard-scenario --hard-occluders 100 --hard-jaywalkers 0


또는 Python 스크립트를 직접 실행할 경우에도 동일한 옵션을 전달하여 사용할 수 있다.

악천후 환경 생성

Hard Scenario 모드에서는 CARLA에서 제공하는 기본 weather preset을 그대로 사용하지 않고, 별도로 정의한 4가지 극단적인 날씨 설정 중 하나를 랜덤으로 적용한다.

강한 태양광 눈부심이 발생하는 맑은 낮 환경

시야가 거의 확보되지 않는 초고농도 안개 환경

강우와 젖은 노면이 동반된 주간 환경

강우와 젖은 노면이 동반된 야간 환경

이를 통해 조명 변화, 가시성 저하, 노면 반사, 센서 노이즈 등 실제 주행 환경에서 발생하는 다양한 어려운 조건을 데이터에 반영할 수 있다.

ego 차량 근처 교통 혼잡도 증가

Hard Scenario 모드에서는 ego 차량 주변에 추가 차량을 강제로 생성하여 시야 차단(occlusion)과 교통 밀집 상황을 만든다.

구현 상 ego 차량의 현재 위치를 기준으로 반경 60m 이내에 존재하는 스폰 포인트들을 후보로 수집하고, 이 중에서 최대 --hard-occluders 값만큼 차량 생성을 시도한다.

즉, 다음 코드에서 60.0 값이 혼잡도 적용 범위를 의미한다.

if sp.location.distance(center_loc) < 60.0:


여기서:

60.0은 ego 차량 기준 탐색 반경(미터 단위)

--hard-occluders는 생성 시도할 최대 차량 수

실제 생성되는 차량 수는 맵 구조, 스폰 포인트 밀도, 충돌 여부에 따라 줄어들 수 있다

생성된 차량들은 autopilot 모드로 주행하도록 설정된다

이 기능을 통해 카메라 전방 및 측면 시야가 차량에 의해 가려지는 현실적인 상황을 반복적으로 생성할 수 있다.

카메라 안정화 처리

시나리오가 전환되거나 occluder 차량이 새로 스폰되는 직후에는 CARLA 내부 동기화 문제로 인해 카메라가 순간적으로 기울어지거나 잘못된 자세를 가지는 경우가 발생할 수 있다.

이를 방지하기 위해 Hard Scenario 모드에서는 다음과 같은 안정화 과정을 거친다.

시나리오 전환 직후 여러 tick 대기

occluder / walker 생성 직후 추가 tick 대기

센서 큐 flush 이후 캡처 수행

이를 통해 카메라 pitch, yaw 값이 정상적으로 고정된 이후에만 프레임을 저장하도록 하여 데이터 품질을 안정화하였다.

주요 파라미터 요약

run_bh.sh 및 data_capture_Argoverse2_BH.py에서 조절 가능한 핵심 파라미터는 다음과 같다.

SCENES : Town 당 생성할 시나리오 수

--sample : ego 차량이 이동하며 캡처할 waypoint 개수

--step : waypoint 간 거리 (미터)

--traffic-level : 배경 차량 밀도 (1 = 없음, 2 = 기본 교통량)

--spawn-offset : ego 시작 위치 시드

--hard-occluders : Hard Scenario에서 생성할 근접 차량 수
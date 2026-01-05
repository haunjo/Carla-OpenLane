#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate carla

SCENES=51

# CARLA Weather Presets (17개)
WEATHER_PRESETS=(
    "ClearNoon"
    "ClearSunset"
    "CloudyNoon"
    "CloudySunset"
    "SoftRainNoon"
    "SoftRainSunset"
    "MidRainyNoon"
    "WetNoon"
    "WetSunset"
    "WetCloudyNoon"
    "MidRainSunset"
    "WetCloudySunset"
    "HardRainNoon"
    "HardRainSunset"
    "ClearNight"
    "CloudyNight"
    "MidRainyNight"
)

# 사용자가 실행할 Town 선택
# - 인자 또는 환경변수 TOWNS 를 통해 지정 가능
# - 허용 예시: "1 2 3", "Town01 Town03", "1,3,7,10"
normalize_town() {
    local raw="$1"
    local t="$raw"
    # 쉼표 제거 및 공백 트림은 상위에서 처리
    # Prefix가 없으면 붙여줌
    if [[ ! "$t" =~ ^[Tt][Oo][Ww][Nn] ]]; then
        t="Town${t}"
    fi
    # Town1 → Town01 로 정규화 (10 은 예외)
    if [[ "$t" =~ ^[Tt][Oo][Ww][Nn]([0-9]+)$ ]]; then
        local num="${BASH_REMATCH[1]}"
        if [[ "$num" == "10" ]]; then
            echo "Town10"
        else
            printf "Town%02d\n" "$num"
        fi
        return 0
    fi
    # 이미 정규화된 형태라면 대문자화만 보장
    echo "${t^}"
}

build_town_list() {
    local -a inputs=("$@")
    local -a out=()
    for it in "${inputs[@]}"; do
        # 콤마 분리 지원
        IFS=',' read -ra parts <<< "$it"
        for p in "${parts[@]}"; do
            [[ -z "$p" ]] && continue
            out+=("$(normalize_town "$p")")
        done
    done
    echo "${out[@]}"
}

if [[ $# -gt 0 ]]; then
    # 스크립트 인자로 Town 선택
    read -ra SELECTED_TOWNS <<< "$(build_town_list "$@")"
elif [[ -n "$TOWNS" ]]; then
    # 환경변수 TOWNS 로 Town 선택
    # 예: TOWNS="1,2,3" 또는 "Town01 Town03"
    read -ra SELECTED_TOWNS <<< "$(build_town_list $TOWNS)"
else
    # 기본값: Town01~Town06(train) Town07,Town10(val)
    SELECTED_TOWNS=(Town11)
fi

echo "🎯 선택된 Towns: ${SELECTED_TOWNS[*]}"

# 함수: Carla 서버가 열릴 때까지 대기
wait_for_carla() {
    local RETRIES=30
    for ((i=0; i<RETRIES; i++)); do
        if nc -z localhost 2000; then
            echo "✅ Carla 서버가 포트 2000에서 실행 중입니다."
            return 0
        fi
        echo "⏳ Carla 서버 대기 중... (${i}/${RETRIES})"
        sleep 1
    done
    echo "❌ Carla 서버가 포트 2000에서 응답하지 않습니다."
    return 1
}


# 기존 CARLA 프로세스 완전 정리
echo "🧹 기존 CARLA 프로세스 정리 중..."

# 1. CARLA 프로세스 종료
if pgrep -x "CarlaUE4" > /dev/null; then
    echo "  - CARLA 프로세스 종료"
    pkill -9 CarlaUE4
    sleep 1
fi

# 2. Python 데이터 수집 프로세스 종료
if pgrep -f "data_capture" > /dev/null; then
    echo "  - 데이터 수집 프로세스 종료"
    pkill -9 -f data_capture
fi

# 3. 포트 2000 (CARLA 서버) 정리
if lsof -i :2000 &>/dev/null; then
    echo "  - 포트 2000 정리"
    fuser -k 2000/tcp 2>/dev/null
fi

# 4. 포트 8000 (Traffic Manager) 정리
if lsof -i :8000 &>/dev/null; then
    echo "  - 포트 8000 (Traffic Manager) 정리"
    fuser -k 8000/tcp 2>/dev/null
fi

# 5. 안정화 대기
sleep 3
echo "✅ 정리 완료"


echo "🚀 Carla 시뮬레이터 실행 시작"
./CarlaUE4.sh -prefernvidia -RenderOffScreen -nosound -quality-level=Epic &
CARLA_PID=$!

# Carla 실행 대기
if ! wait_for_carla; then
    echo "❌ Carla 실행 실패, 스크립트 중단"
    kill -9 $CARLA_PID 2>/dev/null
    exit 1
fi

# 기존 데이터는 offset 0~33을 사용했으므로 34부터 시작 (scene 17~33 수집)
BASE_OFFSET=0

for TOWN in "${SELECTED_TOWNS[@]}"; do
    echo "========================================="
    echo "🗺️  맵 전환: $TOWN"
    echo "========================================="
    python ./PythonAPI/util/config.py --map "$TOWN"

    for traffic_level in 1 2; do
        echo "#########################################"
        echo "# 혼잡도 단계: $traffic_level (맵: $TOWN)"
        echo "#########################################"

        for ((i=35; i<$SCENES; i++)); do
            echo "=============================="
            echo "사이클 $i 시작 (혼잡도 $traffic_level, 맵 $TOWN)"

            # offset 계산: BASE_OFFSET(35) + traffic_level별 offset
            if [[ $traffic_level -eq 1 ]]; then
                offset=$((BASE_OFFSET + i))
            elif [[ $traffic_level -eq 2 ]]; then
                offset=$((BASE_OFFSET + SCENES + i))
            fi

            # Weather preset 선택: scene index를 기준으로 17개 weather 순환
            weather_idx=$((i % 17))
            WEATHER="${WEATHER_PRESETS[$weather_idx]}"

            echo "🌦️  weather=$WEATHER, traffic=$traffic_level, cycle=$i, offset=$offset, town=$TOWN"
            echo "사이클 $i: 데이터 캡처 시작 (offset=$offset, town=$TOWN, weather=$WEATHER)"
            python ./data_capture_Argoverse2.py \
                --scene $SCENES \
                --sample 5 \
                --spawn-offset $offset \
                --traffic-level $traffic_level \
                --weather "$WEATHER" \
                --dir val \
                --pose-format openlanev2
            echo "사이클 $i: 데이터 캡처 완료"
            echo "사이클 $i 끝!"
            echo "=============================="
        done
    done
done

echo "✅ 모든 사이클 종료. Carla 시뮬레이터 종료 중..."
kill $CARLA_PID
wait $CARLA_PID 2>/dev/null
echo "🛑 Carla 시뮬레이터 정상 종료 완료."

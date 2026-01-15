#!/bin/bash
set -euo pipefail

# =========================
# CONFIG
# =========================
source ~/anaconda3/etc/profile.d/conda.sh
conda activate carla

CARLA_PATH="/home/vip-dell/CARLA_0.9.15"
SCENES=51

# 반드시 "Carla-OpenLane/Carla" 에서 실행되도록 강제
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 출력 경로(절대경로로 고정) - 여기 바꾸고 싶으면 여기만!
OUTDIR="/data/ECCV2/Carla-OpenLane/Carla/val_bh"
mkdir -p "$OUTDIR"

# 로그/코어 저장 디렉토리
LOGDIR="$OUTDIR/_logs"
COREDIR="$OUTDIR/_cores"
mkdir -p "$LOGDIR" "$COREDIR"

# 세그폴트 대비: 코어덤프 허용
ulimit -c unlimited

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

# hard 시나리오 옵션 (HARD=0이면 끔)
HARD_ARGS=()
if [[ "${HARD:-1}" -eq 1 ]]; then
  HARD_ARGS=(--hard-scenario --hard-occluders 100 --hard-jaywalkers 0)
fi

normalize_town() {
  local raw="$1"
  local t="$raw"
  if [[ ! "$t" =~ ^[Tt][Oo][Ww][Nn] ]]; then
    t="Town${t}"
  fi
  if [[ "$t" =~ ^[Tt][Oo][Ww][Nn]([0-9]+)$ ]]; then
    local num="${BASH_REMATCH[1]}"
    if [[ "$num" == "10" ]]; then
      echo "Town10"
    else
      printf "Town%02d\n" "$num"
    fi
    return 0
  fi
  echo "${t^}"
}

build_town_list() {
  local -a inputs=("$@")
  local -a out=()
  for it in "${inputs[@]}"; do
    IFS=',' read -ra parts <<< "$it"
    for p in "${parts[@]}"; do
      [[ -z "$p" ]] && continue
      out+=("$(normalize_town "$p")")
    done
  done
  echo "${out[@]}"
}

if [[ $# -gt 0 ]]; then
  read -ra SELECTED_TOWNS <<< "$(build_town_list "$@")"
elif [[ -n "${TOWNS:-}" ]]; then
  read -ra SELECTED_TOWNS <<< "$(build_town_list $TOWNS)"
else
  SELECTED_TOWNS=(Town11)
fi

echo "🎯 선택된 Towns: ${SELECTED_TOWNS[*]}"
echo "📦 OUTDIR: $OUTDIR"
echo "📍 WORKDIR: $SCRIPT_DIR"
echo "🧾 LOGDIR: $LOGDIR"
echo "💥 COREDIR: $COREDIR"
echo "🧠 HARD: ${HARD:-1} (${HARD_ARGS[*]:-disabled})"

wait_for_carla() {
  local RETRIES=60
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

cleanup() {
  echo ""
  echo "🧹 [cleanup] 종료/정리 시작..."

  # 현재 스크립트가 띄운 CARLA만 종료 (전체 pkill 지양)
  if [[ -n "${CARLA_PID:-}" ]] && kill -0 "$CARLA_PID" 2>/dev/null; then
    echo "  - CARLA 종료 (pid=$CARLA_PID)"
    kill "$CARLA_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$CARLA_PID" 2>/dev/null || true
  fi

  # 포트 정리(잔존 시)
  if lsof -i :2000 &>/dev/null; then
    echo "  - 포트 2000 정리"
    fuser -k 2000/tcp 2>/dev/null || true
  fi
  if lsof -i :8000 &>/dev/null; then
    echo "  - 포트 8000 정리"
    fuser -k 8000/tcp 2>/dev/null || true
  fi

  echo "✅ [cleanup] 완료"
}
trap cleanup EXIT INT TERM

# =========================
# PRE-CLEAN (강제 정리)
# =========================
echo "🧹 기존 CARLA/수집 프로세스 정리 중..."
pkill -9 CarlaUE4 2>/dev/null || true
pkill -9 -f "data_capture_Argoverse2_BH.py" 2>/dev/null || true

lsof -i :2000 &>/dev/null && fuser -k 2000/tcp 2>/dev/null || true
lsof -i :8000 &>/dev/null && fuser -k 8000/tcp 2>/dev/null || true
sleep 3
echo "✅ 정리 완료"

# =========================
# RUN CARLA
# =========================
echo "🚀 Carla 시뮬레이터 실행 시작"
"$CARLA_PATH/CarlaUE4.sh" -prefernvidia -RenderOffScreen -nosound -quality-level=Epic &
CARLA_PID=$!

if ! wait_for_carla; then
  echo "❌ Carla 실행 실패"
  exit 1
fi

# =========================
# RUN LOOPS
# =========================
BASE_OFFSET=0

for TOWN in "${SELECTED_TOWNS[@]}"; do
  echo "========================================="
  echo "🗺️  맵 전환: $TOWN"
  echo "========================================="
  python "$CARLA_PATH/PythonAPI/util/config.py" --map "$TOWN"

  # 맵 로딩 안정화 대기
  sleep 3

  for traffic_level in 1 2; do
    echo "#########################################"
    echo "# 혼잡도 단계: $traffic_level (맵: $TOWN)"
    echo "#########################################"

    for ((i=35; i<SCENES; i++)); do
      echo "=============================="
      echo "사이클 $i 시작 (혼잡도 $traffic_level, 맵 $TOWN)"

      if [[ $traffic_level -eq 1 ]]; then
        offset=$((BASE_OFFSET + i))
      else
        offset=$((BASE_OFFSET + SCENES + i))
      fi

      # offset 방어
      if [[ -z "${offset:-}" ]] || ! [[ "$offset" =~ ^-?[0-9]+$ ]]; then
        echo "❌ invalid offset: [$offset]" >&2
        exit 1
      fi

      weather_idx=$((i % 17))
      WEATHER="${WEATHER_PRESETS[$weather_idx]}"

      echo "🌦️  weather=$WEATHER, traffic=$traffic_level, cycle=$i, offset=$offset, town=$TOWN"
      echo "📌 capture -> $OUTDIR"

      LOGFILE="$LOGDIR/${TOWN}_tl${traffic_level}_i${i}_off${offset}_w${WEATHER}.log"
      echo "🧾 log -> $LOGFILE"

      # 세그폴트/에러 발생 시 즉시 종료되게(set -e) + 로그 보존(tee)
      # NOTE: pipefail 때문에 python 비정상 종료/세그폴트 시 전체 스크립트가 즉시 종료됨.
      PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
      python ./data_capture_Argoverse2_BH.py \
        --sample 5 \
        --step 12 \
        --spawn-offset "$offset" \
        --traffic-level "$traffic_level" \
        --weather "$WEATHER" \
        --dir "$OUTDIR" \
        --pose-format openlanev2 \
        "${HARD_ARGS[@]}" \
        2>&1 | tee "$LOGFILE"

      echo "사이클 $i: 데이터 캡처 완료"
      echo "=============================="
    done
  done
done

echo "✅ 모든 사이클 종료."

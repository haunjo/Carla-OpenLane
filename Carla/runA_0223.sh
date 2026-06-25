#!/bin/bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate carla

CARLA_PATH="/home/vip-dell/CARLA_0.9.15"
OUTROOT="${OUTROOT:-/media/vip-dell/HC1/ECCV2/Carla-OpenLane/subsetA_7}"
LOGDIR="$OUTROOT/_logs"
mkdir -p "$OUTROOT" "$LOGDIR"

CAPTURE_PY="./data_capture_unified_BH_0222.py"  # CHANGE: Renamed to unified (supports both Argoverse2 and nuScenes)
SUBSET="argoverse2"

if [[ ! -f "$CAPTURE_PY" ]]; then echo "❌ $CAPTURE_PY not found"; exit 1; fi
if [[ ! -f "./openlane_v2_subset_A.json" ]]; then echo '{}' > ./openlane_v2_subset_A.json; fi
touch "$OUTROOT/_write_test" && rm -f "$OUTROOT/_write_test"

TRAIN_TOWNS=(Town01 Town02 Town03 Town04 Town05 Town06)
VAL_TOWNS=(Town10HD)
SAMPLE_PER_SEQ=20
TARGET_TRAIN_SEQ=272
TARGET_VAL_SEQ=63

# Weather: clear 6% | wet 13% | rain 21% | fog★(NEW) 30% | night 30% (HardRainyNight★ 5%)
# Traffic: sparse 10% | normal 50% | busy 25% | jam 15%  ← 날씨와 독립

COMMON_TOTAL_TRAFFIC=55; COMMON_LOCAL_TRAFFIC=14
ADVERSE_RADIUS=70.0; MIN_DIST=35.0
HOST=127.0.0.1; PORT=2000; TM_PORT=8000; FPS=10; STEP=5.0
SEQS_PER_TOWN=30       # 원본과 동일 (2x 스케일 시 TARGET만 변경)
VAL_SEQS_PER_TOWN=85   # 원본과 동일
RESTART_EVERY=120

wait_for_carla() {
  for ((i=0; i<40; i++)); do
    if nc -z localhost 2000; then echo "✅ Carla OK"; return 0; fi
    echo "⏳ (${i}/40)"; sleep 1
  done; return 1
}
cleanup() {
  fuser -k 2000/tcp 2>/dev/null || true
  for p in {8000..8010}; do fuser -k "$p"/tcp 2>/dev/null || true; done
  sleep 2
}
start_carla() {
  "$CARLA_PATH/CarlaUE4.sh" -prefernvidia -RenderOffScreen -nosound -quality-level=Epic &
  CARLA_PID=$!
  wait_for_carla || { kill -9 "$CARLA_PID" 2>/dev/null || true; return 1; }
}
restart_carla() { kill -9 "$CARLA_PID" 2>/dev/null || true; sleep 2; start_carla; }
count_seq() { local d="$1"; if [[ ! -d "$d" ]]; then echo 0; return; fi; find "$d" -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9]' 2>/dev/null | wc -l | tr -d ' '; }
get_max_seg() {
  local r="$1"; if [[ ! -d "$r" ]]; then echo 0; return; fi
  find "$r" -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9]' -printf "%f\n" 2>/dev/null \
    | sed 's/^0*//' | awk '{print ($0==""?0:$0)}' | sort -n | tail -n 1
}
rand_pick() { local -n a=$1; echo "${a[$((RANDOM % ${#a[@]}))]}"; }
all_done() { (( $(count_seq "$OUTROOT/train") >= TARGET_TRAIN_SEQ && $(count_seq "$OUTROOT/val") >= TARGET_VAL_SEQ )); }

cleanup; start_carla

MAX_SEG=$(( $(get_max_seg "$OUTROOT/train") > $(get_max_seg "$OUTROOT/val") ? $(get_max_seg "$OUTROOT/train") : $(get_max_seg "$OUTROOT/val") ))
BASE_OFFSET=$MAX_SEG
OFFSET_COUNTER=0
SEG_RUNS=0

TRAIN_TOWN_IDX=0
TRAIN_TOWN_COUNT=0
CURRENT_TRAIN_TOWN="${TRAIN_TOWNS[0]}"
VAL_TOWN_IDX=0
VAL_TOWN_COUNT=0
CURRENT_VAL_TOWN="${VAL_TOWNS[0]}"
CURRENT_MAP="${TRAIN_TOWNS[0]}"

echo "========================================="
echo "🎯 SubsetA (Argoverse2 7-cam)"
echo "   train=$TARGET_TRAIN_SEQ / val=$TARGET_VAL_SEQ seqs, sample=$SAMPLE_PER_SEQ"
echo "   weather: clear 6% / wet 13% / rain 21% / fog★ 30% / night 30% (HardRainyNight★5%)"
echo "   traffic: sparse 10% / normal 50% / busy 25% / jam 15%  (독립 샘플링)"
echo "🔁 RESUME from seg $((MAX_SEG+1))"
echo "========================================="

python "$CARLA_PATH/PythonAPI/util/config.py" --map "$CURRENT_TRAIN_TOWN"
sleep 2
echo "🗺️ Initial map: $CURRENT_TRAIN_TOWN"

while ! all_done; do
  TR=$(count_seq "$OUTROOT/train"); VA=$(count_seq "$OUTROOT/val")

  if (( TR < TARGET_TRAIN_SEQ )); then SPLIT="train"
  elif (( VA < TARGET_VAL_SEQ )); then SPLIT="val"
  else break; fi

  if [[ "$SPLIT" == "train" ]]; then
    if (( TRAIN_TOWN_COUNT >= SEQS_PER_TOWN )); then
      TRAIN_TOWN_IDX=$(( (TRAIN_TOWN_IDX + 1) % ${#TRAIN_TOWNS[@]} ))
      CURRENT_TRAIN_TOWN="${TRAIN_TOWNS[$TRAIN_TOWN_IDX]}"
      TRAIN_TOWN_COUNT=0
    fi
    TOWN="$CURRENT_TRAIN_TOWN"
    TRAIN_TOWN_COUNT=$((TRAIN_TOWN_COUNT + 1))
  else
    if (( VAL_TOWN_COUNT >= VAL_SEQS_PER_TOWN )); then
      VAL_TOWN_IDX=$(( (VAL_TOWN_IDX + 1) % ${#VAL_TOWNS[@]} ))
      CURRENT_VAL_TOWN="${VAL_TOWNS[$VAL_TOWN_IDX]}"
      VAL_TOWN_COUNT=0
    fi
    TOWN="$CURRENT_VAL_TOWN"
    VAL_TOWN_COUNT=$((VAL_TOWN_COUNT + 1))
  fi

  if [[ "$CURRENT_MAP" != "$TOWN" ]]; then
    echo "🗺️ Map change → $TOWN"
    python "$CARLA_PATH/PythonAPI/util/config.py" --map "$TOWN"
    sleep 2
    CURRENT_MAP="$TOWN"
  fi

  mkdir -p "$OUTROOT/$SPLIT"
  offset=$((BASE_OFFSET + OFFSET_COUNTER))
  OFFSET_COUNTER=$((OFFSET_COUNTER + 1))

  RAND=$((RANDOM % 100))

  # 날씨 선택: WEATHER/TAG 세팅 후 python 한 번 호출
  if   (( RAND <  3 )); then WEATHER="ClearNoon";        TAG="clearnoon";     echo "☀️ [CLEAR_NOON][$SPLIT] $TOWN"
  elif (( RAND <  6 )); then WEATHER="ClearSunset";      TAG="clearsunset";   echo "☀️ [CLEAR_SUNSET][$SPLIT] $TOWN"
  elif (( RAND <  9 )); then WEATHER="WetNoon";          TAG="wetnoon";       echo "💧 [WET_NOON][$SPLIT] $TOWN"
  elif (( RAND < 12 )); then WEATHER="WetCloudyNoon";    TAG="wetcloudynoon"; echo "💧 [WET_CLOUDY_NOON][$SPLIT] $TOWN"
  elif (( RAND < 16 )); then WEATHER="WetSunset";        TAG="wetsunset";     echo "💧 [WET_SUNSET][$SPLIT] $TOWN"
  elif (( RAND < 19 )); then WEATHER="WetCloudySunset";  TAG="wetcloudyset";  echo "💧 [WET_CLOUDY_SUNSET][$SPLIT] $TOWN"
  elif (( RAND < 24 )); then WEATHER="SoftRainNoon";     TAG="softrainnoon";  echo "🌧️ [SOFT_RAIN_NOON][$SPLIT] $TOWN"
  elif (( RAND < 28 )); then WEATHER="SoftRainSunset";   TAG="softrainsun";   echo "🌧️ [SOFT_RAIN_SUNSET][$SPLIT] $TOWN"
  elif (( RAND < 32 )); then WEATHER="MidRainyNoon";     TAG="midrainynoon";  echo "🌧️ [MID_RAIN_NOON][$SPLIT] $TOWN"
  elif (( RAND < 35 )); then WEATHER="MidRainSunset";    TAG="midrainsun";    echo "🌧️ [MID_RAIN_SUNSET][$SPLIT] $TOWN"
  elif (( RAND < 38 )); then WEATHER="HardRainNoon";     TAG="hardrainnoon";  echo "⛈️ [HARD_RAIN_NOON][$SPLIT] $TOWN"
  elif (( RAND < 40 )); then WEATHER="HardRainSunset";   TAG="hardrainsun";   echo "⛈️ [HARD_RAIN_SUNSET][$SPLIT] $TOWN"
  elif (( RAND < 53 )); then WEATHER="fog_light";        TAG="foglight";      echo "🌫️ [FOG_LIGHT][$SPLIT] $TOWN"
  elif (( RAND < 64 )); then WEATHER="fog_mid";          TAG="fogmid";        echo "🌫️ [FOG_MID][$SPLIT] $TOWN"
  elif (( RAND < 70 )); then WEATHER="fog_heavy";        TAG="fogheavy";      echo "🌫️ [FOG_HEAVY][$SPLIT] $TOWN"
  elif (( RAND < 81 )); then WEATHER="ClearNight";       TAG="nightclear";    echo "🌙 [NIGHT_CLEAR][$SPLIT] $TOWN"
  elif (( RAND < 91 )); then WEATHER="CloudyNight";      TAG="nightcloudy";   echo "🌙 [NIGHT_CLOUDY][$SPLIT] $TOWN"
  elif (( RAND < 95 )); then WEATHER="MidRainyNight";    TAG="nightrain";     echo "🌧️ [NIGHT_RAIN][$SPLIT] $TOWN"
  else                        WEATHER="HardRainyNight";  TAG="nighthardrain"; echo "⛈️ [HARD_RAIN_NIGHT][$SPLIT] $TOWN"
  fi

  # 교통 밀도 (날씨와 독립): sparse 10% / normal 50% / busy 25% / jam 15%
  TL_RAND=$((RANDOM % 100))
  if   (( TL_RAND < 10 )); then TT=20;  LT=5;  echo "🚗 traffic=sparse (TT=$TT LT=$LT)"
  elif (( TL_RAND < 60 )); then TT=55;  LT=14; echo "🚗 traffic=normal (TT=$TT LT=$LT)"
  elif (( TL_RAND < 85 )); then TT=85;  LT=25; echo "🚗 traffic=busy   (TT=$TT LT=$LT)"
  else                           TT=110; LT=38; echo "🚗 traffic=jam    (TT=$TT LT=$LT)"
  fi

  LOG="$LOGDIR/A_${TAG}_${TOWN}_${SPLIT}_off${offset}.log"
  set +e
  python -u "$CAPTURE_PY" --subset "$SUBSET" --dir "$OUTROOT/$SPLIT" \
    --spawn-offset "$offset" --weather "$WEATHER" \
    --total-traffic "$TT" --local-traffic "$LT" \
    --radius "$ADVERSE_RADIUS" --min-dist "$MIN_DIST" --sample "$SAMPLE_PER_SEQ" \
    --fps "$FPS" --host "$HOST" --port "$PORT" --tm-port "$TM_PORT" \
    --step "$STEP" --pose-format openlanev2 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}; set -e

  if (( RC != 0 )); then
    echo "❌ rc=$RC, 재시작..."
    restart_carla || true
    python "$CARLA_PATH/PythonAPI/util/config.py" --map "$CURRENT_MAP"
    sleep 3
    continue
  fi

  SEG_RUNS=$((SEG_RUNS + 1))
  if (( SEG_RUNS % RESTART_EVERY == 0 )); then
    restart_carla || true
    python "$CARLA_PATH/PythonAPI/util/config.py" --map "$CURRENT_MAP"
    sleep 3
  fi

  echo "📊 train=$(count_seq "$OUTROOT/train")/$TARGET_TRAIN_SEQ val=$(count_seq "$OUTROOT/val")/$TARGET_VAL_SEQ"
done

echo "✅ SubsetA 완료! train=$(count_seq "$OUTROOT/train") val=$(count_seq "$OUTROOT/val")"
kill "$CARLA_PID" 2>/dev/null || true
wait "$CARLA_PID" 2>/dev/null || true

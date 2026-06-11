#!/bin/zsh
# Оркестратор v4: ждёт конца генерации -> split -> train -> calibrate -> evaluate.
# Каждый шаг логируется в /tmp/v4_log.txt; полный вывод в /tmp/v4_pipeline.log.
# v3-артефакты НЕ трогаются. Останавливается на первой ошибке (пишет STAGE_FAILED).
cd "$(dirname "$0")/.."          # корень репозитория (скрипт лежит в training/)
source venv/bin/activate
PLOG=/tmp/v4_pipeline.log
SLOG=/tmp/v4_log.txt
: > "$PLOG"
log(){ echo "[$(date +%H:%M)] $1" | tee -a "$SLOG"; echo "[$(date +%H:%M)] $1" >> "$PLOG"; }

log "ORCH: ожидаю окончания генерации (gen_ollama_v4)..."
WAITED=0
while pgrep -f "gen_ollama_v4" > /dev/null; do
  sleep 30; WAITED=$((WAITED+30))
  if [ $WAITED -ge 2700 ]; then   # 45 мин страховка
    log "ORCH: генерация >45мин — убиваю и иду дальше с тем, что есть"
    pkill -f "gen_ollama_v4"; sleep 3; break
  fi
done
LL=$(wc -l < data/llama3_raw.jsonl 2>/dev/null | tr -d ' '); LL=${LL:-0}
MI=$(wc -l < data/mistral_raw.jsonl 2>/dev/null | tr -d ' '); MI=${MI:-0}
log "ORCH: генерация завершена. llama=$LL mistral=$MI (сумма $((LL+MI)))"

log "STAGE split_v4"
python training/split_v4.py >> "$PLOG" 2>&1 || { log "STAGE_FAILED split_v4"; exit 1; }
TOTAL=$((LL+MI))
if [ $TOTAL -lt 200 ]; then
  log "ORCH: новых ИИ-текстов <200 ($TOTAL) — НЕ обучаю v4 (фаза 2.1). STOP."
  log "INSUFFICIENT_DATA"
  exit 2
fi

log "STAGE train_v4 (MPS, max 4 эпохи / 90 мин)"
python training/train_v4.py >> "$PLOG" 2>&1 || { log "STAGE_FAILED train_v4 (см. pipeline.log; пробую калибровать последний чекпойнт)"; }
if [ ! -d models/v4_model ]; then log "ORCH: models/v4_model не создан — STOP"; log "TRAIN_FAILED_NO_CKPT"; exit 3; fi

log "STAGE calibrate_v4"
python training/calibrate_v4.py >> "$PLOG" 2>&1 || { log "STAGE_FAILED calibrate_v4"; exit 4; }

log "STAGE evaluate_v4"
python training/evaluate_v4.py >> "$PLOG" 2>&1 || { log "STAGE_FAILED evaluate_v4"; exit 5; }

log "ORCH: ПАЙПЛАЙН ЗАВЕРШЁН. Итоги — в /tmp/v4_pipeline.log и data/v4_eval.json"
log "PIPELINE_DONE"

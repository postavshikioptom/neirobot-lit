#!/bin/bash

# Задача 219: Automatic Process Watchdog
# Скрипт мониторинга живучести бота с exponential backoff и лимитом рестартов

# Конфигурация
SYMBOL="${1:-BTCUSDT}"
BOT_BINARY="${BOT_BINARY:-./target/release/run-bot}"
BOT_CONFIG="${BOT_CONFIG:-./bots/$SYMBOL/config.toml}"
HEARTBEAT_FILE="./bots/$SYMBOL/state/liveness.heartbeat"
HEARTBEAT_TIMEOUT_SECS=60  # Максимальный возраст heartbeat перед перезапуском
MAX_RESTARTS_PER_HOUR=10
RESTART_WINDOW_SECS=3600
INITIAL_BACKOFF_SECS=5
MAX_BACKOFF_SECS=60
HEARTBEAT_CHECK_INTERVAL_SECS=10  # Проверяем heartbeat каждые 10 секунд

# Переменные состояния
restart_count=0
restart_timestamps=()
backoff_secs=$INITIAL_BACKOFF_SECS
should_exit=false
bot_pid=""

# Обработчик сигналов для корректного завершения
trap 'should_exit=true; if [ -n "$bot_pid" ] && kill -0 $bot_pid 2>/dev/null; then kill -TERM $bot_pid 2>/dev/null; fi' SIGTERM SIGINT

log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $1" >&2
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

log_warn() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARN] $1" >&2
}

# Проверить возраст heartbeat файла
check_heartbeat() {
    if [ ! -f "$HEARTBEAT_FILE" ]; then
        return 1  # Файл не существует
    fi
    
    local heartbeat_ts=$(cat "$HEARTBEAT_FILE" 2>/dev/null || echo "0")
    local current_ts=$(date +%s)
    local age=$((current_ts - heartbeat_ts))
    
    if [ $age -gt $HEARTBEAT_TIMEOUT_SECS ]; then
        return 1  # Heartbeat слишком старый
    fi
    
    return 0  # Heartbeat свежий
}

# Очистить старые timestamps рестартов (старше 1 часа)
cleanup_old_restarts() {
    local current_ts=$(date +%s)
    local new_timestamps=()
    
    for ts in "${restart_timestamps[@]}"; do
        local age=$((current_ts - ts))
        if [ $age -lt $RESTART_WINDOW_SECS ]; then
            new_timestamps+=("$ts")
        fi
    done
    
    restart_timestamps=("${new_timestamps[@]}")
    restart_count=${#restart_timestamps[@]}
}

# Проверить лимит рестартов
check_restart_limit() {
    cleanup_old_restarts
    
    if [ $restart_count -ge $MAX_RESTARTS_PER_HOUR ]; then
        log_error "CRITICAL: Bot crashed $restart_count times in the last hour. Stopping watchdog."
        log_error "Max allowed restarts per hour: $MAX_RESTARTS_PER_HOUR"
        return 1
    fi
    
    return 0
}

# Вычислить exponential backoff
calculate_backoff() {
    backoff_secs=$((backoff_secs * 2))
    if [ $backoff_secs -gt $MAX_BACKOFF_SECS ]; then
        backoff_secs=$MAX_BACKOFF_SECS
    fi
}

# Сбросить backoff при успешном запуске
reset_backoff() {
    backoff_secs=$INITIAL_BACKOFF_SECS
}

log_info "Starting watchdog for $SYMBOL"
log_info "Bot binary: $BOT_BINARY"
log_info "Bot config: $BOT_CONFIG"
log_info "Heartbeat file: $HEARTBEAT_FILE"
log_info "Heartbeat timeout: $HEARTBEAT_TIMEOUT_SECS seconds"
log_info "Max restarts per hour: $MAX_RESTARTS_PER_HOUR"

# Основной цикл мониторинга
while [ "$should_exit" = false ]; do
    # Проверяем лимит рестартов
    if ! check_restart_limit; then
        log_error "Restart limit exceeded. Exiting watchdog."
        exit 1
    fi
    
    log_info "Starting bot process (restart #$((restart_count + 1)))"
    
    # Запускаем бот в фоновом режиме
    "$BOT_BINARY" --config "$BOT_CONFIG" --symbol "$SYMBOL" &
    bot_pid=$!
    
    log_info "Bot started with PID $bot_pid"
    
    # Флаг для отслеживания причины завершения
    deadlock_detected=false
    
    # Мониторим heartbeat пока бот работает
    while kill -0 $bot_pid 2>/dev/null; do
        sleep $HEARTBEAT_CHECK_INTERVAL_SECS
        
        if [ "$should_exit" = true ]; then
            log_info "Received shutdown signal, terminating bot PID $bot_pid"
            kill -TERM $bot_pid 2>/dev/null
            break
        fi
        
        # Проверяем heartbeat
        if ! check_heartbeat; then
            log_error "Deadlock detected via heartbeat! Killing bot PID $bot_pid"
            kill -9 $bot_pid 2>/dev/null
            deadlock_detected=true
            break
        fi
    done
    
    # Ждем завершения процесса если он еще работает
    wait $bot_pid 2>/dev/null
    local exit_code=$?
    
    if [ "$should_exit" = true ]; then
        log_info "Watchdog received shutdown signal, exiting gracefully"
        exit 0
    fi
    
    # Анализируем причину завершения
    if [ "$deadlock_detected" = true ]; then
        log_error "Bot was killed due to deadlock (heartbeat stale)"
    else
        log_error "Bot crashed with exit code $exit_code"
    fi
    
    # Регистрируем рестарт
    restart_timestamps+=("$(date +%s)")
    restart_count=$((restart_count + 1))
    
    # Проверяем лимит после регистрации рестарта
    if ! check_restart_limit; then
        log_error "Restart limit exceeded after crash. Exiting watchdog."
        exit 1
    fi
    
    # Exponential backoff перед следующей попыткой
    calculate_backoff
    log_info "Waiting ${backoff_secs}s before restart (attempt $restart_count/$MAX_RESTARTS_PER_HOUR in last hour)"
    
    # Ждем с проверкой сигналов завершения
    for ((i = 0; i < backoff_secs; i++)); do
        if [ "$should_exit" = true ]; then
            log_info "Received shutdown signal during backoff, exiting watchdog"
            exit 0
        fi
        sleep 1
    done
done

log_info "Watchdog received shutdown signal, exiting gracefully"
exit 0

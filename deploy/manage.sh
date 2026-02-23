#!/usr/bin/env bash
# Neirobot LiT Deployment Manager
# Automated bot deployment script with native and Docker support

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODE="${MODE:-native}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="${PROJECT_ROOT}/deploy"
BOTS_DIR="${PROJECT_ROOT}/bots"
MODELS_DIR="${PROJECT_ROOT}/bots"  # Модели хранятся в ./bots/SYMBOL/model/archive/

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND [ARGS]

Neirobot LiT Deployment Manager

OPTIONS:
    --mode MODE     Deployment mode: native or docker (default: native)
    -h, --help      Show this help message

COMMANDS:
    build           Build the project with musl target
    deploy SYMBOL   Deploy a bot for the specified trading symbol
    deploy-model SYMBOL --version TAG  Deploy specific model version from registry
    rollback-model SYMBOL              Rollback to previous model version
    status          Show status of all deployed bots
    logs SYMBOL     Show logs for the specified bot
    stop SYMBOL     Stop the specified bot
    restart SYMBOL  Restart the specified bot
    enable SYMBOL   Enable autostart for the specified bot (systemd only)
    reload SYMBOL   Reload configuration without restart (graceful SIGHUP)
    update-config SYMBOL CONFIG_PATH  Update config atomically and reload
    top             Monitor resource usage of all bots (systemd-cgtop)
    cleanup         Clean up old logs (journalctl and local logs)

EXAMPLES:
    $0 --mode native build
    $0 --mode native deploy BTCUSDT
    $0 --mode docker deploy ETHUSDT
    $0 deploy-model BTCUSDT --version v2.0
    $0 rollback-model BTCUSDT
    $0 status
    $0 logs BTCUSDT
    $0 enable ETHUSDT
    $0 reload BTCUSDT
    $0 update-config BTCUSDT /path/to/new/config.toml
    $0 top

EOF
    exit 0
}

# Check dependencies
check_dependencies() {
    local deps=("$@")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "Required dependency '$dep' not found"
            exit 1
        fi
    done
}

# Build command
cmd_build() {
    log_info "Building neirobot-lit with musl target..."
    
    check_dependencies cargo
    
    cd "$PROJECT_ROOT"
    
    # Check if musl target is installed
    if ! rustup target list | grep -q "x86_64-unknown-linux-musl (installed)"; then
        log_info "Installing x86_64-unknown-linux-musl target..."
        rustup target add x86_64-unknown-linux-musl
    fi
    
    # Try cargo-zigbuild first (better cross-compilation)
    if command -v cargo-zigbuild &> /dev/null; then
        log_info "Using cargo-zigbuild for cross-compilation..."
        cargo zigbuild --release --target x86_64-unknown-linux-musl
    else
        log_info "Using cargo build (install cargo-zigbuild for better cross-compilation)..."
        cargo build --release --target x86_64-unknown-linux-musl
    fi
    
    local binary_path="target/x86_64-unknown-linux-musl/release/neirobot-lit"
    if [[ -f "$binary_path" ]]; then
        log_info "Build successful: $binary_path"
        log_info "Binary size: $(du -h "$binary_path" | cut -f1)"
    else
        log_error "Build failed: binary not found"
        exit 1
    fi
}

# Validate model existence and metadata
validate_model() {
    local symbol="$1"
    local model_path="${BOTS_DIR}/${symbol}/model/model.onnx"
    local metadata_path="${BOTS_DIR}/${symbol}/model/metadata.json"
    
    if [[ ! -f "$model_path" ]]; then
        log_error "Model not found: $model_path"
        log_info "Please deploy the model first using deploy-model command"
        return 1
    fi
    
    # Проверка метаданных
    if [[ ! -f "$metadata_path" ]]; then
        log_error "Metadata not found: $metadata_path"
        log_info "Please ensure metadata.json exists in model directory"
        return 1
    fi
    
    log_info "Model found: $model_path ($(du -h "$model_path" | cut -f1))"
    log_info "Metadata verified: $metadata_path"
    return 0
}

# Задача 184: Атомарное обновление конфигурации с бэкапом
atomic_update_config() {
    local symbol="$1"
    local new_config_path="$2"
    
    local bot_dir="${BOTS_DIR}/${symbol}"
    local config_path="${bot_dir}/config.toml"
    local config_history_dir="${bot_dir}/config_history"
    
    # Создаем директорию для истории конфигов
    mkdir -p "$config_history_dir"
    
    # Создаем резервную копию текущей конфигурации
    if [[ -f "$config_path" ]]; then
        local timestamp=$(date +%Y%m%d_%H%M%S)
        local backup_path="${config_history_dir}/config.toml.${timestamp}.bak"
        
        cp "$config_path" "$backup_path"
        chmod 600 "$backup_path"
        log_info "Config backup created: $backup_path"
    fi
    
    # Атомарное обновление: копируем в .tmp, затем переименовываем
    cp "$new_config_path" "${config_path}.tmp"
    chmod 600 "${config_path}.tmp"
    mv "${config_path}.tmp" "$config_path"
    
    log_info "Config updated atomically: $config_path"
}

# Задача 185: Деплой конкретной версии модели из реестра
cmd_deploy_model() {
    local symbol="$1"
    local version_tag=""
    
    # Парсим аргументы
    shift
    while [[ $# -gt 0 ]]; do
        case $1 in
            --version)
                version_tag="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: $0 deploy-model SYMBOL --version TAG"
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$symbol" || -z "$version_tag" ]]; then
        log_error "Symbol and version tag are required"
        echo "Usage: $0 deploy-model SYMBOL --version TAG"
        exit 1
    fi
    
    local bot_dir="${BOTS_DIR}/${symbol}"
    local model_dir="${bot_dir}/model"
    local registry_path="${model_dir}/registry.json"
    local metadata_path="${model_dir}/metadata.json"
    
    # Проверяем существование реестра
    if [[ ! -f "$registry_path" ]]; then
        log_error "Model registry not found: $registry_path"
        log_info "Please create registry.json with model versions"
        exit 1
    fi
    
    log_info "Deploying model version $version_tag for $symbol..."
    
    # Проверяем наличие jq для работы с JSON
    if ! command -v jq &> /dev/null; then
        log_error "jq not found. Please install jq package"
        exit 1
    fi
    
    # Ищем версию в реестре
    local entry=$(jq -r ".entries[] | select(.version_tag == \"$version_tag\")" "$registry_path")
    
    if [[ -z "$entry" ]]; then
        log_error "Version $version_tag not found in registry"
        log_info "Available versions:"
        jq -r '.entries[] | "  - \(.version_tag) (MCC: \(.mcc_score), created: \(.created_at))"' "$registry_path"
        exit 1
    fi
    
    # Извлекаем данные из реестра
    local file_path=$(echo "$entry" | jq -r '.file_path')
    local onnx_hash=$(echo "$entry" | jq -r '.onnx_hash')
    local mcc_score=$(echo "$entry" | jq -r '.mcc_score')
    local created_at=$(echo "$entry" | jq -r '.created_at')
    
    log_info "Found model: $file_path"
    log_info "  Hash: $onnx_hash"
    log_info "  MCC Score: $mcc_score"
    log_info "  Created: $created_at"
    
    # Проверяем существование файла модели
    if [[ ! -f "$file_path" ]]; then
        log_error "Model file not found: $file_path"
        exit 1
    fi
    
    # Создаем резервную копию текущей модели
    if [[ -f "${model_dir}/model.onnx" ]]; then
        local timestamp=$(date +%Y%m%d_%H%M%S)
        local backup_path="${model_dir}/model.onnx.${timestamp}.bak"
        
        cp "${model_dir}/model.onnx" "$backup_path"
        log_info "Current model backed up: $backup_path"
    fi
    
    # Копируем новую модель
    log_info "Copying model file..."
    cp "$file_path" "${model_dir}/model.onnx"
    
    # Обновляем metadata.json
    log_info "Updating metadata.json..."
    
    # Создаем резервную копию metadata
    if [[ -f "$metadata_path" ]]; then
        local timestamp=$(date +%Y%m%d_%H%M%S)
        cp "$metadata_path" "${metadata_path}.${timestamp}.bak"
    fi
    
    # Загружаем нормализацию из старого metadata (если есть)
    local normalization="{}"
    if [[ -f "$metadata_path" ]]; then
        normalization=$(jq -r '.normalization' "$metadata_path" 2>/dev/null || echo "{}")
    fi
    
    # Создаем новый metadata.json
    cat > "$metadata_path" << EOF
{
  "onnx_hash": "$onnx_hash",
  "version": "$version_tag",
  "mcc_score": $mcc_score,
  "normalization": $normalization
}
EOF
    
    log_info "Metadata updated successfully"
    log_info "Model version $version_tag deployed successfully!"
    log_info "Restart the bot to apply changes: $0 restart $symbol"
}

# Задача 185: Откат к предыдущей версии модели
cmd_rollback_model() {
    local symbol="$1"
    
    if [[ -z "$symbol" ]]; then
        log_error "Symbol is required"
        echo "Usage: $0 rollback-model SYMBOL"
        exit 1
    fi
    
    local bot_dir="${BOTS_DIR}/${symbol}"
    local model_dir="${bot_dir}/model"
    local registry_path="${model_dir}/registry.json"
    local metadata_path="${model_dir}/metadata.json"
    
    # Проверяем существование реестра и metadata
    if [[ ! -f "$registry_path" ]]; then
        log_error "Model registry not found: $registry_path"
        exit 1
    fi
    
    if [[ ! -f "$metadata_path" ]]; then
        log_error "Metadata not found: $metadata_path"
        exit 1
    fi
    
    log_info "Rolling back model for $symbol..."
    
    # Проверяем наличие jq
    if ! command -v jq &> /dev/null; then
        log_error "jq not found. Please install jq package"
        exit 1
    fi
    
    # Получаем текущую версию
    local current_version=$(jq -r '.version' "$metadata_path")
    log_info "Current version: $current_version"
    
    # Получаем все версии из реестра, отсортированные по дате создания
    local versions=$(jq -r '.entries | sort_by(.created_at) | reverse | .[].version_tag' "$registry_path")
    
    # Находим предыдущую версию
    local found_current=false
    local previous_version=""
    
    while IFS= read -r version; do
        if [[ "$found_current" == true ]]; then
            previous_version="$version"
            break
        fi
        
        if [[ "$version" == "$current_version" ]]; then
            found_current=true
        fi
    done <<< "$versions"
    
    if [[ -z "$previous_version" ]]; then
        log_error "No previous version found in registry"
        log_info "Available versions:"
        jq -r '.entries[] | "  - \(.version_tag) (MCC: \(.mcc_score), created: \(.created_at))"' "$registry_path"
        exit 1
    fi
    
    log_info "Rolling back to version: $previous_version"
    
    # Используем cmd_deploy_model для деплоя предыдущей версии
    cmd_deploy_model "$symbol" --version "$previous_version"
}

# Validate model existence and metadata
cmd_deploy() {
    local symbol="$1"
    
    if [[ -z "$symbol" ]]; then
        log_error "Symbol is required"
        echo "Usage: $0 deploy SYMBOL"
        exit 1
    fi
    
    log_info "Deploying bot for $symbol in $MODE mode..."
    
    # Validate model
    if ! validate_model "$symbol"; then
        exit 1
    fi
    
    # Create bot directory structure
    local bot_dir="${BOTS_DIR}/${symbol}"
    log_info "Creating directory structure: $bot_dir"
    mkdir -p "$bot_dir"/{model/archive,data/raw,logs,config_history}
    
    # Copy model (если существует в python_lab)
    if [[ -f "${PROJECT_ROOT}/python_lab/models/${symbol}.onnx" ]]; then
        log_info "Copying model from python_lab..."
        cp "${PROJECT_ROOT}/python_lab/models/${symbol}.onnx" "${bot_dir}/model/model.onnx"
    else
        log_warn "Model not found in python_lab/models, skipping model copy"
        log_info "Use deploy-model command to deploy model from registry"
    fi
    
    # Generate config from template
    log_info "Generating configuration..."
    export SYMBOL="$symbol"
    export THRESHOLD_UP="${THRESHOLD_UP:-0.7}"
    export THRESHOLD_DOWN="${THRESHOLD_DOWN:-0.7}"
    export MAX_POSITION_SIZE="${MAX_POSITION_SIZE:-100.0}"
    export MAX_DRAWDOWN="${MAX_DRAWDOWN:-0.05}"
    export EXECUTION_PROVIDER="${EXECUTION_PROVIDER:-cpu}"
    export DEVICE_ID="${DEVICE_ID:-0}"
    export INTRA_THREADS="${INTRA_THREADS:-4}"
    
    # Установка пути к модели в зависимости от режима
    if [[ "$MODE" == "docker" ]]; then
        export MODEL_BASE="/app"
    else
        export MODEL_BASE="/opt/neirobot"
    fi
    
    if ! command -v envsubst &> /dev/null; then
        log_error "envsubst not found. Please install gettext package"
        exit 1
    fi
    
    envsubst < "${DEPLOY_DIR}/bot.template.toml" > "${bot_dir}/config.toml"
    log_info "Configuration generated: ${bot_dir}/config.toml"
    
    # Замена пути к модели в конфиге для Docker режима
    if [[ "$MODE" == "docker" ]]; then
        sed -i "s|/opt/neirobot/bots/|/app/bots/|g" "${bot_dir}/config.toml"
        log_info "Config paths adjusted for Docker mode"
    fi
    
    # Deploy based on mode
    case "$MODE" in
        native)
            deploy_native "$symbol"
            ;;
        docker)
            deploy_docker "$symbol"
            ;;
        *)
            log_error "Unknown mode: $MODE"
            exit 1
            ;;
    esac
    
    log_info "Deployment completed successfully!"
}

# Deploy in native mode (systemd)
deploy_native() {
    local symbol="$1"
    
    check_dependencies systemctl
    
    log_info "Deploying in native mode using systemd..."
    
    # Check if running as root or with sudo
    if [[ $EUID -ne 0 ]]; then
        log_warn "Native deployment requires root privileges"
        log_info "Please run with sudo or as root"
        exit 1
    fi
    
    # Determine target installation path
    local install_path="/opt/neirobot"
    
    # Check if binary exists
    local binary_path="${PROJECT_ROOT}/target/x86_64-unknown-linux-musl/release/neirobot-lit"
    if [[ ! -f "$binary_path" ]]; then
        # Try alternative binary name (legacy)
        binary_path="${PROJECT_ROOT}/target/x86_64-unknown-linux-musl/release/run-bot"
        if [[ ! -f "$binary_path" ]]; then
            log_error "Binary not found. Please run 'build' command first"
            exit 1
        fi
    fi
    
    # Create neirobot user if not exists
    if ! id -u neirobot &>/dev/null; then
        log_info "Creating neirobot user..."
        useradd -r -s /bin/false -d "$install_path" neirobot
    fi
    
    # Copy project to /opt/neirobot if not already there
    if [[ "$PROJECT_ROOT" != "$install_path" ]]; then
        log_info "Installing project to $install_path..."
        mkdir -p "$install_path"
        
        # Copy necessary files
        mkdir -p "${install_path}/target/x86_64-unknown-linux-musl/release"
        cp -r "${PROJECT_ROOT}/target/x86_64-unknown-linux-musl" "${install_path}/target/" 2>/dev/null || true
        cp "${PROJECT_ROOT}/global.toml" "$install_path/" 2>/dev/null || true
        cp "${PROJECT_ROOT}/exchange.toml" "$install_path/" 2>/dev/null || true
        
        # Create bots directory
        mkdir -p "${install_path}/bots"
    fi
    
    # Create bot-specific directory structure
    log_info "Creating directory structure for $symbol..."
    mkdir -p "${install_path}/bots/${symbol}"/{model,data/raw,logs}
    
    # Copy model
    log_info "Copying model..."
    cp "${BOTS_DIR}/${symbol}/model/model.onnx" "${install_path}/bots/${symbol}/model/"
    
    # Copy configuration
    log_info "Copying configuration..."
    cp "${BOTS_DIR}/${symbol}/config.toml" "${install_path}/bots/${symbol}/"
    
    # Задача 184: Установка прав доступа на конфигурацию (chmod 600)
    chmod 600 "${install_path}/bots/${symbol}/config.toml"
    log_info "Config permissions set to 600 (read/write for owner only)"
    
    # Copy .env if exists
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        cp "${PROJECT_ROOT}/.env" "${install_path}/bots/${symbol}/"
    fi
    
    # Set ownership
    log_info "Setting ownership to neirobot:neirobot..."
    chown -R neirobot:neirobot "$install_path"
    
    # Make binary executable
    chmod +x "${install_path}/target/x86_64-unknown-linux-musl/release/"* 2>/dev/null || true
    
    # Copy systemd unit
    log_info "Installing systemd service template..."
    cp "${DEPLOY_DIR}/neirobot-lit@.service" /etc/systemd/system/
    
    # Reload systemd and start service
    log_info "Starting systemd service..."
    systemctl daemon-reload
    systemctl enable "neirobot-lit@${symbol}.service"
    systemctl start "neirobot-lit@${symbol}.service"
    
    # Check status
    if systemctl is-active --quiet "neirobot-lit@${symbol}.service"; then
        log_info "Service started successfully"
        systemctl status "neirobot-lit@${symbol}.service" --no-pager
    else
        log_error "Service failed to start"
        systemctl status "neirobot-lit@${symbol}.service" --no-pager
        exit 1
    fi
}

# Deploy in Docker mode
deploy_docker() {
    local symbol="$1"
    
    check_dependencies docker docker-compose
    
    log_info "Deploying in Docker mode..."
    
    # Build Docker image if not exists
    if ! docker images | grep -q "neirobot-lit"; then
        log_info "Building Docker image..."
        docker build -t neirobot-lit:latest -f "${PROJECT_ROOT}/Dockerfile" "$PROJECT_ROOT"
    fi
    
    # Export environment variables for docker-compose
    export SYMBOL="$symbol"
    export API_KEY="${API_KEY:-}"
    export API_SECRET="${API_SECRET:-}"
    export RUST_LOG="${RUST_LOG:-info}"
    
    # Start container
    log_info "Starting Docker container..."
    cd "$DEPLOY_DIR"
    docker-compose up -d
    
    # Check status
    if docker ps | grep -q "neirobot-lit-${symbol}"; then
        log_info "Container started successfully"
        docker ps | grep "neirobot-lit-${symbol}"
    else
        log_error "Container failed to start"
        docker-compose logs
        exit 1
    fi
}

# Status command
cmd_status() {
    log_info "Checking status of all bots..."
    echo ""
    printf "%-15s %-10s %-15s %-20s %-10s\n" "SYMBOL" "MODE" "PID/CONTAINER" "UPTIME" "STATUS"
    printf "%s\n" "--------------------------------------------------------------------------------"
    
    # Check native bots (systemd)
    if command -v systemctl &> /dev/null; then
        while IFS= read -r service; do
            if [[ -n "$service" ]]; then
                local symbol=$(echo "$service" | sed 's/neirobot-lit@\(.*\)\.service/\1/')
                local status=$(systemctl is-active "$service" 2>/dev/null || echo "inactive")
                local pid=$(systemctl show -p MainPID --value "$service" 2>/dev/null || echo "N/A")
                local uptime=$(systemctl show -p ActiveEnterTimestamp --value "$service" 2>/dev/null || echo "N/A")
                
                printf "%-15s %-10s %-15s %-20s %-10s\n" "$symbol" "native" "$pid" "$uptime" "$status"
            fi
        done < <(systemctl list-units --type=service --all | grep "neirobot-lit@" | awk '{print $1}')
    fi
    
    # Check Docker containers
    if command -v docker &> /dev/null; then
        while IFS= read -r line; do
            if [[ -n "$line" ]]; then
                local container_id=$(echo "$line" | awk '{print $1}')
                local container_name=$(echo "$line" | awk '{print $2}')
                local status=$(echo "$line" | awk '{print $3}')
                local uptime=$(echo "$line" | awk '{print $4" "$5}')
                
                # Extract symbol from container name
                local symbol=$(echo "$container_name" | sed 's/neirobot-lit-//')
                
                printf "%-15s %-10s %-15s %-20s %-10s\n" "$symbol" "docker" "$container_id" "$uptime" "$status"
            fi
        done < <(docker ps -a --filter "name=neirobot-lit-" --format "{{.ID}} {{.Names}} {{.Status}}" 2>/dev/null)
    fi
    
    echo ""
}

# Logs command
cmd_logs() {
    local symbol="$1"
    
    if [[ -z "$symbol" ]]; then
        log_error "Symbol is required"
        echo "Usage: $0 logs SYMBOL"
        exit 1
    fi
    
    log_info "Showing logs for $symbol..."
    
    # Try systemd first
    if systemctl list-units --type=service --all | grep -q "neirobot-lit@${symbol}.service"; then
        log_info "Using journalctl (native mode)"
        journalctl -u "neirobot-lit@${symbol}.service" -f
        return
    fi
    
    # Try Docker
    if docker ps -a | grep -q "neirobot-lit-${symbol}"; then
        log_info "Using docker logs (docker mode)"
        docker logs -f "neirobot-lit-${symbol}"
        return
    fi
    
    log_error "Bot not found: $symbol"
    exit 1
}

# Stop command
cmd_stop() {
    local symbol="$1"
    
    if [[ -z "$symbol" ]]; then
        log_error "Symbol is required"
        echo "Usage: $0 stop SYMBOL"
        exit 1
    fi
    
    log_info "Stopping bot for $symbol..."
    
    # Try systemd first
    if systemctl list-units --type=service --all | grep -q "neirobot-lit@${symbol}.service"; then
        log_info "Stopping systemd service..."
        systemctl stop "neirobot-lit@${symbol}.service"
        log_info "Service stopped"
        return
    fi
    
    # Try Docker
    if docker ps | grep -q "neirobot-lit-${symbol}"; then
        log_info "Stopping Docker container..."
        docker stop "neirobot-lit-${symbol}"
        log_info "Container stopped"
        return
    fi
    
    log_error "Bot not found: $symbol"
    exit 1
}

# Restart command
cmd_restart() {
    local symbol="$1"
    
    if [[ -z "$symbol" ]]; then
        log_error "Symbol is required"
        echo "Usage: $0 restart SYMBOL"
        exit 1
    fi
    
    log_info "Restarting bot for $symbol..."
    
    # Try systemd first
    if systemctl list-units --type=service --all | grep -q "neirobot-lit@${symbol}.service"; then
        log_info "Restarting systemd service..."
        systemctl restart "neirobot-lit@${symbol}.service"
        log_info "Service restarted"
        return
    fi
    
    # Try Docker
    if docker ps -a | grep -q "neirobot-lit-${symbol}"; then
        log_info "Restarting Docker container..."
        docker restart "neirobot-lit-${symbol}"
        log_info "Container restarted"
        return
    fi
    
    log_error "Bot not found: $symbol"
    exit 1
}

# Enable command (Задача 183: Автозапуск бота через systemd)
cmd_enable() {
    local symbol="$1"
    
    if [[ -z "$symbol" ]]; then
        log_error "Symbol is required"
        echo "Usage: $0 enable SYMBOL"
        exit 1
    fi
    
    log_info "Enabling autostart for $symbol..."
    
    # Check if running as root or with sudo
    if [[ $EUID -ne 0 ]]; then
        log_warn "Enable command requires root privileges"
        log_info "Please run with sudo or as root"
        exit 1
    fi
    
    # Check if service exists
    if ! systemctl list-unit-files | grep -q "neirobot-lit@.service"; then
        log_error "systemd service template not found"
        log_info "Please deploy the bot first using 'deploy' command"
        exit 1
    fi
    
    # Enable service
    systemctl enable "neirobot-lit@${symbol}.service"
    log_info "Autostart enabled for $symbol"
    log_info "Service will start automatically on system boot"
}

# Reload command (Задача 183: Graceful reload конфигурации)
cmd_reload() {
    local symbol="$1"
    
    if [[ -z "$symbol" ]]; then
        log_error "Symbol is required"
        echo "Usage: $0 reload SYMBOL"
        exit 1
    fi
    
    log_info "Reloading configuration for $symbol (graceful SIGHUP)..."
    
    # Try systemd first
    if systemctl list-units --type=service --all | grep -q "neirobot-lit@${symbol}.service"; then
        # Check if running as root or with sudo
        if [[ $EUID -ne 0 ]]; then
            log_warn "Reload command requires root privileges"
            log_info "Please run with sudo or as root"
            exit 1
        fi
        
        log_info "Sending SIGHUP to systemd service..."
        systemctl reload "neirobot-lit@${symbol}.service"
        log_info "Configuration reload signal sent"
        log_info "Bot will apply new settings without disconnecting from exchange"
        return
    fi
    
    # Try Docker
    if docker ps | grep -q "neirobot-lit-${symbol}"; then
        log_info "Sending SIGHUP to Docker container..."
        docker kill --signal=HUP "neirobot-lit-${symbol}"
        log_info "Configuration reload signal sent"
        return
    fi
    
    log_error "Bot not found: $symbol"
    exit 1
}

# Update config command (Задача 184: Атомарное обновление конфигурации)
cmd_update_config() {
    local symbol="$1"
    local new_config_path="$2"
    
    if [[ -z "$symbol" || -z "$new_config_path" ]]; then
        log_error "Symbol and config path are required"
        echo "Usage: $0 update-config SYMBOL /path/to/new/config.toml"
        exit 1
    fi
    
    if [[ ! -f "$new_config_path" ]]; then
        log_error "Config file not found: $new_config_path"
        exit 1
    fi
    
    log_info "Updating configuration for $symbol..."
    
    # Perform atomic update
    atomic_update_config "$symbol" "$new_config_path"
    
    # Send SIGHUP to reload
    log_info "Sending SIGHUP to reload configuration..."
    if systemctl list-units --type=service --all | grep -q "neirobot-lit@${symbol}.service"; then
        if [[ $EUID -ne 0 ]]; then
            log_warn "SIGHUP requires root privileges"
            log_info "Please run with sudo or as root"
            exit 1
        fi
        systemctl reload "neirobot-lit@${symbol}.service"
        log_info "Configuration updated and reload signal sent"
    elif docker ps | grep -q "neirobot-lit-${symbol}"; then
        docker kill --signal=HUP "neirobot-lit-${symbol}"
        log_info "Configuration updated and reload signal sent"
    else
        log_warn "Bot not running. Configuration updated but reload signal not sent."
        log_info "Start the bot to apply the new configuration"
    fi
}

# Top command (Задача 183: Мониторинг ресурсов через systemd-cgtop)
cmd_top() {
    log_info "Monitoring resource usage of all bots..."
    
    # Check if systemd-cgtop is available
    if ! command -v systemd-cgtop &> /dev/null; then
        log_error "systemd-cgtop not found"
        log_info "This command requires systemd-cgtop utility"
        exit 1
    fi
    
    # Check if running as root or with sudo
    if [[ $EUID -ne 0 ]]; then
        log_warn "Top command may require root privileges for full information"
        log_info "Run with sudo for complete resource monitoring"
    fi
    
    log_info "Press 'q' to quit, 'h' for help"
    echo ""
    
    # Run systemd-cgtop with filter for neirobot services
    # Note: systemd-cgtop doesn't support filtering, so we show all services
    # Users can identify neirobot services by the neirobot-lit@ prefix
    systemd-cgtop
}

# Cleanup command (Задача 182: Очистка логов)
cmd_cleanup() {
    log_info "Cleaning up old logs..."
    
    # Очистка journalctl (если используется systemd)
    if command -v journalctl &> /dev/null; then
        log_info "Cleaning journalctl logs older than 7 days..."
        journalctl --vacuum-time=7d
        log_info "journalctl cleanup completed"
    fi
    
    # Очистка локальных логов ботов
    if [[ -d "$BOTS_DIR" ]]; then
        log_info "Cleaning local bot logs..."
        find "$BOTS_DIR" -name "*.log" -type f -mtime +7 -delete
        log_info "Local log cleanup completed"
    fi
}

# Main command dispatcher
main() {
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Validate mode
    if [[ "$MODE" != "native" && "$MODE" != "docker" ]]; then
        log_error "Invalid mode: $MODE. Must be 'native' or 'docker'"
        exit 1
    fi
    
    # Get command
    local command="${1:-}"
    shift || true
    
    case "$command" in
        build)
            cmd_build
            ;;
        deploy)
            cmd_deploy "$@"
            ;;
        deploy-model)
            cmd_deploy_model "$@"
            ;;
        rollback-model)
            cmd_rollback_model "$@"
            ;;
        status)
            cmd_status
            ;;
        logs)
            cmd_logs "$@"
            ;;
        stop)
            cmd_stop "$@"
            ;;
        restart)
            cmd_restart "$@"
            ;;
        enable)
            cmd_enable "$@"
            ;;
        reload)
            cmd_reload "$@"
            ;;
        update-config)
            cmd_update_config "$@"
            ;;
        top)
            cmd_top
            ;;
        cleanup)
            cmd_cleanup
            ;;
        ""|help)
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            ;;
    esac
}

main "$@"

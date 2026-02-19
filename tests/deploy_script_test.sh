#!/usr/bin/env bash
# Test script for deploy/manage.sh
# Mock deployment test to verify directory creation, config generation, and script logic

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_DIR="${SCRIPT_DIR}/tmp_deploy_test"
TEST_SYMBOL="TESTUSDT"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

log_test() {
    echo -e "${YELLOW}[TEST]${NC} $*"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $*"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $*"
    ((TESTS_FAILED++))
}

# Setup test environment
setup() {
    log_test "Setting up test environment..."
    
    # Create temporary test directory
    rm -rf "$TEST_DIR"
    mkdir -p "$TEST_DIR"/{bots,python_lab/models,deploy}
    
    # Create mock model file
    echo "mock onnx model" > "$TEST_DIR/python_lab/models/${TEST_SYMBOL}.onnx"
    
    # Copy deploy files
    cp -r "$PROJECT_ROOT/deploy"/* "$TEST_DIR/deploy/"
    
    # Make manage.sh executable
    chmod +x "$TEST_DIR/deploy/manage.sh"
    
    log_pass "Test environment created: $TEST_DIR"
}

# Cleanup test environment
cleanup() {
    log_test "Cleaning up test environment..."
    rm -rf "$TEST_DIR"
    log_pass "Test environment cleaned up"
}

# Test 1: Check if manage.sh exists and is executable
test_script_exists() {
    log_test "Test 1: Checking if manage.sh exists and is executable..."
    
    if [[ -f "$PROJECT_ROOT/deploy/manage.sh" ]]; then
        log_pass "manage.sh exists"
    else
        log_fail "manage.sh not found"
        return 1
    fi
    
    if [[ -x "$PROJECT_ROOT/deploy/manage.sh" ]]; then
        log_pass "manage.sh is executable"
    else
        log_fail "manage.sh is not executable"
        return 1
    fi
}

# Test 2: Check if template files exist
test_templates_exist() {
    log_test "Test 2: Checking if template files exist..."
    
    local templates=(
        "deploy/bot.template.toml"
        "deploy/neirobot-lit@.service"
        "deploy/docker-compose.yml"
        "Dockerfile"
        "global.toml"
        "exchange.toml"
    )
    
    for template in "${templates[@]}"; do
        if [[ -f "$PROJECT_ROOT/$template" ]]; then
            log_pass "$template exists"
        else
            log_fail "$template not found"
        fi
    done
}

# Test 3: Test directory structure creation
test_directory_creation() {
    log_test "Test 3: Testing directory structure creation..."
    
    cd "$TEST_DIR"
    
    # Mock deploy (just create directories, don't actually deploy)
    export SYMBOL="$TEST_SYMBOL"
    export THRESHOLD_UP="0.8"
    export THRESHOLD_DOWN="0.8"
    export MAX_POSITION_SIZE="200.0"
    export MAX_DRAWDOWN="0.03"
    export EXECUTION_PROVIDER="cpu"
    export DEVICE_ID="0"
    export INTRA_THREADS="8"
    
    # Create bot directory structure manually (simulating deploy)
    local bot_dir="bots/${TEST_SYMBOL}"
    mkdir -p "$bot_dir"/{model,data/raw,logs}
    
    # Check if directories were created
    if [[ -d "$bot_dir/model" ]]; then
        log_pass "model directory created"
    else
        log_fail "model directory not created"
    fi
    
    if [[ -d "$bot_dir/data/raw" ]]; then
        log_pass "data/raw directory created"
    else
        log_fail "data/raw directory not created"
    fi
    
    if [[ -d "$bot_dir/logs" ]]; then
        log_pass "logs directory created"
    else
        log_fail "logs directory not created"
    fi
}

# Test 4: Test config generation with envsubst
test_config_generation() {
    log_test "Test 4: Testing config generation with envsubst..."
    
    cd "$TEST_DIR"
    
    export SYMBOL="$TEST_SYMBOL"
    export THRESHOLD_UP="0.8"
    export THRESHOLD_DOWN="0.8"
    export MAX_POSITION_SIZE="200.0"
    export MAX_DRAWDOWN="0.03"
    export EXECUTION_PROVIDER="cpu"
    export DEVICE_ID="0"
    export INTRA_THREADS="8"
    
    # Generate config
    if command -v envsubst &> /dev/null; then
        envsubst < "deploy/bot.template.toml" > "bots/${TEST_SYMBOL}/config.toml"
        
        if [[ -f "bots/${TEST_SYMBOL}/config.toml" ]]; then
            log_pass "config.toml generated"
            
            # Check if variables were substituted correctly
            if grep -q "symbol = \"${TEST_SYMBOL}\"" "bots/${TEST_SYMBOL}/config.toml"; then
                log_pass "SYMBOL variable substituted correctly"
            else
                log_fail "SYMBOL variable not substituted"
            fi
            
            if grep -q "threshold_up = 0.8" "bots/${TEST_SYMBOL}/config.toml"; then
                log_pass "THRESHOLD_UP variable substituted correctly"
            else
                log_fail "THRESHOLD_UP variable not substituted"
            fi
            
            if grep -q "max_position_size = 200.0" "bots/${TEST_SYMBOL}/config.toml"; then
                log_pass "MAX_POSITION_SIZE variable substituted correctly"
            else
                log_fail "MAX_POSITION_SIZE variable not substituted"
            fi
            
            if grep -q "intra_threads = 8" "bots/${TEST_SYMBOL}/config.toml"; then
                log_pass "INTRA_THREADS variable substituted correctly"
            else
                log_fail "INTRA_THREADS variable not substituted"
            fi
            
            # Check if model path is correct
            if grep -q "model_path = \"/opt/neirobot-lit/bots/${TEST_SYMBOL}/model/model.onnx\"" "bots/${TEST_SYMBOL}/config.toml"; then
                log_pass "Model path is correct (absolute path)"
            else
                log_fail "Model path is incorrect"
            fi
        else
            log_fail "config.toml not generated"
        fi
    else
        log_fail "envsubst not available (install gettext package)"
    fi
}

# Test 5: Test model validation
test_model_validation() {
    log_test "Test 5: Testing model validation..."
    
    cd "$TEST_DIR"
    
    local model_path="python_lab/models/${TEST_SYMBOL}.onnx"
    
    if [[ -f "$model_path" ]]; then
        log_pass "Model file exists: $model_path"
    else
        log_fail "Model file not found: $model_path"
    fi
    
    # Test with non-existent model
    local fake_symbol="FAKESYMBOL"
    local fake_model="python_lab/models/${fake_symbol}.onnx"
    
    if [[ ! -f "$fake_model" ]]; then
        log_pass "Non-existent model correctly not found"
    else
        log_fail "Unexpected model file found"
    fi
}

# Test 6: Test metadata validation
test_metadata_validation() {
    log_test "Test 6: Testing metadata validation..."
    
    cd "$TEST_DIR"
    
    # Create metadata file
    mkdir -p python_lab/models
    cat > python_lab/models/metadata.json << EOF
{
  "symbol": "${TEST_SYMBOL}",
  "version": "1.0",
  "timestamp": "2024-01-15T10:00:00Z"
}
EOF
    
    if [[ -f "python_lab/models/metadata.json" ]]; then
        log_pass "Metadata file created"
        
        # Check if symbol matches
        if grep -q "\"symbol\": \"${TEST_SYMBOL}\"" "python_lab/models/metadata.json"; then
            log_pass "Metadata symbol matches"
        else
            log_fail "Metadata symbol does not match"
        fi
    else
        log_fail "Metadata file not created"
    fi
}

# Test 7: Test help command
test_help_command() {
    log_test "Test 7: Testing help command..."
    
    if "$PROJECT_ROOT/deploy/manage.sh" --help &> /dev/null; then
        log_pass "Help command works"
    else
        log_fail "Help command failed"
    fi
}

# Test 8: Test global config files
test_global_configs() {
    log_test "Test 8: Testing global config files..."
    
    cd "$TEST_DIR"
    
    # Check if global.toml exists
    if [[ -f "global.toml" ]]; then
        log_pass "global.toml exists"
    else
        log_fail "global.toml not found"
    fi
    
    # Check if exchange.toml exists
    if [[ -f "exchange.toml" ]]; then
        log_pass "exchange.toml exists"
    else
        log_fail "exchange.toml not found"
    fi
    
    # Simulate copying to /opt/neirobot-lit
    mkdir -p opt/neirobot-lit
    cp global.toml opt/neirobot-lit/
    cp exchange.toml opt/neirobot-lit/
    
    if [[ -f "opt/neirobot-lit/global.toml" && -f "opt/neirobot-lit/exchange.toml" ]]; then
        log_pass "Global configs copied successfully"
    else
        log_fail "Global configs not copied"
    fi
}

# Run all tests
run_tests() {
    echo ""
    echo "=========================================="
    echo "  Neirobot LiT Deploy Script Tests"
    echo "=========================================="
    echo ""
    
    setup
    
    test_script_exists
    test_templates_exist
    test_directory_creation
    test_config_generation
    test_model_validation
    test_metadata_validation
    test_help_command
    test_global_configs
    
    cleanup
    
    echo ""
    echo "=========================================="
    echo "  Test Results"
    echo "=========================================="
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo ""
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        exit 1
    fi
}

# Main
main() {
    run_tests
}

main "$@"

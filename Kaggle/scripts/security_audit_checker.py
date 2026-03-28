#!/usr/bin/env python3
"""
Скрипт для проверки целостности логов аудита безопасности.

Валидирует криптографическую цепочку HMAC-SHA256 и обнаруживает аномалии.
"""

import csv
import hmac
import hashlib
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def validate_audit_chain(csv_path: str, master_key: str) -> Tuple[bool, List[str]]:
    """
    Валидирует цепочку хешей в файле аудита.
    
    Args:
        csv_path: Путь к файлу security_audit.csv
        master_key: Мастер-ключ для HMAC
    
    Returns:
        Кортеж (is_valid, errors)
    """
    errors = []
    
    if not os.path.exists(csv_path):
        errors.append(f"Audit log file not found: {csv_path}")
        return False, errors
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if reader.fieldnames is None:
                errors.append("CSV file is empty or malformed")
                return False, errors
            
            # Проверяем наличие необходимых полей
            required_fields = {'timestamp', 'actor', 'action', 'status', 'old_value', 'new_value', 'hash'}
            if not required_fields.issubset(set(reader.fieldnames)):
                errors.append(f"Missing required fields. Expected: {required_fields}")
                return False, errors
            
            # Генезис-хеш: HMAC-SHA256 от master_key с данными "GENESIS"
            genesis_mac = hmac.new(
                master_key.encode('utf-8'),
                b'GENESIS',
                hashlib.sha256
            )
            prev_hash = genesis_mac.digest()
            
            row_num = 1
            for row in reader:
                row_num += 1
                
                # Формируем строку для хеширования
                row_data = f"{row['timestamp']},{row['actor']},{row['action']},{row['status']},{row['old_value']},{row['new_value']}"
                
                # Вычисляем ожидаемый хеш: HMAC(key: master_key, data: prev_hash + row_data)
                mac = hmac.new(master_key.encode('utf-8'), hashlib.sha256)
                mac.update(prev_hash)
                mac.update(row_data.encode('utf-8'))
                expected_hash = mac.hexdigest()
                
                # Сравниваем с хешем из файла
                actual_hash = row['hash'].strip()
                if actual_hash != expected_hash:
                    errors.append(
                        f"Row {row_num}: Hash mismatch. "
                        f"Expected: {expected_hash}, Got: {actual_hash}"
                    )
                    return False, errors
                
                # Обновляем prev_hash для следующей итерации
                prev_hash = bytes.fromhex(expected_hash)
        
        if not errors:
            print(f"✓ Audit chain validation PASSED for {csv_path}")
            return True, errors
        
    except Exception as e:
        errors.append(f"Error reading audit log: {str(e)}")
        return False, errors
    
    return len(errors) == 0, errors


def detect_anomalies(csv_path: str) -> Dict[str, any]:
    """
    Обнаруживает аномалии в логах аудита.
    
    Проверяет:
    - Более 3 неудачных попыток расшифровки в час
    - Срабатывание более 5 риск-гейтов за 10 минут
    
    Args:
        csv_path: Путь к файлу security_audit.csv
    
    Returns:
        Словарь с результатами анализа
    """
    anomalies = {
        'decryption_failures': [],
        'risk_gate_triggers': [],
        'alerts': []
    }
    
    if not os.path.exists(csv_path):
        anomalies['alerts'].append(f"Audit log file not found: {csv_path}")
        return anomalies
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Группируем события по часам и 10-минутным интервалам
            decryption_by_hour = defaultdict(list)
            risk_gates_by_10min = defaultdict(list)
            
            for row in reader:
                timestamp_str = row['timestamp']
                action = row['action']
                status = row['status']
                
                try:
                    # Парсим timestamp (RFC3339 формат)
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except ValueError:
                    continue
                
                # Проверяем неудачные попытки расшифровки
                if action == 'CONFIG_DECRYPTION' and status == 'FAILURE':
                    hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                    decryption_by_hour[hour_key].append(timestamp)
                
                # Проверяем срабатывания риск-гейтов
                if action.startswith('RISK_GATE_') and status == 'TRIGGERED':
                    # Группируем по 10-минутным интервалам
                    minute_bucket = (timestamp.minute // 10) * 10
                    time_key = timestamp.strftime(f'%Y-%m-%d %H:{minute_bucket:02d}')
                    risk_gates_by_10min[time_key].append(timestamp)
            
            # Анализируем неудачные попытки расшифровки
            for hour, timestamps in decryption_by_hour.items():
                if len(timestamps) > 3:
                    anomalies['decryption_failures'].append({
                        'hour': hour,
                        'count': len(timestamps),
                        'timestamps': [ts.isoformat() for ts in timestamps]
                    })
                    anomalies['alerts'].append(
                        f"ALERT: {len(timestamps)} failed decryption attempts in hour {hour}"
                    )
            
            # Анализируем срабатывания риск-гейтов
            for time_bucket, timestamps in risk_gates_by_10min.items():
                if len(timestamps) > 5:
                    anomalies['risk_gate_triggers'].append({
                        'time_bucket': time_bucket,
                        'count': len(timestamps),
                        'timestamps': [ts.isoformat() for ts in timestamps]
                    })
                    anomalies['alerts'].append(
                        f"ALERT: {len(timestamps)} risk gate triggers in 10-minute bucket {time_bucket}"
                    )
        
    except Exception as e:
        anomalies['alerts'].append(f"Error analyzing anomalies: {str(e)}")
    
    return anomalies


def print_validation_report(csv_path: str, master_key: str) -> None:
    """
    Выводит полный отчет о валидации аудита.
    
    Args:
        csv_path: Путь к файлу security_audit.csv
        master_key: Мастер-ключ для HMAC
    """
    print("\n" + "="*70)
    print("SECURITY AUDIT LOG VALIDATION REPORT")
    print("="*70)
    print(f"Audit Log: {csv_path}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-"*70)
    
    # Валидация цепочки
    print("\n[1] CHAIN INTEGRITY CHECK")
    is_valid, errors = validate_audit_chain(csv_path, master_key)
    
    if is_valid:
        print("✓ PASSED: Audit chain is intact")
    else:
        print("✗ FAILED: Log tampering detected!")
        for error in errors:
            print(f"  - {error}")
    
    # Анализ аномалий
    print("\n[2] ANOMALY DETECTION")
    anomalies = detect_anomalies(csv_path)
    
    if anomalies['alerts']:
        print("⚠ ALERTS DETECTED:")
        for alert in anomalies['alerts']:
            print(f"  - {alert}")
    else:
        print("✓ No anomalies detected")
    
    if anomalies['decryption_failures']:
        print("\n  Decryption Failures by Hour:")
        for failure in anomalies['decryption_failures']:
            print(f"    - {failure['hour']}: {failure['count']} failures")
    
    if anomalies['risk_gate_triggers']:
        print("\n  Risk Gate Triggers by 10-min Bucket:")
        for trigger in anomalies['risk_gate_triggers']:
            print(f"    - {trigger['time_bucket']}: {trigger['count']} triggers")
    
    print("\n" + "="*70)
    
    # Итоговый статус
    if is_valid and not anomalies['alerts']:
        print("OVERALL STATUS: ✓ SECURE")
    elif is_valid and anomalies['alerts']:
        print("OVERALL STATUS: ⚠ WARNINGS")
    else:
        print("OVERALL STATUS: ✗ CRITICAL")
    
    print("="*70 + "\n")


def main():
    """Главная функция скрипта."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate security audit log integrity and detect anomalies'
    )
    parser.add_argument(
        '--symbol',
        required=True,
        help='Trading symbol (e.g., BTCUSDT)'
    )
    parser.add_argument(
        '--master-key',
        help='Master key for HMAC validation (reads from NEIRO_MASTER_KEY env var if not provided)'
    )
    parser.add_argument(
        '--audit-log',
        help='Path to audit log file (default: bots/SYMBOL/logs/security_audit.csv)'
    )
    
    args = parser.parse_args()
    
    # Получаем мастер-ключ
    master_key = args.master_key or os.environ.get('NEIRO_MASTER_KEY')
    if not master_key:
        print("ERROR: Master key not provided. Set NEIRO_MASTER_KEY environment variable or use --master-key")
        sys.exit(1)
    
    # Определяем путь к файлу аудита
    if args.audit_log:
        csv_path = args.audit_log
    else:
        csv_path = f"bots/{args.symbol}/logs/security_audit.csv"
    
    # Выводим отчет
    print_validation_report(csv_path, master_key)
    
    # Проверяем валидность и выходим с соответствующим кодом
    is_valid, _ = validate_audit_chain(csv_path, master_key)
    anomalies = detect_anomalies(csv_path)
    
    if not is_valid:
        sys.exit(2)  # Критическая ошибка
    elif anomalies['alerts']:
        sys.exit(1)  # Предупреждения
    else:
        sys.exit(0)  # OK


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Задача 208: Анализ оптимизации переключения Passive -> Aggressive

Функционал:
- Анализ "упущенной выгоды" (Opportunity Cost)
- Поиск баланса между экономией на Maker-комиссии и потерей Alpha из-за неисполнения
- Оценка эффективности переключения на агрессивный режим
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import json
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SwitchMetrics:
    """Метрики переключения Passive -> Aggressive"""
    order_id: str
    symbol: str
    side: str
    initial_price: float
    switch_price: float
    final_price: float
    qty: float
    executed_qty: float
    remaining_qty: float
    switch_latency_ms: float
    maker_fee: float  # Обычно -0.02% (отрицательное значение = скидка)
    taker_fee: float  # Обычно 0.055%
    opportunity_cost: float  # Потеря Alpha из-за неисполнения
    fee_savings: float  # Экономия на комиссии благодаря Maker
    net_benefit: float  # fee_savings - opportunity_cost


class SwitchOptimizer:
    """Оптимизатор параметров переключения"""
    
    def __init__(self, maker_fee: float = -0.0002, taker_fee: float = 0.00055):
        """
        Args:
            maker_fee: Maker комиссия (обычно отрицательная = скидка)
            taker_fee: Taker комиссия
        """
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
    
    def calculate_fee_impact(
        self,
        qty: float,
        price: float,
        executed_qty: float,
        is_maker: bool,
    ) -> float:
        """
        Расчет влияния комиссии на PnL
        
        Args:
            qty: Объем ордера
            price: Цена ордера
            executed_qty: Исполненный объем
            is_maker: True если это Maker, False если Taker
        
        Returns:
            Комиссия в абсолютном значении (отрицательное = скидка)
        """
        notional = executed_qty * price
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return notional * fee_rate
    
    def calculate_opportunity_cost(
        self,
        initial_price: float,
        switch_price: float,
        final_price: float,
        remaining_qty: float,
        side: str,
    ) -> float:
        """
        Расчет упущенной выгоды (Opportunity Cost)
        
        Если ордер не был исполнен по начальной цене и пришлось переключиться,
        это означает потерю потенциальной выгоды.
        
        Args:
            initial_price: Начальная цена лимитного ордера
            switch_price: Цена переключения (когда отменили лимит)
            final_price: Финальная цена исполнения
            remaining_qty: Остаток, который не был исполнен
            side: "Buy" или "Sell"
        
        Returns:
            Opportunity cost в абсолютном значении
        """
        if remaining_qty <= 0:
            return 0.0
        
        if side.upper() == "BUY":
            # Для покупки: если цена выросла, мы потеряли на разнице
            price_diff = switch_price - initial_price
            opportunity_cost = remaining_qty * price_diff
        else:  # SELL
            # Для продажи: если цена упала, мы потеряли на разнице
            price_diff = initial_price - switch_price
            opportunity_cost = remaining_qty * price_diff
        
        return max(0.0, opportunity_cost)
    
    def analyze_switch_event(
        self,
        metrics: SwitchMetrics,
    ) -> Dict[str, float]:
        """
        Анализ одного события переключения
        
        Args:
            metrics: Метрики переключения
        
        Returns:
            Словарь с анализом
        """
        # Расчет комиссий
        maker_fee_cost = self.calculate_fee_impact(
            metrics.qty,
            metrics.initial_price,
            metrics.executed_qty,
            is_maker=True,
        )
        
        taker_fee_cost = self.calculate_fee_impact(
            metrics.qty,
            metrics.switch_price,
            metrics.remaining_qty,
            is_maker=False,
        )
        
        # Экономия на комиссии (если бы весь ордер был Maker)
        full_maker_fee = self.calculate_fee_impact(
            metrics.qty,
            metrics.initial_price,
            metrics.qty,
            is_maker=True,
        )
        
        # Стоимость Taker части
        taker_cost = self.calculate_fee_impact(
            metrics.qty,
            metrics.switch_price,
            metrics.remaining_qty,
            is_maker=False,
        )
        
        fee_savings = full_maker_fee - (maker_fee_cost + taker_cost)
        
        # Opportunity cost
        opportunity_cost = self.calculate_opportunity_cost(
            metrics.initial_price,
            metrics.switch_price,
            metrics.final_price,
            metrics.remaining_qty,
            metrics.side,
        )
        
        # Net benefit
        net_benefit = fee_savings - opportunity_cost
        
        return {
            "maker_fee_cost": maker_fee_cost,
            "taker_fee_cost": taker_fee_cost,
            "fee_savings": fee_savings,
            "opportunity_cost": opportunity_cost,
            "net_benefit": net_benefit,
            "switch_latency_ms": metrics.switch_latency_ms,
            "execution_ratio": metrics.executed_qty / metrics.qty if metrics.qty > 0 else 0.0,
        }
    
    def optimize_switch_threshold(
        self,
        events: List[SwitchMetrics],
    ) -> Dict[str, any]:
        """
        Оптимизация порога переключения на основе исторических данных
        
        Args:
            events: Список событий переключения
        
        Returns:
            Рекомендации по оптимизации
        """
        if not events:
            logger.warning("No switch events to analyze")
            return {}
        
        results = []
        for event in events:
            analysis = self.analyze_switch_event(event)
            analysis["order_id"] = event.order_id
            analysis["symbol"] = event.symbol
            analysis["side"] = event.side
            results.append(analysis)
        
        df = pd.DataFrame(results)
        
        # Статистика
        stats = {
            "total_events": len(df),
            "avg_net_benefit": df["net_benefit"].mean(),
            "median_net_benefit": df["net_benefit"].median(),
            "std_net_benefit": df["net_benefit"].std(),
            "profitable_switches": (df["net_benefit"] > 0).sum(),
            "profitable_ratio": (df["net_benefit"] > 0).sum() / len(df),
            "avg_opportunity_cost": df["opportunity_cost"].mean(),
            "avg_fee_savings": df["fee_savings"].mean(),
            "avg_switch_latency_ms": df["switch_latency_ms"].mean(),
            "avg_execution_ratio": df["execution_ratio"].mean(),
        }
        
        # Рекомендации
        recommendations = []
        
        if stats["profitable_ratio"] < 0.5:
            recommendations.append(
                "Переключение на агрессивный режим убыточно в более чем 50% случаев. "
                "Рассмотрите увеличение базового timeout или уменьшение urgency."
            )
        
        if stats["avg_opportunity_cost"] > stats["avg_fee_savings"]:
            recommendations.append(
                "Opportunity cost превышает экономию на комиссии. "
                "Переключение происходит слишком рано."
            )
        
        if stats["avg_switch_latency_ms"] > 500:
            recommendations.append(
                "Среднее время переключения > 500мс. "
                "Рассмотрите оптимизацию сетевой задержки или логики отмены."
            )
        
        return {
            "statistics": stats,
            "recommendations": recommendations,
            "detailed_results": df.to_dict(orient="records"),
        }


def load_execution_logs(log_path: Path) -> List[SwitchMetrics]:
    """
    Загрузка логов исполнения и извлечение событий переключения
    
    Args:
        log_path: Путь к файлу логов (CSV или JSON)
    
    Returns:
        Список SwitchMetrics
    """
    events = []
    
    if log_path.suffix == ".csv":
        df = pd.read_csv(log_path)
        
        # Фильтруем только события с переключением
        switch_events = df[df.get("final_type") == "Taker_After_Passive"]
        
        for _, row in switch_events.iterrows():
            try:
                metrics = SwitchMetrics(
                    order_id=str(row.get("order_id", "")),
                    symbol=str(row.get("symbol", "")),
                    side=str(row.get("side", "")),
                    initial_price=float(row.get("initial_price", 0)),
                    switch_price=float(row.get("switch_price", 0)),
                    final_price=float(row.get("final_price", 0)),
                    qty=float(row.get("qty", 0)),
                    executed_qty=float(row.get("executed_qty", 0)),
                    remaining_qty=float(row.get("remaining_qty", 0)),
                    switch_latency_ms=float(row.get("switch_latency_ms", 0)),
                    maker_fee=-0.0002,
                    taker_fee=0.00055,
                    opportunity_cost=0.0,
                    fee_savings=0.0,
                    net_benefit=0.0,
                )
                events.append(metrics)
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse row: {e}")
                continue
    
    elif log_path.suffix == ".json":
        with open(log_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if item.get("final_type") == "Taker_After_Passive":
                        try:
                            metrics = SwitchMetrics(**item)
                            events.append(metrics)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Failed to parse item: {e}")
    
    return events


def main():
    """Основная функция для анализа"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Анализ оптимизации переключения Passive -> Aggressive"
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Путь к файлу логов исполнения (CSV или JSON)",
    )
    parser.add_argument(
        "--maker-fee",
        type=float,
        default=-0.0002,
        help="Maker комиссия (default: -0.0002)",
    )
    parser.add_argument(
        "--taker-fee",
        type=float,
        default=0.00055,
        help="Taker комиссия (default: 0.00055)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Путь для сохранения результатов анализа",
    )
    
    args = parser.parse_args()
    
    optimizer = SwitchOptimizer(
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
    )
    
    if args.log_path and args.log_path.exists():
        logger.info(f"Loading execution logs from {args.log_path}")
        events = load_execution_logs(args.log_path)
        
        if events:
            logger.info(f"Found {len(events)} switch events")
            results = optimizer.optimize_switch_threshold(events)
            
            logger.info("\n=== Switch Optimization Analysis ===")
            logger.info(f"Statistics: {json.dumps(results['statistics'], indent=2)}")
            logger.info(f"Recommendations: {json.dumps(results['recommendations'], indent=2)}")
            
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {args.output}")
        else:
            logger.warning("No switch events found in logs")
    else:
        logger.info("No log path provided. Running with example data...")
        
        # Пример данных
        example_events = [
            SwitchMetrics(
                order_id="order_1",
                symbol="BTCUSDT",
                side="BUY",
                initial_price=45000.0,
                switch_price=45050.0,
                final_price=45100.0,
                qty=1.0,
                executed_qty=0.5,
                remaining_qty=0.5,
                switch_latency_ms=250.0,
                maker_fee=-0.0002,
                taker_fee=0.00055,
                opportunity_cost=0.0,
                fee_savings=0.0,
                net_benefit=0.0,
            ),
        ]
        
        results = optimizer.optimize_switch_threshold(example_events)
        logger.info(f"Example analysis: {json.dumps(results['statistics'], indent=2)}")


if __name__ == "__main__":
    main()

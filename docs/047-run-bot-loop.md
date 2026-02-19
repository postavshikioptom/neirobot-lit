# 047 - Run Bot Main Loop
Цель задачи: Реализовать отказоустойчивый основной цикл обработки событий в src/bin/run-bot.rs. Цикл должен координировать поток данных от WebSocket к модулю исполнения, обеспечивать обработку ошибок без остановки бота, замерять задержку инференса и подготавливать почву для механизмов реконнекта и проверки целостности данных.

Файлы: src/bin/run-bot.rs (обновить)

Инструкции для Gemini:

Структура цикла: Реализовать бесконечный цикл с tokio::select!, обрабатывающий сигналы завершения и входящие сообщения из канала.
Отказоустойчивость: Использовать match или if let вместо оператора ? внутри цикла, чтобы ошибки парсинга или инференса не приводили к завершению работы бота.
ML Pipeline и Latency: Добавить замер времени выполнения engine.predict с использованием std::time::Instant. Логировать задержку, если она превышает 50мс (задача 082).
Обработка разрыва соединения: При закрытии канала (None от rx.recv()) инициировать логику переподключения (задача 048).
// Пример реализации основного цикла:
info!("Starting main loop for {}", args.symbol);

loop {
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("Shutdown signal received. Closing bot...");
            break;
        }
        msg = rx.recv() => {
            let text = match msg {
                Some(t) => t,
                None => {
                    error!("WebSocket channel closed! Attempting to reconnect...");
                    // TODO: Reconnect logic (Task 048)
                    continue; 
                }
            };

            // Обработка данных с защитой от паники
            if let Err(e) = handle_message(&text, &args.symbol, &mut ob, &mut tensor_builder, &engine, &mut execution) {
                error!("Error processing message: {}", e);
            }
        }
    }
}

fn handle_message(...) -> Result<()> {
    // 1. Парсинг
    let update = parse_orderbook_update(text, symbol).context("Parse failed")?;
    if update.is_none() { return Ok(()); }
    let update = update.unwrap();

    // 2. Валидация Checksum (Задача 049)
    // if !validate_checksum(&ob, &update) { bail!("Checksum mismatch!"); }

    // 3. Обновление LOB
    ob.apply_update(update);
    let price = ob.get_mid_price();
    if price == 0.0 { return Ok(()); }

    // 4. ML Pipeline
    if let Some(tensor) = tensor_builder.process_snapshot(&ob)? {
        let start = std::time::Instant::now();
        let inference = engine.predict(&tensor).context("Inference failed")?;
        let latency = start.elapsed();
        
        if latency.as_millis() > 50 {
            warn!("High inference latency: {:?} for {}", latency, symbol);
        }

        // 5. Execution
        execution.on_inference_output(inference, price).context("Execution failed")?;
    }
    Ok(())
}
Технические требования:

Resilience: Любая ошибка внутри handle_message должна логироваться как error!, но не прерывать основной цикл loop.
Latency: Использовать Instant для мониторинга производительности нейросети.
Checksum: Оставить место (комментарий/заглушку) для проверки контрольной суммы Bybit.
Reconnect: При rx.recv() -> None бот должен пытаться восстановить соединение (пересоздать клиент и канал), а не завершаться.
Почему это важно: Стабильность — главное качество торгового бота. Этот цикл превращает набор компонентов в надежную систему, которая может работать неделями, игнорируя единичные ошибки данных и отслеживая критические задержки инференса.
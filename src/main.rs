use anyhow::Result;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main]
async fn main() -> Result<()> {
    // Задача 219: Установка panic hook для записи backtrace в лог
    std::panic::set_hook(Box::new(|panic_info| {
        let backtrace = std::backtrace::Backtrace::capture();
        
        // Пытаемся логировать через tracing если инициализирован
        if let Some(location) = panic_info.location() {
            eprintln!(
                "PANIC at {}:{}:{}\n{}",
                location.file(),
                location.line(),
                location.column(),
                backtrace
            );
        } else {
            eprintln!("PANIC (location unknown)\n{}", backtrace);
        }
        
        // Также пытаемся вывести сообщение паники
        if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            eprintln!("Message: {}", s);
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            eprintln!("Message: {}", s);
        }
    }));
    
    println!("Neirobot LIT — проект инициализирован");
    // TODO: в будущем — парсинг CLI-аргументов и выбор режима
    Ok(())
}

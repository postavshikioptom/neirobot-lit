pub mod types;
pub mod order;
pub mod order_manager;
pub mod position_manager;

pub mod execution;
pub mod rest_client;
pub mod emergency;
pub mod regime_detector;
pub mod state;

pub use types::*;
// Экспортируем новую Order как основную
pub use order::Order;
// Старая Order доступна через types::LegacyOrder
pub use order_manager::*;
pub use position_manager::*;

pub use execution::*;
pub use rest_client::{BybitRestClient, BybitRestClientTrait, BybitOrderListResponse, RemoteOrder};
pub use regime_detector::RegimeDetector;
pub use state::{BotStateData, PersistentState, StatePersistenceManager, save_state, load_state};
// BotState берется из types.rs (pub type BotState = RiskState)

//! Logical planning: turning an AST into a typed, resolved plan, then
//! rewriting that plan into a cheaper equivalent.

pub mod expr;
pub mod optimizer;
pub mod plan;
pub mod planner;

pub use expr::{AggregateExpr, AggregateFunction, LogicalExpr};
pub use optimizer::{optimize, Optimizer};
pub use plan::LogicalPlan;
pub use planner::Planner;

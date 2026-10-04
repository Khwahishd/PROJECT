//! Physical planning and vectorized execution.

pub mod aggregate;
pub mod expr;
pub mod join;
pub mod plan;
pub mod planner;

pub use plan::{collect, explain, Executor};
pub use planner::create_physical_plan;

//! # quarry
//!
//! A vectorized analytical SQL query engine, written from scratch with no
//! third-party dependencies.
//!
//! A query flows through four stages, each in its own module:
//!
//! ```text
//!   SQL text
//!     │  sql::parse            hand-written lexer + Pratt parser
//!     ▼
//!   AST                        a faithful record of what was written
//!     │  logical::planner      name resolution, type checking, aggregate extraction
//!     ▼
//!   LogicalPlan                what to compute
//!     │  logical::optimizer    projection/predicate pushdown, constant folding
//!     ▼
//!   LogicalPlan (optimized)
//!     │  physical::planner     operator selection
//!     ▼
//!   PhysicalPlan               how to compute it -- vectorized, batch at a time
//! ```
//!
//! ## Example
//!
//! ```no_run
//! use quarry::Context;
//!
//! let mut ctx = Context::new();
//! ctx.register_csv("trips", "data/trips.csv").unwrap();
//!
//! let results = ctx.sql("
//!     SELECT city, COUNT(*) AS n, AVG(fare) AS avg_fare
//!     FROM trips
//!     WHERE fare > 10
//!     GROUP BY city
//!     ORDER BY n DESC
//!     LIMIT 5
//! ").unwrap();
//!
//! println!("{}", results.to_table());
//! ```

#![warn(missing_docs)]

pub mod array;
pub mod batch;
pub mod csv;
pub mod error;
pub mod logical;
pub mod physical;
pub mod sql;
pub mod types;

mod context;

pub use batch::{RecordBatch, DEFAULT_BATCH_SIZE};
pub use context::{Context, QueryResult};
pub use error::{Error, Result};
pub use types::{DataType, Field, Schema, Value};

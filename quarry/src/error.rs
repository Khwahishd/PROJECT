//! Error types for every stage of the engine.

use std::fmt;

/// The result type used throughout quarry.
pub type Result<T> = std::result::Result<T, Error>;

/// An error produced while parsing, planning or executing a query.
///
/// Errors carry the stage they came from, because the same underlying problem
/// ("no column named x") means something quite different depending on whether
/// it surfaced during name resolution or halfway through execution -- the
/// latter is an engine bug, not a user error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The query text could not be tokenized or parsed.
    Parse(String),
    /// The query parsed but does not describe a valid plan: unknown table or
    /// column, type mismatch, aggregate in the wrong place.
    Plan(String),
    /// A failure during execution. These generally indicate a bug in the
    /// engine rather than a problem with the query.
    Execution(String),
    /// A type was used where it is not supported.
    Type(String),
    /// An I/O or data-source failure.
    Io(String),
    /// A feature that is recognised but not implemented.
    NotImplemented(String),
}

impl Error {
    /// Convenience constructor for a parse error.
    pub fn parse(msg: impl Into<String>) -> Self {
        Error::Parse(msg.into())
    }
    /// Convenience constructor for a planning error.
    pub fn plan(msg: impl Into<String>) -> Self {
        Error::Plan(msg.into())
    }
    /// Convenience constructor for an execution error.
    pub fn exec(msg: impl Into<String>) -> Self {
        Error::Execution(msg.into())
    }
    /// Convenience constructor for a type error.
    pub fn typ(msg: impl Into<String>) -> Self {
        Error::Type(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse(m) => write!(f, "parse error: {m}"),
            Error::Plan(m) => write!(f, "planning error: {m}"),
            Error::Execution(m) => write!(f, "execution error: {m}"),
            Error::Type(m) => write!(f, "type error: {m}"),
            Error::Io(m) => write!(f, "io error: {m}"),
            Error::NotImplemented(m) => write!(f, "not implemented: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

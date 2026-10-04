//! The abstract syntax tree produced by the parser.
//!
//! The AST is deliberately a faithful, untyped record of what the user wrote.
//! No name resolution, type checking or rewriting happens here -- that is the
//! planner's job. Keeping the two separate means parse errors are always about
//! syntax and planning errors are always about meaning.

use crate::types::Value;
use std::fmt;

/// A complete statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// A query, optionally wrapped in EXPLAIN.
    Query(Box<Query>),
    /// `EXPLAIN <query>`, which prints plans instead of executing.
    Explain(Box<Query>),
}

/// A SELECT query.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    /// Whether DISTINCT was specified.
    pub distinct: bool,
    /// The projection list.
    pub projection: Vec<SelectItem>,
    /// The FROM clause, if any.
    pub from: Option<TableRef>,
    /// The WHERE predicate.
    pub selection: Option<Expr>,
    /// GROUP BY expressions.
    pub group_by: Vec<Expr>,
    /// The HAVING predicate.
    pub having: Option<Expr>,
    /// ORDER BY terms.
    pub order_by: Vec<OrderByExpr>,
    /// LIMIT.
    pub limit: Option<usize>,
    /// OFFSET.
    pub offset: Option<usize>,
}

/// One entry in a SELECT list.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem {
    /// `*`
    Wildcard,
    /// `alias.*`
    QualifiedWildcard(String),
    /// An expression, with an optional `AS` alias.
    Expr {
        /// The expression.
        expr: Expr,
        /// The alias, if given.
        alias: Option<String>,
    },
}

/// A table in the FROM clause, possibly a join tree.
#[derive(Debug, Clone, PartialEq)]
pub enum TableRef {
    /// A named table with an optional alias.
    Table {
        /// Table name.
        name: String,
        /// Alias, if given.
        alias: Option<String>,
    },
    /// A join of two table references.
    Join {
        /// Left input.
        left: Box<TableRef>,
        /// Right input.
        right: Box<TableRef>,
        /// Join kind.
        join_type: JoinType,
        /// The ON predicate; absent for a cross join.
        on: Option<Expr>,
    },
}

/// The kind of join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Rows matching on both sides.
    Inner,
    /// All left rows, NULL-extended where unmatched.
    Left,
    /// All right rows, NULL-extended where unmatched.
    Right,
    /// The Cartesian product.
    Cross,
}

impl fmt::Display for JoinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            JoinType::Inner => "Inner",
            JoinType::Left => "Left",
            JoinType::Right => "Right",
            JoinType::Cross => "Cross",
        })
    }
}

/// An ORDER BY term.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderByExpr {
    /// The sort key.
    pub expr: Expr,
    /// Ascending if true.
    pub asc: bool,
}

/// A scalar expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A literal value.
    Literal(Value),
    /// A column reference, optionally qualified by a table or alias.
    Column {
        /// The qualifying table or alias, if written.
        qualifier: Option<String>,
        /// The column name.
        name: String,
    },
    /// A binary operation.
    Binary {
        /// Left operand.
        left: Box<Expr>,
        /// The operator.
        op: BinaryOp,
        /// Right operand.
        right: Box<Expr>,
    },
    /// A unary operation.
    Unary {
        /// The operator.
        op: UnaryOp,
        /// The operand.
        expr: Box<Expr>,
    },
    /// A function call, which may be an aggregate.
    Function {
        /// Function name, upper-cased.
        name: String,
        /// Arguments; empty with `star` set for `COUNT(*)`.
        args: Vec<Expr>,
        /// Whether the single argument was `*`.
        star: bool,
        /// Whether DISTINCT was given inside the call.
        distinct: bool,
    },
    /// `expr IS NULL` / `IS NOT NULL`.
    IsNull {
        /// The operand.
        expr: Box<Expr>,
        /// True for `IS NOT NULL`.
        negated: bool,
    },
    /// `expr IN (list)`.
    InList {
        /// The operand.
        expr: Box<Expr>,
        /// The candidate values.
        list: Vec<Expr>,
        /// True for `NOT IN`.
        negated: bool,
    },
    /// `expr BETWEEN low AND high`.
    Between {
        /// The operand.
        expr: Box<Expr>,
        /// Lower bound, inclusive.
        low: Box<Expr>,
        /// Upper bound, inclusive.
        high: Box<Expr>,
        /// True for `NOT BETWEEN`.
        negated: bool,
    },
    /// `expr LIKE pattern`.
    Like {
        /// The operand.
        expr: Box<Expr>,
        /// The pattern.
        pattern: Box<Expr>,
        /// True for `NOT LIKE`.
        negated: bool,
    },
    /// `CASE WHEN ... THEN ... ELSE ... END`.
    Case {
        /// The WHEN/THEN pairs.
        when_then: Vec<(Expr, Expr)>,
        /// The ELSE branch, if any.
        else_expr: Option<Box<Expr>>,
    },
    /// `CAST(expr AS type)`.
    Cast {
        /// The operand.
        expr: Box<Expr>,
        /// The target type name as written.
        data_type: String,
    },
    /// A parenthesized expression, preserved so that EXPLAIN output can round
    /// trip faithfully.
    Nested(Box<Expr>),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum BinaryOp {
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

impl BinaryOp {
    /// Whether the operator yields a boolean.
    pub fn is_comparison(&self) -> bool {
        use BinaryOp::*;
        matches!(self, Eq | NotEq | Lt | LtEq | Gt | GtEq)
    }

    /// Whether the operator is a boolean connective.
    pub fn is_logical(&self) -> bool {
        matches!(self, BinaryOp::And | BinaryOp::Or)
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use BinaryOp::*;
        f.write_str(match self {
            Plus => "+",
            Minus => "-",
            Multiply => "*",
            Divide => "/",
            Modulo => "%",
            Eq => "=",
            NotEq => "<>",
            Lt => "<",
            LtEq => "<=",
            Gt => ">",
            GtEq => ">=",
            And => "AND",
            Or => "OR",
        })
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum UnaryOp {
    Not,
    Negate,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            UnaryOp::Not => "NOT ",
            UnaryOp::Negate => "-",
        })
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Literal(v) => match v {
                Value::Utf8(s) => write!(f, "'{s}'"),
                other => write!(f, "{other}"),
            },
            Expr::Column { qualifier: Some(q), name } => write!(f, "{q}.{name}"),
            Expr::Column { qualifier: None, name } => f.write_str(name),
            Expr::Binary { left, op, right } => write!(f, "{left} {op} {right}"),
            Expr::Unary { op, expr } => write!(f, "{op}{expr}"),
            Expr::Function { name, args, star, distinct } => {
                let inner = if *star {
                    "*".to_string()
                } else {
                    args.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ")
                };
                let d = if *distinct { "DISTINCT " } else { "" };
                write!(f, "{name}({d}{inner})")
            }
            Expr::IsNull { expr, negated } => {
                write!(f, "{expr} IS {}NULL", if *negated { "NOT " } else { "" })
            }
            Expr::InList { expr, list, negated } => {
                let items = list.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", ");
                write!(f, "{expr} {}IN ({items})", if *negated { "NOT " } else { "" })
            }
            Expr::Between { expr, low, high, negated } => {
                write!(f, "{expr} {}BETWEEN {low} AND {high}", if *negated { "NOT " } else { "" })
            }
            Expr::Like { expr, pattern, negated } => {
                write!(f, "{expr} {}LIKE {pattern}", if *negated { "NOT " } else { "" })
            }
            Expr::Case { when_then, else_expr } => {
                write!(f, "CASE")?;
                for (w, t) in when_then {
                    write!(f, " WHEN {w} THEN {t}")?;
                }
                if let Some(e) = else_expr {
                    write!(f, " ELSE {e}")?;
                }
                write!(f, " END")
            }
            Expr::Cast { expr, data_type } => write!(f, "CAST({expr} AS {data_type})"),
            Expr::Nested(e) => write!(f, "({e})"),
        }
    }
}

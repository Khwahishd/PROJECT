//! The SQL lexer.
//!
//! A hand-written scanner rather than a generated one: SQL's lexical grammar is
//! small, and writing it by hand keeps precise control over where a token
//! starts and ends -- which is what lets parse errors point at the offending
//! character instead of "somewhere in this statement".

use crate::error::{Error, Result};
use std::fmt;

/// A lexical token together with the byte offset where it began.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// What kind of token this is.
    pub kind: TokenKind,
    /// Byte offset of the token's first character, used in error messages.
    pub pos: usize,
}

/// The kinds of token the lexer produces.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// A bare identifier or a quoted one.
    Ident(String),
    /// A reserved word, normalized to upper case.
    Keyword(Keyword),
    /// An integer literal.
    Int(i64),
    /// A floating-point literal.
    Float(f64),
    /// A single-quoted string literal.
    String(String),

    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `,`
    Comma,
    /// `.`
    Dot,
    /// `*`
    Star,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `=`
    Eq,
    /// `<>` or `!=`
    NotEq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
    /// `;`
    Semicolon,

    /// End of input.
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Ident(s) => write!(f, "identifier {s:?}"),
            TokenKind::Keyword(k) => write!(f, "keyword {k:?}"),
            TokenKind::Int(v) => write!(f, "integer {v}"),
            TokenKind::Float(v) => write!(f, "float {v}"),
            TokenKind::String(s) => write!(f, "string {s:?}"),
            TokenKind::Eof => f.write_str("end of input"),
            other => write!(f, "{}", symbol_text(other)),
        }
    }
}

fn symbol_text(t: &TokenKind) -> &'static str {
    match t {
        TokenKind::LParen => "(",
        TokenKind::RParen => ")",
        TokenKind::Comma => ",",
        TokenKind::Dot => ".",
        TokenKind::Star => "*",
        TokenKind::Plus => "+",
        TokenKind::Minus => "-",
        TokenKind::Slash => "/",
        TokenKind::Percent => "%",
        TokenKind::Eq => "=",
        TokenKind::NotEq => "<>",
        TokenKind::Lt => "<",
        TokenKind::LtEq => "<=",
        TokenKind::Gt => ">",
        TokenKind::GtEq => ">=",
        TokenKind::Semicolon => ";",
        _ => "?",
    }
}

/// SQL reserved words recognized by the parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum Keyword {
    Select,
    From,
    Where,
    Group,
    By,
    Having,
    Order,
    Limit,
    Offset,
    As,
    And,
    Or,
    Not,
    Null,
    Is,
    In,
    Between,
    Like,
    Join,
    Inner,
    Left,
    Right,
    Full,
    Outer,
    On,
    Cross,
    Asc,
    Desc,
    Distinct,
    Case,
    When,
    Then,
    Else,
    End,
    True,
    False,
    Explain,
    Cast,
    Create,
    Table,
    Insert,
    Into,
    Values,
}

impl Keyword {
    fn from_str(s: &str) -> Option<Keyword> {
        use Keyword::*;
        Some(match s.to_ascii_uppercase().as_str() {
            "SELECT" => Select,
            "FROM" => From,
            "WHERE" => Where,
            "GROUP" => Group,
            "BY" => By,
            "HAVING" => Having,
            "ORDER" => Order,
            "LIMIT" => Limit,
            "OFFSET" => Offset,
            "AS" => As,
            "AND" => And,
            "OR" => Or,
            "NOT" => Not,
            "NULL" => Null,
            "IS" => Is,
            "IN" => In,
            "BETWEEN" => Between,
            "LIKE" => Like,
            "JOIN" => Join,
            "INNER" => Inner,
            "LEFT" => Left,
            "RIGHT" => Right,
            "FULL" => Full,
            "OUTER" => Outer,
            "ON" => On,
            "CROSS" => Cross,
            "ASC" => Asc,
            "DESC" => Desc,
            "DISTINCT" => Distinct,
            "CASE" => Case,
            "WHEN" => When,
            "THEN" => Then,
            "ELSE" => Else,
            "END" => End,
            "TRUE" => True,
            "FALSE" => False,
            "EXPLAIN" => Explain,
            "CAST" => Cast,
            "CREATE" => Create,
            "TABLE" => Table,
            "INSERT" => Insert,
            "INTO" => Into,
            "VALUES" => Values,
            _ => return None,
        })
    }
}

/// Converts SQL text into a token stream.
pub fn tokenize(input: &str) -> Result<Vec<Token>> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        let start = i;
        let c = bytes[i] as char;

        // Whitespace.
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Line comments.
        if c == '-' && i + 1 < bytes.len() && bytes[i + 1] == b'-' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Block comments. Unterminated ones are an error rather than being
        // silently swallowed to end of input.
        if c == '/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            let open = i;
            i += 2;
            loop {
                if i + 1 >= bytes.len() {
                    return Err(Error::parse(format!(
                        "unterminated block comment starting at byte {open}"
                    )));
                }
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }

        // Numbers.
        if c.is_ascii_digit() {
            let mut saw_dot = false;
            let mut saw_exp = false;
            while i < bytes.len() {
                let ch = bytes[i] as char;
                if ch.is_ascii_digit() {
                    i += 1;
                } else if ch == '.' && !saw_dot && !saw_exp {
                    saw_dot = true;
                    i += 1;
                } else if (ch == 'e' || ch == 'E') && !saw_exp && i + 1 < bytes.len() {
                    // Only treat `e` as an exponent if a number actually
                    // follows, so that `1east` lexes as 1 then an identifier.
                    let next = bytes[i + 1];
                    let has_digits = next.is_ascii_digit()
                        || ((next == b'+' || next == b'-')
                            && i + 2 < bytes.len()
                            && bytes[i + 2].is_ascii_digit());
                    if !has_digits {
                        break;
                    }
                    saw_exp = true;
                    i += if next == b'+' || next == b'-' { 2 } else { 1 };
                } else {
                    break;
                }
            }
            let text = &input[start..i];
            let kind = if saw_dot || saw_exp {
                TokenKind::Float(text.parse().map_err(|_| {
                    Error::parse(format!("invalid numeric literal {text:?} at byte {start}"))
                })?)
            } else {
                match text.parse::<i64>() {
                    Ok(v) => TokenKind::Int(v),
                    // Too large for i64: fall back to a float rather than
                    // rejecting the query outright.
                    Err(_) => TokenKind::Float(text.parse().map_err(|_| {
                        Error::parse(format!("invalid numeric literal {text:?} at byte {start}"))
                    })?),
                }
            };
            tokens.push(Token { kind, pos: start });
            continue;
        }

        // Identifiers and keywords.
        if c.is_ascii_alphabetic() || c == '_' {
            while i < bytes.len()
                && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_')
            {
                i += 1;
            }
            let text = &input[start..i];
            let kind = match Keyword::from_str(text) {
                Some(k) => TokenKind::Keyword(k),
                None => TokenKind::Ident(text.to_string()),
            };
            tokens.push(Token { kind, pos: start });
            continue;
        }

        // Double-quoted identifiers.
        if c == '"' {
            i += 1;
            let s = start + 1;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(Error::parse(format!(
                    "unterminated quoted identifier starting at byte {start}"
                )));
            }
            tokens.push(Token { kind: TokenKind::Ident(input[s..i].to_string()), pos: start });
            i += 1;
            continue;
        }

        // String literals, with '' as an escaped quote.
        if c == '\'' {
            i += 1;
            let mut value = String::new();
            loop {
                if i >= bytes.len() {
                    return Err(Error::parse(format!(
                        "unterminated string literal starting at byte {start}"
                    )));
                }
                if bytes[i] == b'\'' {
                    if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                        value.push('\'');
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                let ch_start = i;
                // Advance by whole UTF-8 characters so multi-byte text works.
                i += 1;
                while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
                    i += 1;
                }
                value.push_str(&input[ch_start..i]);
            }
            tokens.push(Token { kind: TokenKind::String(value), pos: start });
            continue;
        }

        // Operators and punctuation.
        let (kind, width) = match c {
            '(' => (TokenKind::LParen, 1),
            ')' => (TokenKind::RParen, 1),
            ',' => (TokenKind::Comma, 1),
            '.' => (TokenKind::Dot, 1),
            '*' => (TokenKind::Star, 1),
            '+' => (TokenKind::Plus, 1),
            '-' => (TokenKind::Minus, 1),
            '/' => (TokenKind::Slash, 1),
            '%' => (TokenKind::Percent, 1),
            ';' => (TokenKind::Semicolon, 1),
            '=' => (TokenKind::Eq, 1),
            '<' => match bytes.get(i + 1) {
                Some(b'=') => (TokenKind::LtEq, 2),
                Some(b'>') => (TokenKind::NotEq, 2),
                _ => (TokenKind::Lt, 1),
            },
            '>' => match bytes.get(i + 1) {
                Some(b'=') => (TokenKind::GtEq, 2),
                _ => (TokenKind::Gt, 1),
            },
            '!' => match bytes.get(i + 1) {
                Some(b'=') => (TokenKind::NotEq, 2),
                _ => {
                    return Err(Error::parse(format!(
                        "unexpected character '!' at byte {start} (did you mean '!='?)"
                    )))
                }
            },
            other => {
                return Err(Error::parse(format!("unexpected character {other:?} at byte {start}")))
            }
        };
        tokens.push(Token { kind, pos: start });
        i += width;
    }

    tokens.push(Token { kind: TokenKind::Eof, pos: input.len() });
    Ok(tokens)
}

//! A recursive-descent parser with Pratt (precedence-climbing) expression
//! parsing.
//!
//! Precedence climbing is used rather than a separate grammar rule per
//! precedence level because SQL has about ten levels: encoding them as
//! recursive productions means ten function calls to parse `1`, while a single
//! loop driven by a precedence table parses any expression in one pass.

use super::ast::*;
use super::token::{tokenize, Keyword, Token, TokenKind};
use crate::error::{Error, Result};
use crate::types::Value;

/// Parses a single SQL statement.
pub fn parse(sql: &str) -> Result<Statement> {
    let tokens = tokenize(sql)?;
    let mut p = Parser { tokens, pos: 0, src: sql };
    let stmt = p.parse_statement()?;
    p.skip_if(&TokenKind::Semicolon);
    p.expect_eof()?;
    Ok(stmt)
}

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    src: &'a str,
}

/// Binding powers for binary operators. Higher binds tighter.
fn precedence(kind: &TokenKind) -> Option<(u8, BinaryOp)> {
    use BinaryOp as B;
    Some(match kind {
        TokenKind::Keyword(Keyword::Or) => (1, B::Or),
        TokenKind::Keyword(Keyword::And) => (2, B::And),
        TokenKind::Eq => (3, B::Eq),
        TokenKind::NotEq => (3, B::NotEq),
        TokenKind::Lt => (3, B::Lt),
        TokenKind::LtEq => (3, B::LtEq),
        TokenKind::Gt => (3, B::Gt),
        TokenKind::GtEq => (3, B::GtEq),
        TokenKind::Plus => (5, B::Plus),
        TokenKind::Minus => (5, B::Minus),
        TokenKind::Star => (6, B::Multiply),
        TokenKind::Slash => (6, B::Divide),
        TokenKind::Percent => (6, B::Modulo),
        _ => return None,
    })
}

/// Precedence of the postfix predicates (IS NULL, IN, BETWEEN, LIKE), which sit
/// just above comparison so that `a = 1 AND b IN (2)` groups correctly.
const POSTFIX_PRECEDENCE: u8 = 4;

impl<'a> Parser<'a> {
    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    fn peek_at(&self, n: usize) -> &TokenKind {
        let i = (self.pos + n).min(self.tokens.len() - 1);
        &self.tokens[i].kind
    }

    fn advance(&mut self) -> TokenKind {
        let k = self.tokens[self.pos].kind.clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        k
    }

    fn skip_if(&mut self, kind: &TokenKind) -> bool {
        if self.peek() == kind {
            self.advance();
            true
        } else {
            false
        }
    }

    fn skip_keyword(&mut self, kw: Keyword) -> bool {
        self.skip_if(&TokenKind::Keyword(kw))
    }

    /// Reports an error pointing at the current token's position in the source.
    fn error(&self, msg: impl std::fmt::Display) -> Error {
        let tok = &self.tokens[self.pos];
        let (line, col) = line_col(self.src, tok.pos);
        Error::parse(format!("{msg} at line {line}, column {col} (found {})", tok.kind))
    }

    fn expect(&mut self, kind: TokenKind) -> Result<()> {
        if *self.peek() == kind {
            self.advance();
            Ok(())
        } else {
            Err(self.error(format!("expected {kind}")))
        }
    }

    fn expect_keyword(&mut self, kw: Keyword) -> Result<()> {
        self.expect(TokenKind::Keyword(kw))
    }

    fn expect_eof(&self) -> Result<()> {
        if matches!(self.peek(), TokenKind::Eof) {
            Ok(())
        } else {
            Err(self.error("expected end of statement"))
        }
    }

    fn parse_statement(&mut self) -> Result<Statement> {
        if self.skip_keyword(Keyword::Explain) {
            return Ok(Statement::Explain(Box::new(self.parse_query()?)));
        }
        Ok(Statement::Query(Box::new(self.parse_query()?)))
    }

    fn parse_query(&mut self) -> Result<Query> {
        self.expect_keyword(Keyword::Select)?;
        let distinct = self.skip_keyword(Keyword::Distinct);
        let projection = self.parse_select_list()?;

        let from =
            if self.skip_keyword(Keyword::From) { Some(self.parse_table_ref()?) } else { None };

        let selection =
            if self.skip_keyword(Keyword::Where) { Some(self.parse_expr(0)?) } else { None };

        let group_by = if self.skip_keyword(Keyword::Group) {
            self.expect_keyword(Keyword::By)?;
            let mut exprs = vec![self.parse_expr(0)?];
            while self.skip_if(&TokenKind::Comma) {
                exprs.push(self.parse_expr(0)?);
            }
            exprs
        } else {
            Vec::new()
        };

        let having =
            if self.skip_keyword(Keyword::Having) { Some(self.parse_expr(0)?) } else { None };

        let order_by = if self.skip_keyword(Keyword::Order) {
            self.expect_keyword(Keyword::By)?;
            let mut terms = vec![self.parse_order_term()?];
            while self.skip_if(&TokenKind::Comma) {
                terms.push(self.parse_order_term()?);
            }
            terms
        } else {
            Vec::new()
        };

        let limit =
            if self.skip_keyword(Keyword::Limit) { Some(self.parse_usize("LIMIT")?) } else { None };
        let offset = if self.skip_keyword(Keyword::Offset) {
            Some(self.parse_usize("OFFSET")?)
        } else {
            None
        };

        Ok(Query {
            distinct,
            projection,
            from,
            selection,
            group_by,
            having,
            order_by,
            limit,
            offset,
        })
    }

    fn parse_usize(&mut self, what: &str) -> Result<usize> {
        match self.peek().clone() {
            TokenKind::Int(v) if v >= 0 => {
                self.advance();
                Ok(v as usize)
            }
            _ => Err(self.error(format!("{what} requires a non-negative integer"))),
        }
    }

    fn parse_order_term(&mut self) -> Result<OrderByExpr> {
        let expr = self.parse_expr(0)?;
        let asc = if self.skip_keyword(Keyword::Desc) {
            false
        } else {
            self.skip_keyword(Keyword::Asc);
            true
        };
        Ok(OrderByExpr { expr, asc })
    }

    fn parse_select_list(&mut self) -> Result<Vec<SelectItem>> {
        let mut items = vec![self.parse_select_item()?];
        while self.skip_if(&TokenKind::Comma) {
            items.push(self.parse_select_item()?);
        }
        Ok(items)
    }

    fn parse_select_item(&mut self) -> Result<SelectItem> {
        if self.skip_if(&TokenKind::Star) {
            return Ok(SelectItem::Wildcard);
        }
        // `alias.*`
        if let (TokenKind::Ident(name), TokenKind::Dot, TokenKind::Star) =
            (self.peek().clone(), self.peek_at(1).clone(), self.peek_at(2).clone())
        {
            self.advance();
            self.advance();
            self.advance();
            return Ok(SelectItem::QualifiedWildcard(name));
        }

        let expr = self.parse_expr(0)?;
        let alias = if self.skip_keyword(Keyword::As) {
            Some(self.parse_ident()?)
        } else if let TokenKind::Ident(name) = self.peek().clone() {
            // An identifier directly after an expression is an implicit alias.
            self.advance();
            Some(name)
        } else {
            None
        };
        Ok(SelectItem::Expr { expr, alias })
    }

    fn parse_ident(&mut self) -> Result<String> {
        match self.advance() {
            TokenKind::Ident(s) => Ok(s),
            other => Err(Error::parse(format!("expected an identifier but found {other}"))),
        }
    }

    fn parse_table_ref(&mut self) -> Result<TableRef> {
        let mut left = self.parse_table_factor()?;

        loop {
            let join_type = if self.skip_keyword(Keyword::Cross) {
                self.expect_keyword(Keyword::Join)?;
                JoinType::Cross
            } else if self.skip_keyword(Keyword::Inner) {
                self.expect_keyword(Keyword::Join)?;
                JoinType::Inner
            } else if self.skip_keyword(Keyword::Left) {
                self.skip_keyword(Keyword::Outer);
                self.expect_keyword(Keyword::Join)?;
                JoinType::Left
            } else if self.skip_keyword(Keyword::Right) {
                self.skip_keyword(Keyword::Outer);
                self.expect_keyword(Keyword::Join)?;
                JoinType::Right
            } else if self.skip_keyword(Keyword::Join) {
                JoinType::Inner
            } else if self.skip_if(&TokenKind::Comma) {
                // `FROM a, b` is a cross join.
                JoinType::Cross
            } else {
                break;
            };

            let right = self.parse_table_factor()?;
            let on = if join_type == JoinType::Cross {
                None
            } else if self.skip_keyword(Keyword::On) {
                Some(self.parse_expr(0)?)
            } else {
                return Err(self.error(format!("{join_type} JOIN requires an ON clause")));
            };

            left = TableRef::Join { left: Box::new(left), right: Box::new(right), join_type, on };
        }
        Ok(left)
    }

    fn parse_table_factor(&mut self) -> Result<TableRef> {
        let name = self.parse_ident()?;
        let alias = if self.skip_keyword(Keyword::As) {
            Some(self.parse_ident()?)
        } else if let TokenKind::Ident(a) = self.peek().clone() {
            self.advance();
            Some(a)
        } else {
            None
        };
        Ok(TableRef::Table { name, alias })
    }

    /// Pratt expression parser. `min_bp` is the minimum binding power an
    /// operator must have to be absorbed into the current expression.
    fn parse_expr(&mut self, min_bp: u8) -> Result<Expr> {
        let mut lhs = self.parse_prefix()?;

        loop {
            // Postfix predicates bind tighter than AND/OR but looser than
            // comparison, so they are checked before the binary table.
            if POSTFIX_PRECEDENCE >= min_bp {
                if let Some(next) = self.try_parse_postfix(lhs.clone())? {
                    lhs = next;
                    continue;
                }
            }

            let Some((bp, op)) = precedence(self.peek()) else { break };
            if bp < min_bp {
                break;
            }
            self.advance();
            // Left-associative: the right operand must bind strictly tighter.
            let rhs = self.parse_expr(bp + 1)?;
            lhs = Expr::Binary { left: Box::new(lhs), op, right: Box::new(rhs) };
        }
        Ok(lhs)
    }

    /// Handles IS [NOT] NULL, [NOT] IN, [NOT] BETWEEN and [NOT] LIKE.
    ///
    /// Returns `None` when the next token starts none of them, leaving the
    /// position untouched.
    fn try_parse_postfix(&mut self, lhs: Expr) -> Result<Option<Expr>> {
        if self.skip_keyword(Keyword::Is) {
            let negated = self.skip_keyword(Keyword::Not);
            self.expect_keyword(Keyword::Null)?;
            return Ok(Some(Expr::IsNull { expr: Box::new(lhs), negated }));
        }

        // `NOT` here must be followed by IN/BETWEEN/LIKE to be a postfix form;
        // otherwise it belongs to a following prefix expression and must not be
        // consumed.
        let negated = match (self.peek(), self.peek_at(1)) {
            (
                TokenKind::Keyword(Keyword::Not),
                TokenKind::Keyword(Keyword::In | Keyword::Between | Keyword::Like),
            ) => {
                self.advance();
                true
            }
            _ => false,
        };

        if self.skip_keyword(Keyword::In) {
            self.expect(TokenKind::LParen)?;
            let mut list = Vec::new();
            if !matches!(self.peek(), TokenKind::RParen) {
                list.push(self.parse_expr(0)?);
                while self.skip_if(&TokenKind::Comma) {
                    list.push(self.parse_expr(0)?);
                }
            }
            self.expect(TokenKind::RParen)?;
            return Ok(Some(Expr::InList { expr: Box::new(lhs), list, negated }));
        }

        if self.skip_keyword(Keyword::Between) {
            // Parse the bounds above AND's precedence so that the AND
            // separating them is not mistaken for a boolean connective.
            let low = self.parse_expr(3)?;
            self.expect_keyword(Keyword::And)?;
            let high = self.parse_expr(3)?;
            return Ok(Some(Expr::Between {
                expr: Box::new(lhs),
                low: Box::new(low),
                high: Box::new(high),
                negated,
            }));
        }

        if self.skip_keyword(Keyword::Like) {
            let pattern = self.parse_expr(POSTFIX_PRECEDENCE + 1)?;
            return Ok(Some(Expr::Like {
                expr: Box::new(lhs),
                pattern: Box::new(pattern),
                negated,
            }));
        }

        if negated {
            // Unreachable given the lookahead above, but fail loudly rather
            // than silently dropping a NOT if that ever changes.
            return Err(self.error("dangling NOT"));
        }
        Ok(None)
    }

    fn parse_prefix(&mut self) -> Result<Expr> {
        match self.peek().clone() {
            TokenKind::Keyword(Keyword::Not) => {
                self.advance();
                let expr = self.parse_expr(3)?;
                Ok(Expr::Unary { op: UnaryOp::Not, expr: Box::new(expr) })
            }
            TokenKind::Minus => {
                self.advance();
                // Fold negation into numeric literals so that `-1` is a
                // constant rather than a negation node the optimizer must
                // later simplify.
                match self.peek().clone() {
                    TokenKind::Int(v) => {
                        self.advance();
                        Ok(Expr::Literal(Value::Int64(-v)))
                    }
                    TokenKind::Float(v) => {
                        self.advance();
                        Ok(Expr::Literal(Value::Float64(-v)))
                    }
                    _ => {
                        let expr = self.parse_expr(7)?;
                        Ok(Expr::Unary { op: UnaryOp::Negate, expr: Box::new(expr) })
                    }
                }
            }
            TokenKind::Plus => {
                self.advance();
                self.parse_prefix()
            }
            TokenKind::Int(v) => {
                self.advance();
                Ok(Expr::Literal(Value::Int64(v)))
            }
            TokenKind::Float(v) => {
                self.advance();
                Ok(Expr::Literal(Value::Float64(v)))
            }
            TokenKind::String(s) => {
                self.advance();
                Ok(Expr::Literal(Value::Utf8(s)))
            }
            TokenKind::Keyword(Keyword::True) => {
                self.advance();
                Ok(Expr::Literal(Value::Boolean(true)))
            }
            TokenKind::Keyword(Keyword::False) => {
                self.advance();
                Ok(Expr::Literal(Value::Boolean(false)))
            }
            TokenKind::Keyword(Keyword::Null) => {
                self.advance();
                Ok(Expr::Literal(Value::Null))
            }
            TokenKind::Keyword(Keyword::Case) => self.parse_case(),
            TokenKind::Keyword(Keyword::Cast) => self.parse_cast(),
            TokenKind::LParen => {
                self.advance();
                let e = self.parse_expr(0)?;
                self.expect(TokenKind::RParen)?;
                Ok(Expr::Nested(Box::new(e)))
            }
            TokenKind::Ident(name) => {
                self.advance();
                // Function call.
                if matches!(self.peek(), TokenKind::LParen) {
                    return self.parse_function_call(name);
                }
                // Qualified column.
                if matches!(self.peek(), TokenKind::Dot) {
                    self.advance();
                    let col = self.parse_ident()?;
                    return Ok(Expr::Column { qualifier: Some(name), name: col });
                }
                Ok(Expr::Column { qualifier: None, name })
            }
            _ => Err(self.error("expected an expression")),
        }
    }

    fn parse_function_call(&mut self, name: String) -> Result<Expr> {
        self.expect(TokenKind::LParen)?;
        let distinct = self.skip_keyword(Keyword::Distinct);

        if self.skip_if(&TokenKind::Star) {
            self.expect(TokenKind::RParen)?;
            return Ok(Expr::Function {
                name: name.to_ascii_uppercase(),
                args: Vec::new(),
                star: true,
                distinct,
            });
        }

        let mut args = Vec::new();
        if !matches!(self.peek(), TokenKind::RParen) {
            args.push(self.parse_expr(0)?);
            while self.skip_if(&TokenKind::Comma) {
                args.push(self.parse_expr(0)?);
            }
        }
        self.expect(TokenKind::RParen)?;
        Ok(Expr::Function { name: name.to_ascii_uppercase(), args, star: false, distinct })
    }

    fn parse_case(&mut self) -> Result<Expr> {
        self.expect_keyword(Keyword::Case)?;
        let mut when_then = Vec::new();
        while self.skip_keyword(Keyword::When) {
            let w = self.parse_expr(0)?;
            self.expect_keyword(Keyword::Then)?;
            let t = self.parse_expr(0)?;
            when_then.push((w, t));
        }
        if when_then.is_empty() {
            return Err(self.error("CASE requires at least one WHEN branch"));
        }
        let else_expr = if self.skip_keyword(Keyword::Else) {
            Some(Box::new(self.parse_expr(0)?))
        } else {
            None
        };
        self.expect_keyword(Keyword::End)?;
        Ok(Expr::Case { when_then, else_expr })
    }

    fn parse_cast(&mut self) -> Result<Expr> {
        self.expect_keyword(Keyword::Cast)?;
        self.expect(TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        self.expect_keyword(Keyword::As)?;
        let data_type = self.parse_ident()?;
        self.expect(TokenKind::RParen)?;
        Ok(Expr::Cast { expr: Box::new(expr), data_type: data_type.to_ascii_uppercase() })
    }
}

/// Converts a byte offset into a 1-based line and column, for error messages.
fn line_col(src: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in src.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::ast::Statement;

    /// Parses an expression by wrapping it in a trivial SELECT.
    fn expr(sql: &str) -> Expr {
        let stmt = parse(&format!("SELECT {sql}")).unwrap();
        let Statement::Query(q) = stmt else { panic!("expected a query") };
        match &q.projection[0] {
            SelectItem::Expr { expr, .. } => expr.clone(),
            other => panic!("expected an expression, got {other:?}"),
        }
    }

    /// Round-trips an expression through the parser and its Display impl, so
    /// that the parse tree's shape is visible as parenthesized text.
    fn shape(sql: &str) -> String {
        fn render(e: &Expr) -> String {
            match e {
                Expr::Binary { left, op, right } => {
                    format!("({} {} {})", render(left), op, render(right))
                }
                Expr::Unary { op, expr } => format!("({op}{})", render(expr)),
                Expr::Nested(inner) => render(inner),
                other => other.to_string(),
            }
        }
        render(&expr(sql))
    }

    #[test]
    fn arithmetic_precedence() {
        assert_eq!(shape("1 + 2 * 3"), "(1 + (2 * 3))");
        assert_eq!(shape("1 * 2 + 3"), "((1 * 2) + 3)");
        assert_eq!(shape("(1 + 2) * 3"), "((1 + 2) * 3)");
        // Left associativity: 10 - 3 - 2 is 5, not 9.
        assert_eq!(shape("10 - 3 - 2"), "((10 - 3) - 2)");
        assert_eq!(shape("8 / 4 / 2"), "((8 / 4) / 2)");
    }

    #[test]
    fn comparison_binds_tighter_than_and_which_binds_tighter_than_or() {
        assert_eq!(shape("a = 1 AND b = 2"), "((a = 1) AND (b = 2))");
        assert_eq!(shape("a OR b AND c"), "(a OR (b AND c))");
        assert_eq!(shape("a AND b OR c"), "((a AND b) OR c)");
        assert_eq!(shape("a + 1 > b * 2"), "((a + 1) > (b * 2))");
    }

    #[test]
    fn not_binds_looser_than_comparison() {
        // NOT a = 1 means NOT (a = 1), not (NOT a) = 1.
        assert_eq!(shape("NOT a = 1"), "(NOT (a = 1))");
    }

    #[test]
    fn negative_literals_are_folded_at_parse_time() {
        assert_eq!(expr("-5"), Expr::Literal(Value::Int64(-5)));
        assert_eq!(expr("-5.5"), Expr::Literal(Value::Float64(-5.5)));
        // But negating a column is a real unary operation.
        assert!(matches!(expr("-a"), Expr::Unary { .. }));
    }

    #[test]
    fn between_does_not_swallow_a_following_and() {
        // The AND inside BETWEEN must not be read as a boolean connective, or
        // `x BETWEEN 1 AND 2 AND y` parses as `x BETWEEN 1 AND (2 AND y)`.
        let e = expr("x BETWEEN 1 AND 2 AND y = 3");
        let Expr::Binary { left, op: BinaryOp::And, right } = &e else {
            panic!("expected a top-level AND, got {e:?}")
        };
        assert!(matches!(**left, Expr::Between { .. }), "left should be the BETWEEN");
        assert!(matches!(**right, Expr::Binary { .. }), "right should be y = 3");
    }

    #[test]
    fn not_before_in_between_like_is_a_postfix_negation() {
        assert!(matches!(expr("x NOT IN (1, 2)"), Expr::InList { negated: true, .. }));
        assert!(matches!(expr("x NOT BETWEEN 1 AND 2"), Expr::Between { negated: true, .. }));
        assert!(matches!(expr("x NOT LIKE 'a%'"), Expr::Like { negated: true, .. }));
        assert!(matches!(expr("x IS NOT NULL"), Expr::IsNull { negated: true, .. }));
        // A NOT not followed by one of those is an ordinary prefix operator.
        assert!(matches!(expr("NOT x"), Expr::Unary { .. }));
    }

    #[test]
    fn string_literals_handle_escapes_and_unicode() {
        assert_eq!(expr("'it''s'"), Expr::Literal(Value::Utf8("it's".into())));
        assert_eq!(expr("'héllo wörld'"), Expr::Literal(Value::Utf8("héllo wörld".into())));
        assert_eq!(expr("''"), Expr::Literal(Value::Utf8(String::new())));
    }

    #[test]
    fn numeric_literal_forms() {
        assert_eq!(expr("42"), Expr::Literal(Value::Int64(42)));
        assert_eq!(expr("4.25"), Expr::Literal(Value::Float64(4.25)));
        assert_eq!(expr("1e3"), Expr::Literal(Value::Float64(1000.0)));
        assert_eq!(expr("1.5e-2"), Expr::Literal(Value::Float64(0.015)));
        // An integer too large for i64 widens rather than failing to parse.
        assert!(matches!(expr("99999999999999999999"), Expr::Literal(Value::Float64(_))));
    }

    #[test]
    fn identifiers_quoted_and_qualified() {
        assert_eq!(expr("t.col"), Expr::Column { qualifier: Some("t".into()), name: "col".into() });
        assert_eq!(
            expr("\"weird name\""),
            Expr::Column { qualifier: None, name: "weird name".into() }
        );
        // A quoted identifier that looks like a keyword is still an identifier.
        assert_eq!(expr("\"select\""), Expr::Column { qualifier: None, name: "select".into() });
    }

    #[test]
    fn comments_are_skipped() {
        let stmt = parse("SELECT a -- trailing\nFROM t /* block */ WHERE b = 1").unwrap();
        let Statement::Query(q) = stmt else { panic!() };
        assert!(q.from.is_some());
        assert!(q.selection.is_some());
    }

    #[test]
    fn full_query_shape() {
        let stmt = parse(
            "SELECT DISTINCT a, b AS bee, COUNT(*) FROM t JOIN u ON t.k = u.k
             WHERE a > 1 GROUP BY a, b HAVING COUNT(*) > 2 ORDER BY a DESC, 2 LIMIT 10 OFFSET 5",
        )
        .unwrap();
        let Statement::Query(q) = stmt else { panic!() };
        assert!(q.distinct);
        assert_eq!(q.projection.len(), 3);
        assert_eq!(q.group_by.len(), 2);
        assert!(q.having.is_some());
        assert_eq!(q.order_by.len(), 2);
        assert!(!q.order_by[0].asc);
        assert!(q.order_by[1].asc, "ASC is the default");
        assert_eq!(q.limit, Some(10));
        assert_eq!(q.offset, Some(5));
    }

    #[test]
    fn explain_wraps_a_query() {
        assert!(matches!(parse("EXPLAIN SELECT 1").unwrap(), Statement::Explain(_)));
    }

    #[test]
    fn malformed_input_is_rejected_with_a_position() {
        for bad in [
            "SELECT",
            "SELECT FROM t",
            "SELECT a FROM",
            "SELECT a FROM t WHERE",
            "SELECT a FROM t GROUP a",
            "SELECT a FROM t ORDER a",
            "SELECT a FROM t LIMIT x",
            "SELECT a FROM t JOIN u",
            "SELECT 'unterminated",
            "SELECT \"unterminated",
            "SELECT /* unterminated",
            "SELECT a ! b",
            "SELECT CASE END",
            // `FROM t EXTRA` is a valid implicit alias; a second one is not.
            "SELECT a FROM t t2 t3",
            "SELECT a, FROM t",
            "SELECT a FROM t WHERE b IN",
        ] {
            let err = parse(bad).unwrap_err();
            assert!(matches!(err, Error::Parse(_)), "{bad:?} should be a parse error, got {err:?}");
        }
    }

    #[test]
    fn a_trailing_semicolon_is_accepted() {
        assert!(parse("SELECT 1;").is_ok());
        assert!(parse("  SELECT 1 ;  ").is_ok());
    }

    #[test]
    fn exponent_marker_does_not_swallow_an_identifier() {
        // `1east` must lex as the number 1 followed by the identifier `east`,
        // not as a malformed exponent.
        let stmt = parse("SELECT 1 east FROM t").unwrap();
        let Statement::Query(q) = stmt else { panic!() };
        match &q.projection[0] {
            SelectItem::Expr { expr, alias } => {
                assert_eq!(*expr, Expr::Literal(Value::Int64(1)));
                assert_eq!(alias.as_deref(), Some("east"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}

//! The Basalt SQL front end: lexer, syntax tree, parser and printer.
//!
//! Everything here is hand-written. There is no parser generator and no
//! `sqlparser` dependency, for one reason: **a parser is judged by its errors**,
//! and a generated one can only tell you that a token was unexpected. This one
//! tells you what it wanted, what it got, where, and — where the fix is
//! unambiguous — what to do about it.
//!
//! ```
//! use basalt_sql::parse;
//!
//! let err = parse("SELECT * FORM orders").unwrap_err();
//! println!("{}", err.render("SELECT * FORM orders"));
//! // error: expected FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT or end of
//! //        statement, found identifier `form`
//! //  --> 1:10
//! //   |
//! // 1 | SELECT * FORM orders
//! //   |          ^^^^
//! //   = hint: did you mean `FROM`?
//! assert_eq!(err.hint(), Some("did you mean `FROM`?"));
//! ```
//!
//! # The three pieces
//!
//! * [`tokenize`] turns text into [`Token`]s carrying byte [`Span`]s. Its
//!   lexical decisions — identifier folding, nested block comments, `''`
//!   escaping — are documented in [`lexer`].
//! * [`parse`] runs a recursive-descent parser whose expression grammar is a
//!   Pratt loop over the precedence table documented on
//!   [`Precedence`](ast::Precedence).
//! * [`Display`](std::fmt::Display) on the [`ast`] prints SQL back. The printer
//!   guarantees `parse(print(parse(sql))) == parse(sql)`, which is a property
//!   test, not a hope.
//!
//! # What the tree does and does not record
//!
//! The AST records *meaning*. Spans ride along for diagnostics but take no part
//! in equality; parentheses are not stored because the structure already says
//! where they were; a numeric literal keeps its exact spelling because that is
//! what decides its type. See [`ast`] for the reasoning.
//!
//! ```
//! use basalt_sql::{parse, ast::Statement};
//!
//! let sql = "select t.a, count(*) n from orders t \
//!            where t.status in ('open','held') group by t.a having count(*) > 2";
//! let statement = parse(sql)?;
//! assert!(matches!(statement, Statement::Query(_)));
//!
//! // Printing is round-trip exact.
//! assert_eq!(parse(&statement.to_string())?, statement);
//! # Ok::<(), basalt_sql::SqlError>(())
//! ```
//!
//! # Scope
//!
//! This crate answers "is this valid SQL, and what does it say". It does **not**
//! know what a table is: names are not resolved, types are not checked, `*` is
//! not expanded. That is [`basalt-plan`](https://docs.rs/basalt-plan)'s job, and
//! keeping the boundary sharp is what lets the parser be tested exhaustively
//! against text alone.
#![forbid(unsafe_code)]

pub mod ast;
pub mod display;
pub mod error;
pub mod lexer;
pub mod parser;
pub mod span;

pub use ast::{Expr, Statement};
pub use error::SqlError;
pub use lexer::{tokenize, Keyword, Parameter, Symbol, Token, TokenKind};
pub use parser::{parse, parse_expression, parse_many, ParseResult, Parser, MAX_RECURSION_DEPTH};
pub use span::{NodeSpan, Span};

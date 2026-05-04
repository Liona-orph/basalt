//! The tokenizer: SQL text in, spanned [`Token`]s out.
//!
//! # Lexical decisions
//!
//! Every dialect makes these choices differently, so they are stated here
//! rather than left to be discovered by reading the code.
//!
//! **Identifier folding.** A bare identifier is folded to *lower* case, a
//! `"quoted"` one is preserved verbatim. Basalt therefore behaves like
//! PostgreSQL and not like the ANSI standard (which folds to upper case): the
//! standard's choice makes `SELECT Name FROM t` look for a column called `NAME`,
//! which surprises everyone who did not read the standard. The consequence is
//! that `Foo`, `foo` and `"foo"` are the same column while `"Foo"` is a
//! different one.
//!
//! **Keywords.** Matched case-insensitively against a table of 90-odd words,
//! only for *bare* identifiers. `"select"` is always a column name. Keywords are
//! split into reserved and non-reserved (see [`Keyword::is_reserved`]); a
//! non-reserved keyword may be used as an identifier, which is what lets
//! `SELECT first, last FROM t` work even though `FIRST` and `LAST` appear in
//! `NULLS FIRST`.
//!
//! **Strings.** Single-quoted, with `''` as the escape for an embedded quote.
//! No backslash escapes: `'a\nb'` is four characters, `a`, `\`, `n`, `b`. This
//! is the standard behaviour and the one that does not silently reinterpret
//! Windows paths.
//!
//! **Comments.** `--` runs to the end of the line. `/* */` **nests**, so
//! `/* a /* b */ c */` is one comment. Nesting is the PostgreSQL behaviour and
//! the only one that lets you comment out a block of SQL that already contains
//! a comment; the cost is that an unbalanced `*/` inside a string-free comment
//! is a lexical error instead of silently ending it early. Nesting depth is
//! capped ([`MAX_COMMENT_NESTING`]) so that pathological input cannot make the
//! lexer loop for a long time.
//!
//! **Numbers.** `123`, `1.5`, `.5`, `1.`, `1e9`, `1.5E-3`. The lexer does not
//! interpret them; it hands the parser the original text (see
//! [`crate::ast::Literal::Number`]) because the *type* of a numeric literal is
//! a semantic question that belongs to the binder.
//!
//! **Parameters.** `?` is a placeholder numbered by order of appearance; `$1`
//! is an explicitly numbered one. Both become the same [`Parameter`] token, and
//! using both spellings in one statement is a *lexical error*: `?` counts only
//! `?`s, so `SELECT ?, $2, ?` would give two placeholders the index 2.

use std::fmt;

use crate::error::{did_you_mean, SqlError};
use crate::span::Span;

/// Maximum nesting depth of `/* ... */` comments.
///
/// A comment nested deeper than this is almost certainly not a comment; the cap
/// turns a potential denial of service into a diagnostic.
pub const MAX_COMMENT_NESTING: usize = 64;

macro_rules! keywords {
    ($($variant:ident => $text:literal, $reserved:literal;)*) => {
        /// A SQL keyword.
        ///
        /// The variants are the closed set the parser knows about; anything else
        /// that looks like a word is an identifier.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[allow(missing_docs, reason = "one line of doc per keyword variant would say only the keyword")]
        pub enum Keyword {
            $($variant,)*
        }

        impl Keyword {
            /// Every keyword, in declaration order. Used by "did you mean"
            /// suggestions and by the round-trip tests.
            pub const ALL: &'static [Keyword] = &[$(Keyword::$variant,)*];

            /// The canonical upper-case spelling.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Keyword::$variant => $text,)*
                }
            }

            /// `true` if the word may **not** be used as a bare identifier.
            ///
            /// Reserving a word is a cost paid by every user who has a column
            /// called that, so the set is kept as small as the grammar allows:
            /// a word is reserved only when allowing it as an identifier would
            /// make a statement ambiguous rather than merely awkward to parse.
            pub const fn is_reserved(self) -> bool {
                match self {
                    $(Keyword::$variant => $reserved,)*
                }
            }

            /// Look up a keyword by its **upper-cased** spelling.
            pub fn from_uppercase(word: &str) -> Option<Keyword> {
                Some(match word {
                    $($text => Keyword::$variant,)*
                    _ => return None,
                })
            }
        }
    };
}

keywords! {
    All => "ALL", true;
    Analyze => "ANALYZE", false;
    And => "AND", true;
    As => "AS", true;
    Asc => "ASC", false;
    Between => "BETWEEN", true;
    By => "BY", true;
    Case => "CASE", true;
    Cast => "CAST", true;
    Create => "CREATE", true;
    Cross => "CROSS", true;
    Current => "CURRENT", false;
    Default => "DEFAULT", true;
    Desc => "DESC", false;
    Distinct => "DISTINCT", true;
    Drop => "DROP", true;
    Else => "ELSE", true;
    End => "END", true;
    Escape => "ESCAPE", false;
    Except => "EXCEPT", true;
    Exists => "EXISTS", true;
    Explain => "EXPLAIN", true;
    Extract => "EXTRACT", false;
    False => "FALSE", true;
    First => "FIRST", false;
    Following => "FOLLOWING", false;
    From => "FROM", true;
    Full => "FULL", true;
    Group => "GROUP", true;
    Having => "HAVING", true;
    If => "IF", false;
    Ilike => "ILIKE", true;
    In => "IN", true;
    Inner => "INNER", true;
    Insert => "INSERT", true;
    Intersect => "INTERSECT", true;
    Into => "INTO", true;
    Is => "IS", true;
    Join => "JOIN", true;
    Key => "KEY", false;
    Last => "LAST", false;
    Left => "LEFT", true;
    Like => "LIKE", true;
    Limit => "LIMIT", true;
    Natural => "NATURAL", true;
    Not => "NOT", true;
    Null => "NULL", true;
    Nulls => "NULLS", false;
    Offset => "OFFSET", true;
    On => "ON", true;
    Or => "OR", true;
    Order => "ORDER", true;
    Outer => "OUTER", true;
    Over => "OVER", false;
    Partition => "PARTITION", false;
    Preceding => "PRECEDING", false;
    Primary => "PRIMARY", false;
    Range => "RANGE", false;
    Recursive => "RECURSIVE", false;
    Replace => "REPLACE", false;
    Right => "RIGHT", true;
    Row => "ROW", false;
    Rows => "ROWS", false;
    Select => "SELECT", true;
    Table => "TABLE", true;
    Then => "THEN", true;
    True => "TRUE", true;
    Unbounded => "UNBOUNDED", false;
    Union => "UNION", true;
    Unique => "UNIQUE", false;
    Using => "USING", true;
    Values => "VALUES", true;
    Verbose => "VERBOSE", false;
    View => "VIEW", false;
    When => "WHEN", true;
    Where => "WHERE", true;
    With => "WITH", true;
}

impl fmt::Display for Keyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single-character or multi-character operator / punctuation token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Symbol {
    /// `(`
    LeftParen,
    /// `)`
    RightParen,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `.`
    Period,
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
    /// `||`
    Concat,
    /// `=`
    Eq,
    /// `<>` or `!=` — the two spellings are folded into one token because they
    /// mean exactly the same thing and keeping them apart would force every
    /// consumer to handle both.
    NotEq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
}

impl Symbol {
    /// The canonical spelling. `NotEq` prints as `<>`, the standard form.
    pub const fn as_str(self) -> &'static str {
        match self {
            Symbol::LeftParen => "(",
            Symbol::RightParen => ")",
            Symbol::Comma => ",",
            Symbol::Semicolon => ";",
            Symbol::Period => ".",
            Symbol::Star => "*",
            Symbol::Plus => "+",
            Symbol::Minus => "-",
            Symbol::Slash => "/",
            Symbol::Percent => "%",
            Symbol::Concat => "||",
            Symbol::Eq => "=",
            Symbol::NotEq => "<>",
            Symbol::Lt => "<",
            Symbol::LtEq => "<=",
            Symbol::Gt => ">",
            Symbol::GtEq => ">=",
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A bind parameter, identified by its 1-based index.
///
/// Both spellings produce the same token: `?` is resolved to its ordinal by the
/// lexer, so the *n*-th `?` is exactly `$n`. Keeping them apart in the token
/// would be recording provenance that has no semantic content — and would break
/// the printer, which has to emit a position-independent form (`$n`) because a
/// printed fragment may be re-parsed without the `?`s that preceded it.
///
/// Mixing the two spellings in one statement is a lexical error (see
/// [`tokenize`]): with `SELECT ?, $2, ?` the second `?` and `$2` would silently
/// be the same parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Parameter(pub u32);

impl Parameter {
    /// The 1-based parameter index.
    pub const fn index(self) -> u32 {
        self.0
    }
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${}", self.0)
    }
}

/// What a [`Token`] is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    /// A recognised keyword (always from a bare, case-folded word).
    Keyword(Keyword),
    /// An identifier. Bare identifiers arrive lower-cased; quoted ones verbatim.
    Ident(String),
    /// A numeric literal, in its original spelling.
    Number(String),
    /// A string literal, already unescaped (`''` collapsed to `'`).
    String(String),
    /// A bind parameter placeholder.
    Parameter(Parameter),
    /// An operator or punctuation mark.
    Symbol(Symbol),
    /// The virtual token at the end of the input.
    Eof,
}

impl TokenKind {
    /// How to refer to this token in an error message.
    ///
    /// Deliberately concrete — "keyword `FROM`" rather than "token" — because
    /// the whole point of the message is to tell the reader where to look.
    pub fn describe(&self) -> String {
        match self {
            TokenKind::Keyword(k) => format!("keyword `{k}`"),
            TokenKind::Ident(name) => format!("identifier `{name}`"),
            TokenKind::Number(text) => format!("number `{text}`"),
            TokenKind::String(_) => String::from("a string literal"),
            TokenKind::Parameter(p) => format!("parameter `{p}`"),
            TokenKind::Symbol(s) => format!("`{s}`"),
            TokenKind::Eof => String::from("end of input"),
        }
    }
}

/// A token and the byte range it occupies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// What kind of token this is.
    pub kind: TokenKind,
    /// Where it came from.
    pub span: Span,
}

impl Token {
    /// Pair a kind with a span.
    pub const fn new(kind: TokenKind, span: Span) -> Token {
        Token { kind, span }
    }
}

/// Split `sql` into tokens, ending with exactly one [`TokenKind::Eof`].
///
/// Whitespace and comments are consumed and not represented: nothing downstream
/// needs them, and a token stream that contains trivia forces every consumer to
/// filter it. The `Eof` token is real rather than an `Option`, so the parser can
/// always report a span for "unexpected end of input".
///
/// ```
/// use basalt_sql::{tokenize, TokenKind, Keyword};
///
/// let tokens = tokenize("SELECT 1 -- hi")?;
/// assert_eq!(tokens[0].kind, TokenKind::Keyword(Keyword::Select));
/// assert_eq!(tokens[1].kind, TokenKind::Number("1".into()));
/// assert_eq!(tokens[2].kind, TokenKind::Eof);
/// # Ok::<(), basalt_sql::SqlError>(())
/// ```
pub fn tokenize(sql: &str) -> Result<Vec<Token>, SqlError> {
    Lexer::new(sql).tokenize()
}

/// The tokenizer state machine.
///
/// Kept as a struct rather than a free function because the positional
/// parameter counter and the byte cursor have to travel together.
#[derive(Debug)]
struct Lexer<'a> {
    source: &'a str,
    bytes: &'a [u8],
    pos: usize,
    positional: u32,
    /// Span of the first `?`, if one has been seen.
    saw_anonymous: Option<Span>,
    /// Span of the first `$n`, if one has been seen.
    saw_numbered: Option<Span>,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Lexer<'a> {
        Lexer {
            source,
            bytes: source.as_bytes(),
            pos: 0,
            positional: 0,
            saw_anonymous: None,
            saw_numbered: None,
        }
    }

    fn tokenize(mut self) -> Result<Vec<Token>, SqlError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_trivia()?;
            let start = self.pos;
            let Some(byte) = self.peek() else {
                tokens.push(Token::new(TokenKind::Eof, Span::at(start)));
                return Ok(tokens);
            };
            let kind = match byte {
                b'\'' => self.lex_string()?,
                b'"' => self.lex_quoted_ident()?,
                b'?' => {
                    self.pos += 1;
                    self.positional += 1;
                    let span = Span::new(start, self.pos);
                    self.saw_anonymous.get_or_insert(span);
                    self.check_parameter_styles(span)?;
                    TokenKind::Parameter(Parameter(self.positional))
                }
                b'$' => self.lex_numbered_parameter()?,
                b'0'..=b'9' => self.lex_number()?,
                b'.' if self.peek_at(1).is_some_and(|b| b.is_ascii_digit()) => self.lex_number()?,
                b if is_ident_start(b) => self.lex_word(),
                _ => self.lex_symbol()?,
            };
            tokens.push(Token::new(kind, Span::new(start, self.pos)));
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos + offset).copied()
    }

    /// Consume whitespace and comments until the next real token.
    fn skip_trivia(&mut self) -> Result<(), SqlError> {
        loop {
            match self.peek() {
                Some(b) if b.is_ascii_whitespace() => self.pos += 1,
                Some(b'-') if self.peek_at(1) == Some(b'-') => {
                    self.pos += 2;
                    while let Some(b) = self.peek() {
                        if b == b'\n' {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                Some(b'/') if self.peek_at(1) == Some(b'*') => self.skip_block_comment()?,
                _ => return Ok(()),
            }
        }
    }

    fn skip_block_comment(&mut self) -> Result<(), SqlError> {
        let start = self.pos;
        let mut depth = 0usize;
        loop {
            match (self.peek(), self.peek_at(1)) {
                (Some(b'/'), Some(b'*')) => {
                    depth += 1;
                    if depth > MAX_COMMENT_NESTING {
                        return Err(SqlError::new(
                            Span::new(start, self.pos + 2),
                            format!("block comments nested more than {MAX_COMMENT_NESTING} deep"),
                        )
                        .with_hint("this is almost always an unbalanced `/*`; close it"));
                    }
                    self.pos += 2;
                }
                (Some(b'*'), Some(b'/')) => {
                    self.pos += 2;
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                (Some(_), _) => self.pos += 1,
                (None, _) => {
                    return Err(SqlError::new(
                        Span::new(start, self.source.len()),
                        "unterminated block comment",
                    )
                    .with_hint("add `*/`; remember that Basalt nests block comments, so each `/*` needs its own `*/`"));
                }
            }
        }
    }

    /// A bare word: identifier or keyword.
    fn lex_word(&mut self) -> TokenKind {
        let start = self.pos;
        while self.peek().is_some_and(is_ident_continue) {
            self.pos += 1;
        }
        let raw = &self.source[start..self.pos];
        let upper = raw.to_ascii_uppercase();
        match Keyword::from_uppercase(&upper) {
            Some(kw) => TokenKind::Keyword(kw),
            None => TokenKind::Ident(raw.to_ascii_lowercase()),
        }
    }

    fn lex_quoted_ident(&mut self) -> Result<TokenKind, SqlError> {
        let start = self.pos;
        self.pos += 1; // opening quote
        let mut value = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(SqlError::new(
                        Span::new(start, self.source.len()),
                        "unterminated quoted identifier",
                    )
                    .with_hint("add a closing `\"`; to put a `\"` inside the name, double it"));
                }
                Some(b'"') if self.peek_at(1) == Some(b'"') => {
                    value.push('"');
                    self.pos += 2;
                }
                Some(b'"') => {
                    self.pos += 1;
                    break;
                }
                Some(_) => {
                    let ch = self.source[self.pos..].chars().next().unwrap_or('\0');
                    value.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
        if value.is_empty() {
            return Err(SqlError::new(
                Span::new(start, self.pos),
                "a quoted identifier must not be empty",
            )
            .with_hint("`\"\"` names no column; delete it or put a name between the quotes"));
        }
        Ok(TokenKind::Ident(value))
    }

    fn lex_string(&mut self) -> Result<TokenKind, SqlError> {
        let start = self.pos;
        self.pos += 1;
        let mut value = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(SqlError::new(
                        Span::new(start, self.source.len()),
                        "unterminated string literal",
                    )
                    .with_hint(
                        "add a closing `'`; to put a `'` inside the text, double it (`'it''s'`)",
                    ));
                }
                Some(b'\'') if self.peek_at(1) == Some(b'\'') => {
                    value.push('\'');
                    self.pos += 2;
                }
                Some(b'\'') => {
                    self.pos += 1;
                    break;
                }
                Some(_) => {
                    let ch = self.source[self.pos..].chars().next().unwrap_or('\0');
                    value.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
        Ok(TokenKind::String(value))
    }

    fn lex_numbered_parameter(&mut self) -> Result<TokenKind, SqlError> {
        let start = self.pos;
        self.pos += 1; // `$`
        let digits_start = self.pos;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        let digits = &self.source[digits_start..self.pos];
        if digits.is_empty() {
            return Err(SqlError::new(
                Span::new(start, self.pos + 1),
                "expected a digit after `$`",
            )
            .with_hint("bind parameters are written `$1`, `$2`, ... or `?`"));
        }
        let n: u32 = digits.parse().map_err(|_| {
            SqlError::new(
                Span::new(start, self.pos),
                format!("parameter number `{digits}` is too large"),
            )
        })?;
        if n == 0 {
            return Err(
                SqlError::new(Span::new(start, self.pos), "parameter numbers start at 1")
                    .with_hint("write `$1` for the first parameter"),
            );
        }
        let span = Span::new(start, self.pos);
        self.saw_numbered.get_or_insert(span);
        self.check_parameter_styles(span)?;
        Ok(TokenKind::Parameter(Parameter(n)))
    }

    /// Reject `?` and `$n` in the same statement.
    ///
    /// `?` is numbered by counting `?`s, so `SELECT ?, $2, ?` would give the
    /// second `?` the index 2 as well and quietly bind one value to two places.
    /// Refusing the mixture is cheaper than explaining the collision later.
    fn check_parameter_styles(&self, span: Span) -> Result<(), SqlError> {
        if self.saw_anonymous.is_some() && self.saw_numbered.is_some() {
            return Err(SqlError::new(
                span,
                "a statement may use `?` placeholders or `$n` placeholders, but not both",
            )
            .with_hint(
                "`?` is numbered by position, so mixing the forms lets two placeholders collide",
            ));
        }
        Ok(())
    }

    /// `digits [ '.' digits ] [ ('e'|'E') ['+'|'-'] digits ]`, also allowing a
    /// leading `.`.
    fn lex_number(&mut self) -> Result<TokenKind, SqlError> {
        let start = self.pos;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            // Only commit to an exponent if it is actually well formed;
            // otherwise `1e` is a number followed by the identifier `e`, which
            // is what a user writing `SELECT 1e FROM t` would expect to be told.
            let sign = usize::from(matches!(self.peek_at(1), Some(b'+' | b'-')));
            if self.peek_at(1 + sign).is_some_and(|b| b.is_ascii_digit()) {
                self.pos += 1 + sign;
                while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                    self.pos += 1;
                }
            }
        }
        // `1abc` is not a number followed by an identifier in any SQL dialect;
        // catching it here gives a much better message than "expected FROM".
        if self.peek().is_some_and(is_ident_start) {
            let bad_start = self.pos;
            while self.peek().is_some_and(is_ident_continue) {
                self.pos += 1;
            }
            return Err(SqlError::new(
                Span::new(start, self.pos),
                format!(
                    "`{}` is not a valid number: digits are followed by `{}`",
                    &self.source[start..self.pos],
                    &self.source[bad_start..self.pos]
                ),
            )
            .with_hint(
                "separate the number from the word, or quote the word if it is an identifier",
            ));
        }
        Ok(TokenKind::Number(self.source[start..self.pos].to_string()))
    }

    fn lex_symbol(&mut self) -> Result<TokenKind, SqlError> {
        let start = self.pos;
        let two =
            |a: u8, b: u8, this: &Lexer<'_>| this.peek() == Some(a) && this.peek_at(1) == Some(b);
        let symbol = if two(b'<', b'=', self) {
            self.pos += 2;
            Symbol::LtEq
        } else if two(b'>', b'=', self) {
            self.pos += 2;
            Symbol::GtEq
        } else if two(b'<', b'>', self) || two(b'!', b'=', self) {
            self.pos += 2;
            Symbol::NotEq
        } else if two(b'|', b'|', self) {
            self.pos += 2;
            Symbol::Concat
        } else if two(b':', b':', self) {
            self.pos += 2;
            return Err(
                SqlError::new(Span::new(start, self.pos), "`::` is not supported")
                    .with_hint("write `CAST(expr AS type)` instead"),
            );
        } else {
            let byte = self.peek().unwrap_or(b'\0');
            let simple = match byte {
                b'(' => Symbol::LeftParen,
                b')' => Symbol::RightParen,
                b',' => Symbol::Comma,
                b';' => Symbol::Semicolon,
                b'.' => Symbol::Period,
                b'*' => Symbol::Star,
                b'+' => Symbol::Plus,
                b'-' => Symbol::Minus,
                b'/' => Symbol::Slash,
                b'%' => Symbol::Percent,
                b'=' => Symbol::Eq,
                b'<' => Symbol::Lt,
                b'>' => Symbol::Gt,
                _ => return Err(self.unexpected_character(start)),
            };
            self.pos += 1;
            simple
        };
        Ok(TokenKind::Symbol(symbol))
    }

    fn unexpected_character(&mut self, start: usize) -> SqlError {
        let ch = self.source[start..].chars().next().unwrap_or('\0');
        self.pos = start + ch.len_utf8();
        let hint = match ch {
            '!' => Some(String::from("did you mean `!=`?")),
            '|' => Some(String::from("did you mean `||` (string concatenation)?")),
            '&' => Some(String::from("SQL spells conjunction `AND`")),
            '#' | '`' => Some(String::from(
                "quote identifiers with double quotes: \"my column\"",
            )),
            '\\' => Some(String::from(
                "Basalt does not use backslash escapes; double a quote to embed it",
            )),
            _ => None,
        };
        SqlError::new(
            Span::new(start, self.pos),
            format!("unexpected character `{ch}`"),
        )
        .with_optional_hint(hint)
    }
}

/// `true` if `b` may start a bare identifier.
///
/// Bytes >= 0x80 are accepted so that identifiers may contain non-ASCII letters;
/// because UTF-8 continuation bytes are all >= 0x80, testing bytes rather than
/// characters is sufficient and avoids decoding on the hot path.
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b >= 0x80
}

/// `true` if `b` may continue a bare identifier.
fn is_ident_continue(b: u8) -> bool {
    is_ident_start(b) || b.is_ascii_digit() || b == b'$'
}

/// The closest keyword to `word`, for "did you mean" hints.
pub(crate) fn closest_keyword(word: &str) -> Option<&'static str> {
    did_you_mean(word, Keyword::ALL.iter().map(|k| k.as_str()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn kinds(sql: &str) -> Vec<TokenKind> {
        tokenize(sql).unwrap().into_iter().map(|t| t.kind).collect()
    }

    fn err(sql: &str) -> SqlError {
        tokenize(sql).unwrap_err()
    }

    #[test]
    fn keywords_are_case_insensitive_but_identifiers_fold_down() {
        assert_eq!(
            kinds("SeLeCt Foo"),
            vec![
                TokenKind::Keyword(Keyword::Select),
                TokenKind::Ident("foo".into()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn quoted_identifiers_keep_case_and_unescape() {
        assert_eq!(
            kinds(r#""Foo" "a""b" "select""#),
            vec![
                TokenKind::Ident("Foo".into()),
                TokenKind::Ident("a\"b".into()),
                TokenKind::Ident("select".into()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn strings_use_doubled_quotes_and_no_backslash_escapes() {
        assert_eq!(
            kinds(r"'it''s' 'a\nb'"),
            vec![
                TokenKind::String("it's".into()),
                TokenKind::String(r"a\nb".into()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn numbers_in_every_shape() {
        assert_eq!(
            kinds("1 1.5 .5 1. 1e9 1.5E-3 0"),
            vec![
                TokenKind::Number("1".into()),
                TokenKind::Number("1.5".into()),
                TokenKind::Number(".5".into()),
                TokenKind::Number("1.".into()),
                TokenKind::Number("1e9".into()),
                TokenKind::Number("1.5E-3".into()),
                TokenKind::Number("0".into()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn a_dangling_exponent_is_a_number_then_a_word() {
        assert_eq!(
            kinds("1 e"),
            vec![
                TokenKind::Number("1".into()),
                TokenKind::Ident("e".into()),
                TokenKind::Eof
            ]
        );
        let e = err("1e");
        assert!(e.message().contains("not a valid number"), "{e}");
    }

    #[test]
    fn multi_character_operators() {
        assert_eq!(
            kinds("<= >= <> != || < > = + - * / %"),
            vec![
                TokenKind::Symbol(Symbol::LtEq),
                TokenKind::Symbol(Symbol::GtEq),
                TokenKind::Symbol(Symbol::NotEq),
                TokenKind::Symbol(Symbol::NotEq),
                TokenKind::Symbol(Symbol::Concat),
                TokenKind::Symbol(Symbol::Lt),
                TokenKind::Symbol(Symbol::Gt),
                TokenKind::Symbol(Symbol::Eq),
                TokenKind::Symbol(Symbol::Plus),
                TokenKind::Symbol(Symbol::Minus),
                TokenKind::Symbol(Symbol::Star),
                TokenKind::Symbol(Symbol::Slash),
                TokenKind::Symbol(Symbol::Percent),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn parameters_are_numbered() {
        assert_eq!(
            kinds("? ? ?"),
            vec![
                TokenKind::Parameter(Parameter(1)),
                TokenKind::Parameter(Parameter(2)),
                TokenKind::Parameter(Parameter(3)),
                TokenKind::Eof
            ]
        );
        assert_eq!(
            kinds("$3 $1"),
            vec![
                TokenKind::Parameter(Parameter(3)),
                TokenKind::Parameter(Parameter(1)),
                TokenKind::Eof
            ]
        );
        assert!(err("$0").message().contains("start at 1"));
        assert!(err("$x").message().contains("expected a digit after `$`"));
        assert!(err("$99999999999").message().contains("too large"));
        let mixed = err("? $2");
        assert!(mixed.message().contains("but not both"), "{mixed}");
        assert_eq!(mixed.span(), Span::new(2, 4));
        assert!(err("$2 ?").message().contains("but not both"));
    }

    #[test]
    fn comments_nest() {
        assert_eq!(
            kinds("1 /* a /* b */ c */ 2 -- trailing\n3"),
            vec![
                TokenKind::Number("1".into()),
                TokenKind::Number("2".into()),
                TokenKind::Number("3".into()),
                TokenKind::Eof
            ]
        );
        let e = err("1 /* a /* b */");
        assert!(e.message().contains("unterminated block comment"), "{e}");
        assert_eq!(e.span().start, 2);
        let deep = "/*".repeat(MAX_COMMENT_NESTING + 1);
        assert!(err(&deep).message().contains("nested more than"));
    }

    #[test]
    fn spans_cover_exactly_the_token() {
        let sql = "SELECT  foo";
        let tokens = tokenize(sql).unwrap();
        assert_eq!(tokens[0].span, Span::new(0, 6));
        assert_eq!(tokens[1].span, Span::new(8, 11));
        assert_eq!(tokens[2].span, Span::at(11));
        assert_eq!(tokens[1].span.slice(sql), "foo");
    }

    #[test]
    fn lexical_errors_carry_useful_hints() {
        let e = err("a ! b");
        assert!(e.message().contains("unexpected character `!`"), "{e}");
        assert_eq!(e.span(), Span::new(2, 3));
        assert_eq!(e.hint(), Some("did you mean `!=`?"));

        assert!(err("a & b").hint().unwrap().contains("AND"));
        assert!(err("x::int").message().contains("`::` is not supported"));
        assert!(err("'oops").message().contains("unterminated string"));
        assert!(err("\"oops").message().contains("unterminated quoted"));
        assert!(err("\"\"").message().contains("must not be empty"));
    }

    #[test]
    fn non_ascii_identifiers_are_allowed() {
        assert_eq!(
            kinds("naïve_π"),
            vec![TokenKind::Ident("naïve_π".into()), TokenKind::Eof]
        );
    }

    #[test]
    fn keyword_table_is_consistent() {
        for kw in Keyword::ALL {
            assert_eq!(Keyword::from_uppercase(kw.as_str()), Some(*kw));
            assert_eq!(kw.to_string(), kw.as_str());
            assert!(
                kw.as_str().chars().all(|c| c.is_ascii_uppercase()),
                "{kw} must be spelled in upper case"
            );
        }
        assert!(Keyword::Select.is_reserved());
        assert!(!Keyword::First.is_reserved());
        assert_eq!(Keyword::from_uppercase("NOPE"), None);
    }

    #[test]
    fn describe_names_the_token() {
        assert_eq!(
            TokenKind::Keyword(Keyword::From).describe(),
            "keyword `FROM`"
        );
        assert_eq!(TokenKind::Ident("a".into()).describe(), "identifier `a`");
        assert_eq!(TokenKind::Number("1".into()).describe(), "number `1`");
        assert_eq!(TokenKind::String("a".into()).describe(), "a string literal");
        assert_eq!(TokenKind::Symbol(Symbol::Comma).describe(), "`,`");
        assert_eq!(TokenKind::Eof.describe(), "end of input");
        assert_eq!(
            TokenKind::Parameter(Parameter(2)).describe(),
            "parameter `$2`"
        );
        assert_eq!(Parameter(7).index(), 7);
    }

    #[test]
    fn empty_input_is_just_eof() {
        assert_eq!(kinds(""), vec![TokenKind::Eof]);
        assert_eq!(kinds("   \n\t -- x\n"), vec![TokenKind::Eof]);
    }
}

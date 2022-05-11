use crate::edn::{Edn, Error, List, Map, Set, Vector};

pub(crate) fn tokenize(edn: &str) -> std::iter::Enumerate<std::str::Chars> {
    edn.chars().enumerate()
}

pub(crate) fn parse(
    c: Option<(usize, char)>,
    chars: &mut std::iter::Enumerate<std::str::Chars>,
) -> Result<Edn, Error> {
    Ok(match c {
        Some((_, '[')) => read_vec(chars)?,
        Some((_, '(')) => read_list(chars)?,
        Some((_, '@')) => read_set(chars)?,
        Some((_, '{')) => read_map(chars)?,
        edn => parse_edn(edn, chars)?,
    })
}

pub(crate) fn parse_edn(
    c: Option<(usize, char)>,
    chars: &mut std::iter::Enumerate<std::str::Chars>,
) -> Result<Edn, Error> {
    match c {
        Some((_, '\"')) => read_str(chars),
        Some((_, ':')) => read_key_or_nsmap(chars),
        Some((_, '#')) => Ok(read_tagged(chars)?),
        Some((_, '-')) => Ok(read_number('-', chars)?),
        Some((_, '\\')) => Ok(read_char(chars)?),
        Some((_, b)) if b == 't' || b == 'f' || b == 'n' => Ok(read_bool_or_nil(b, chars)?),
        Some((_, n)) if n.is_numeric() => Ok(read_number(n, chars)?),
        Some((_, a)) => Ok(read_symbol(a, chars)?),
        None => Err(Error::ParseEdn("Edn could not be parsed".to_string())),
    }
}

fn read_key_or_nsmap(chars: &mut std::iter::Enumerate<std::str::Chars>) -> Result<Edn, Error> {
    let mut key_chars = chars.clone().take_while(|c| {
        !c.1.is_whitespace() && c.1 != ',' && c.1 != ')' && c.1 != ']' && c.1 != '}'
    });
    let c_len = key_chars.clone().count();

    Ok(match key_chars.find(|c| c.1 == '{') {
        Some(_) => read_namespaced_map(chars)?,
        None => read_key(chars, c_len),
    })
}

fn read_key(chars: &mut std::iter::Enumerate<std::str::Chars>, c_len: usize) -> Edn {
    let mut key = String::from(":");
    let key_chars = chars.take(c_len).map(|c| c.1).collect::<String>();
    key.push_str(&key_chars);
    Edn::Key(key)
}

fn read_str(chars: &mut std::iter::Enumerate<std::str::Chars>) -> Result<Edn, Error> {
    let result = chars.try_fold(
        (false, String::new()),
        |(last_was_escape, mut s), (_, c)| {
            if last_was_escape {
                // Supported escape characters, per https://github.com/edn-format/edn#strings
                match c {
                    't' => s.push('\t'),
                    'r' => s.push('\r'),
                    'n' => s.push('\n'),
                    '\\' => s.push('\\'),
                    '\"' => s.push('\"'),
                    _ => {
                        return Err(Err(Error::ParseEdn(format!(
                            "Invalid escape sequence \\{}",
                            c
                        ))))
                    }
                };

                Ok((false, s))
            } else if c == '\"' {
                // Unescaped quote means we're done
                Err(Ok(s))
            } else if c == '\\' {
                Ok((true, s))
            } else {
                s.push(c);
                Ok((false, s))
            }
        },
    );

    match result {
        // An Ok means we actually finished parsing *without* seeing the end of the string, so that's
        // an error.
        Ok(_) => Err(Error::ParseEdn("Unterminated string".to_string())),
        Err(Err(e)) => Err(e),
        Err(Ok(string)) => Ok(Edn::Str(string)),
    }
}

fn read_symbol(a: char, chars: &mut std::iter::Enumerate<std::str::Chars>) -> Result<Edn, Error> {
    let c_len = chars
        .clone()
        .enumerate()
        .take_while(|&(i, c)| {
            i <= 200 && !c.1.is_whitespace() && c.1 != ')' && c.1 != '}' && c.1 != ']'
        })
        .count();
    let i = chars
        .clone()
        .next()
        .ok_or_else(|| Error::ParseEdn("Could not identify symbol index".to_string()))?
        .0;

    if a.is_whitespace() {
        return Err(Error::ParseEdn(format!(
            "1\"{}\" could not be parsed at char count {}",
            a, i
        )));
    }

    let mut symbol = String::from(a);
    let symbol_chars = chars.take(c_len).map(|c| c.1).collect::<String>();
    symbol.push_str(&symbol_chars);
    Ok(Edn::Symbol(symbol))
}

fn read_tagged(chars: &mut std::iter::Enumerate<std::str::Chars>) -> Result<Edn, Error> {
    let tag = chars
        .take_while(|c| !c.1.is_whitespace())
        .map(|c| c.1)
        .collect::<String>();

    if tag.starts_with("inst") {
        return Ok(Edn::Inst(
            chars
                .skip_while(|c| c.1 == '\"' || c.1.is_whitespace())
                .take_while(|c| c.1 != '\"')
                .map(|c| c.1)
                .collect::<String>(),
        ));
    }

    if tag.starts_with("uuid") {
        return Ok(Edn::Uuid(
            chars
                .skip_while(|c| c.1 == '\"' || c.1.is_whitespace())
                .take_while(|c| c.1 != '\"')
                .map(|c| c.1)
                .collect::<String>(),
        ));
    }

    let next_char = chars.next();
    let content = read_tagged_chars(next_char, chars);
    let mut next_chars = content.chars().enumerate();

    Ok(Edn::Tagged(
        tag,
        Box::new(parse(next_chars.next(), &mut next_chars)?),
    ))
}

fn read_tagged_chars(
    c: Option<(usize, char)>,
    chars: &mut std::iter::Enumerate<std::str::Chars>,
) -> String {
    match c {
        Some((_, '[')) => format!(
            "[ {} ]",
            chars
                .take_while(|ch| ch.1 != ']')
                .map(|ch| ch.1)
                .collect::<String>()
        ),
        Some((_, '(')) => format!(
            "( {} )",
            chars
                .take_while(|ch| ch.1 != ')')
                .map(|ch| ch.1)
                .collect::<String>()
        ),
        Some((_, '{')) => format!(
            "{{ {} }}",
            chars
                .take_while(|ch| ch.1 != '}')
                .map(|ch| ch.1)
                .collect::<String>()
        ),
        Some((_, '@')) => format!(
            "@{{ {} }}",
            chars
                .take_while(|ch| ch.1 != '}')
                .map(|ch| ch.1)
                .collect::<String>()
        ),
        Some((_, '\"')) => format!(
            "\" {} \"",
            chars
                .take_while(|ch| ch.1 != '\"')
                .map(|ch| ch.1)
                .collect::<String>()
        ),
        _ => chars
            .take_while(|ch| !ch.1.is_whitespace())
            .map(|ch| ch.1)
            .collect::<String>(),
    }
}

fn read_number(n: char, chars: &mut std::iter::Enumerate<std::str::Chars>) -> Result<Edn, Error> {
    let i = chars
        .clone()
        .next()
        .ok_or_else(|| Error::ParseEdn("Could not identify symbol index".to_string()))?
        .0;
    let c_len = chars
        .clone()
        .take_while(|c| c.1.is_numeric() || c.1 == '.' || c.1 == '/')
        .count();
    let mut number = String::new();
    let string = chars.take(c_len).map(|c| c.1).collect::<String>();
    number.push(n);
    number.push_str(&string);

    match number {
        n if n.parse::<usize>().is_ok() => Ok(Edn::UInt(n.parse::<usize>()?)),
        n if n.parse::<isize>().is_ok() => Ok(Edn::Int(n.parse::<isize>()?)),
        n if n.parse::<f64>().is_ok() => Ok(Edn::Double(n.parse::<f64>()?.into())),
        n if n.contains('/') && n.split('/').all(|d| d.parse::<f64>().is_ok()) => {
            Ok(Edn::Rational(n))
        }
        _ => Err(Error::ParseEdn(format!(
            "2 {} could not be parsed at char count {}",
            number, i
        ))),
    }
}

fn read_char(chars: &mut std::iter::Enumerate<std::str::Chars>) -> Result<Edn, Error> {
    let i = chars
        .clone()
        .next()
        .ok_or_else(|| Error::ParseEdn("Could not identify symbol index".to_string()))?
        .0;
    let c = chars.next();
    c.ok_or(format!("3 {:?} could not be parsed at char count {}", c, i))
        .map(|c| c.1)
        .map(Edn::Char)
        .map_err(Error::ParseEdn)
}

fn read_bool_or_nil(
    c: char,
    chars: &mut std::iter::Enumerate<std::str::Chars>,
) -> Result<Edn, Error> {
    let i = chars
        .clone()
        .next()
        .ok_or_else(|| Error::ParseEdn("Could not identify symbol index".to_string()))?
        .0;
    match c.clone() {
        't' if {
            let val = chars.clone().take(4).map(|c| c.1).collect::<String>();
            val.eq("rue ")
                || val.eq("rue,")
                || val.eq("rue]")
                || val.eq("rue}")
                || val.eq("rue)")
                || val.eq("rue")
        } =>
        {
            let mut string = String::new();
            let t = chars.take(3).map(|c| c.1).collect::<String>();
            string.push(c);
            string.push_str(&t);
            Ok(Edn::Bool(string.parse::<bool>()?))
        }
        'f' if {
            let val = chars.clone().take(5).map(|c| c.1).collect::<String>();
            val.eq("alse ")
                || val.eq("alse,")
                || val.eq("alse]")
                || val.eq("alse}")
                || val.eq("alse)")
                || val.eq("alse")
        } =>
        {
            let mut string = String::new();
            let f = chars.take(4).map(|c| c.1).collect::<String>();
            string.push(c);
            string.push_str(&f);
            Ok(Edn::Bool(string.parse::<bool>()?))
        }
        'n' if {
            let val = chars.clone().take(3).map(|c| c.1).collect::<String>();
            val.eq("il ")
                || val.eq("il,")
                || val.eq("il]")
                || val.eq("il}")
                || val.eq("il)")
                || val.eq("il")
        } =>
        {
            let mut string = String::new();
            let n = chars.take(2).map(|c| c.1).collect::<String>();
            string.push(c);
            string.push_str(&n);
            match &string[..] {
                "nil" => Ok(Edn::Nil),
                _ => Err(Error::ParseEdn(format!(
                    "4 {} could not be parsed at char count {}",
                    string, i
                ))),
            }
        }
        _ => read_symbol(c, chars),
    }
}

fn read_vec(chars: &mut std::iter::Enumerate<std::str::Chars>) -> Result<Edn, Error> {
    let i = chars
        .clone()
        .next()
        .ok_or_else(|| Error::ParseEdn("Could not identify symbol index".to_string()))?
        .0;
    let mut res: Vec<Edn> = vec![];
    loop {
        match chars.next() {
            Some((_, ']')) => return Ok(Edn::Vector(Vector::new(res))),
            Some(c) if !c.1.is_whitespace() && c.1 != ',' => {
                res.push(parse(Some(c), chars)?);
            }
            Some(c) if c.1.is_whitespace() || c.1 == ',' => (),
            err => {
                return Err(Error::ParseEdn(format!(
                    "5 {:?} could not be parsed at char count {}",
                    err, i
                )))
            }
        }
    }
}

fn read_list(chars: &mut std::iter::Enumerate<std::str::Chars>) -> Result<Edn, Error> {
    let i = chars
        .clone()
        .next()
        .ok_or_else(|| Error::ParseEdn("Could not identify symbol index".to_string()))?
        .0;
    let mut res: Vec<Edn> = vec![];
    loop {
        match chars.next() {
            Some((_, ')')) => return Ok(Edn::List(List::new(res))),
            Some(c) if !c.1.is_whitespace() && c.1 != ',' => {
                res.push(parse(Some(c), chars)?);
            }
            Some(c) if c.1.is_whitespace() || c.1 == ',' => (),
            err => {
                return Err(Error::ParseEdn(format!(
                    "6 {:?} could not be parsed at char count {}",
                    err, i
                )))
            }
        }
    }
}

fn read_set(chars: &mut std::iter::Enumerate<std::str::Chars>) -> Result<Edn, Error> {
    let i = chars
        .clone()
        .next()
        .ok_or_else(|| Error::ParseEdn("Could not identify symbol index".to_string()))?
        .0;
    use std::collections::BTreeSet;
    let mut res: BTreeSet<Edn> = BTreeSet::new();
    loop {
        match chars.next() {
            Some((_, '}')) => return Ok(Edn::Set(Set::new(res))),
            Some(c) if !c.1.is_whitespace() && c.1 != ',' => {
                res.insert(parse(Some(c), chars)?);
            }
            Some(c) if c.1.is_whitespace() || c.1 == ',' => (),
            err => {
                return Err(Error::ParseEdn(format!(
                    "7 {:?} could not be parsed at char count {}",
                    err, i
                )))
            }
        }
    }
}

fn read_namespaced_map(chars: &mut std::iter::Enumerate<std::str::Chars>) -> Result<Edn, Error> {
    let i = chars
        .clone()
        .next()
        .ok_or_else(|| Error::ParseEdn("Could not identify symbol index".to_string()))?
        .0;
    use std::collections::BTreeMap;
    let mut res: BTreeMap<String, Edn> = BTreeMap::new();
    let mut key: Option<Edn> = None;
    let mut val: Option<Edn> = None;
    let namespace = chars
        .take_while(|c| c.1 != '{')
        .map(|c| c.1)
        .collect::<String>();

    loop {
        match chars.next() {
            Some((_, '}')) => return Ok(Edn::NamespacedMap(namespace, Map::new(res))),
            Some(c) if !c.1.is_whitespace() && c.1 != ',' => {
                if key.is_some() {
                    val = Some(parse(Some(c), chars)?);
                } else {
                    key = Some(parse(Some(c), chars)?);
                }
            }
            Some(c) if c.1.is_whitespace() || c.1 == ',' => (),
            err => {
                return Err(Error::ParseEdn(format!(
                    "8 {:?} could not be parsed at char count {}",
                    err, i
                )))
            }
        }

        if key.is_some() && val.is_some() {
            res.insert(key.unwrap().to_string(), val.unwrap());
            key = None;
            val = None;
        }
    }
}

fn read_map(chars: &mut std::iter::Enumerate<std::str::Chars>) -> Result<Edn, Error> {
    let i = chars
        .clone()
        .next()
        .ok_or_else(|| Error::ParseEdn("Could not identify symbol index".to_string()))?
        .0;
    use std::collections::BTreeMap;
    let mut res: BTreeMap<String, Edn> = BTreeMap::new();
    let mut key: Option<Edn> = None;
    let mut val: Option<Edn> = None;
    loop {
        match chars.next() {
            Some((_, '}')) => return Ok(Edn::Map(Map::new(res))),
            Some(c) if !c.1.is_whitespace() && c.1 != ',' => {
                if key.is_some() {
                    val = Some(parse(Some(c), chars)?);
                } else {
                    key = Some(parse(Some(c), chars)?);
                }
            }
            Some(c) if c.1.is_whitespace() || c.1 == ',' => (),
            err => {
                return Err(Error::ParseEdn(format!(
                    "9 {:?} could not be parsed at char count {}",
                    err, i
                )))
            }
        }

        if key.is_some() && val.is_some() {
            res.insert(key.unwrap().to_string(), val.unwrap());
            key = None;
            val = None;
        }
    }
}

use std::borrow::Cow;

pub trait MaybeReplaceExt<'a> {
    fn maybe_replace(self, find: &str, replacement: &str) -> Cow<'a, str>;
}

impl<'a> MaybeReplaceExt<'a> for &'a str {
    fn maybe_replace(self, find: &str, replacement: &str) -> Cow<'a, str> {
        if self.contains(find) {
            self.replace(find, replacement).into()
        } else {
            self.into()
        }
    }
}

impl<'a> MaybeReplaceExt<'a> for Cow<'a, str> {
    fn maybe_replace(self, find: &str, replacement: &str) -> Cow<'a, str> {
        if self.contains(find) {
            self.replace(find, replacement).into()
        } else {
            self
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::edn::{Double, Map, Set};
    use crate::{map, set};

    #[test]
    fn parse_keyword() {
        let mut key = ":keyword".chars().enumerate();

        assert_eq!(
            parse_edn(key.next(), &mut key).unwrap(),
            Edn::Key(":keyword".to_string())
        )
    }

    #[test]
    fn parse_str() {
        let mut string = "\"hello world, from      RUST\"".chars().enumerate();

        assert_eq!(
            parse_edn(string.next(), &mut string).unwrap(),
            Edn::Str("hello world, from      RUST".to_string())
        )
    }

    #[test]
    fn parse_str_with_escaped_characters() {
        let mut string = r##""hello\n \r \t \"world\" with escaped \\ characters""##
            .chars()
            .enumerate();

        assert_eq!(
            parse_edn(string.next(), &mut string).unwrap(),
            Edn::Str("hello\n \r \t \"world\" with escaped \\ characters".to_string())
        )
    }

    #[test]
    fn parse_str_with_invalid_escape() {
        let mut string = r##""hello\n \r \t \"world\" with escaped \\ \g characters""##
            .chars()
            .enumerate();

        assert_eq!(
            parse_edn(string.next(), &mut string),
            Err(Error::ParseEdn("Invalid escape sequence \\g".to_string()))
        )
    }

    #[test]
    fn parse_unterminated_string() {
        let mut string = r##""hello\n \r \t \"world\" with escaped \\ characters"##
            .chars()
            .enumerate();

        assert_eq!(
            parse_edn(string.next(), &mut string),
            Err(Error::ParseEdn("Unterminated string".to_string()))
        )
    }

    #[test]
    fn parse_number() {
        let mut uint = "143".chars().enumerate();
        let mut int = "-435143".chars().enumerate();
        let mut f = "-43.5143".chars().enumerate();
        let mut r = "43/5143".chars().enumerate();
        assert_eq!(parse_edn(uint.next(), &mut uint).unwrap(), Edn::UInt(143));
        assert_eq!(parse_edn(int.next(), &mut int).unwrap(), Edn::Int(-435143));
        assert_eq!(
            parse_edn(f.next(), &mut f).unwrap(),
            Edn::Double(Double::from(-43.5143))
        );
        assert_eq!(
            parse_edn(r.next(), &mut r).unwrap(),
            Edn::Rational("43/5143".to_string())
        );
    }

    #[test]
    fn parse_char() {
        let mut c = "\\k".chars().enumerate();

        assert_eq!(parse_edn(c.next(), &mut c).unwrap(), Edn::Char('k'))
    }

    #[test]
    fn parse_bool_or_nil() {
        let mut t = "true".chars().enumerate();
        let mut f = "false".chars().enumerate();
        let mut n = "nil".chars().enumerate();
        let mut s = "\"true\"".chars().enumerate();
        assert_eq!(parse_edn(t.next(), &mut t).unwrap(), Edn::Bool(true));
        assert_eq!(parse_edn(f.next(), &mut f).unwrap(), Edn::Bool(false));
        assert_eq!(parse_edn(n.next(), &mut n).unwrap(), Edn::Nil);
        assert_eq!(
            parse_edn(s.next(), &mut s).unwrap(),
            Edn::Str("true".to_string())
        );
    }

    #[test]
    fn parse_simple_vec() {
        let mut edn = "[11 \"2\" 3.3 :b true \\c]".chars().enumerate();

        assert_eq!(
            parse(edn.next(), &mut edn).unwrap(),
            Edn::Vector(Vector::new(vec![
                Edn::UInt(11),
                Edn::Str("2".to_string()),
                Edn::Double(3.3.into()),
                Edn::Key(":b".to_string()),
                Edn::Bool(true),
                Edn::Char('c')
            ]))
        );
    }

    #[test]
    fn parse_list() {
        let mut edn = "(1 \"2\" 3.3 :b )".chars().enumerate();

        assert_eq!(
            parse(edn.next(), &mut edn).unwrap(),
            Edn::List(List::new(vec![
                Edn::UInt(1),
                Edn::Str("2".to_string()),
                Edn::Double(3.3.into()),
                Edn::Key(":b".to_string()),
            ]))
        );
    }

    #[test]
    fn parse_set() {
        let mut edn = "@true \\c 3 }".chars().enumerate();

        assert_eq!(
            parse(edn.next(), &mut edn).unwrap(),
            Edn::Set(Set::new(set![
                Edn::Bool(true),
                Edn::Char('c'),
                Edn::UInt(3)
            ]))
        )
    }

    #[test]
    fn parse_complex() {
        let mut edn = "[:b ( 5 \\c @true \\c 3 } ) ]".chars().enumerate();

        assert_eq!(
            parse(edn.next(), &mut edn).unwrap(),
            Edn::Vector(Vector::new(vec![
                Edn::Key(":b".to_string()),
                Edn::List(List::new(vec![
                    Edn::UInt(5),
                    Edn::Char('c'),
                    Edn::Set(Set::new(set![
                        Edn::Bool(true),
                        Edn::Char('c'),
                        Edn::UInt(3)
                    ]))
                ]))
            ]))
        )
    }

    #[test]
    fn parse_simple_map() {
        let mut edn = "{:a \"2\" :b false :c nil }".chars().enumerate();

        assert_eq!(
            parse(edn.next(), &mut edn).unwrap(),
            Edn::Map(Map::new(
                map! {":a".to_string() => Edn::Str("2".to_string()),
                ":b".to_string() => Edn::Bool(false), ":c".to_string() => Edn::Nil}
            ))
        );
    }

    #[test]
    fn parse_inst() {
        let mut edn = "{:date  #inst \"2020-07-16T21:53:14.628-00:00\"}"
            .chars()
            .enumerate();

        assert_eq!(
            parse(edn.next(), &mut edn).unwrap(),
            Edn::Map(Map::new(map! {
                ":date".to_string() => Edn::Inst("2020-07-16T21:53:14.628-00:00".to_string())
            }))
        )
    }

    #[test]
    fn parse_edn_with_inst() {
        let mut edn = "@ :a :b {:c :d :date  #inst \"2020-07-16T21:53:14.628-00:00\" ::c ::d} nil}"
            .chars()
            .enumerate();

        assert_eq!(
            parse(edn.next(), &mut edn).unwrap(),
            Edn::Set(Set::new(set! {
                Edn::Key(":a".to_string()),
                Edn::Key(":b".to_string()),
                Edn::Map(Map::new(map! {
                    ":c".to_string() => Edn::Key(":d".to_string()),
                    ":date".to_string() => Edn::Inst("2020-07-16T21:53:14.628-00:00".to_string()),
                    "::c".to_string() => Edn::Key("::d".to_string())
                })),
                Edn::Nil
            }))
        )
    }

    #[test]
    fn parse_tagged_int() {
        let mut edn = "#iasdf 234".chars().enumerate();
        let res = parse(edn.next(), &mut edn).unwrap();

        assert_eq!(
            res,
            Edn::Tagged(String::from("iasdf"), Box::new(Edn::UInt(234)))
        )
    }

    #[test]
    fn parse_map_keyword_with_commas() {
        let mut edn = "{ :a :something, :b false, :c nil, }".chars().enumerate();

        assert_eq!(
            parse(edn.next(), &mut edn).unwrap(),
            Edn::Map(Map::new(
                map! {":a".to_string() => Edn::Key(":something".to_string()),
                ":b".to_string() => Edn::Bool(false), ":c".to_string() => Edn::Nil}
            ))
        );
    }

    #[test]
    fn parse_map_with_special_char_str1() {
        let mut edn = "{ :a \"hello\n \r \t \\\"world\\\" with escaped \\\\ characters\" }"
            .chars()
            .enumerate();

        assert_eq!(
            parse(edn.next(), &mut edn).unwrap(),
            Edn::Map(Map::new(
                map! {":a".to_string() => Edn::Str("hello\n \r \t \"world\" with escaped \\ characters".to_string())}
            ))
        );
    }

    #[test]
    fn parse_tagged_vec() {
        let mut edn = "#domain/model [1 2 3]".chars().enumerate();
        let res = parse(edn.next(), &mut edn).unwrap();

        assert_eq!(
            res,
            Edn::Tagged(
                String::from("domain/model"),
                Box::new(Edn::Vector(Vector::new(vec![
                    Edn::UInt(1),
                    Edn::UInt(2),
                    Edn::UInt(3)
                ])))
            )
        )
    }

    #[test]
    fn parse_map_with_tagged_vec() {
        let mut edn = "{ :model #domain/model [1 2 3] :int 2 }"
            .chars()
            .enumerate();
        let res = parse(edn.next(), &mut edn).unwrap();

        assert_eq!(
            res,
            Edn::Map(Map::new(map! {
                ":int".to_string() => Edn::UInt(2),
                ":model".to_string() => Edn::Tagged(
                String::from("domain/model"),
                Box::new(Edn::Vector(Vector::new(vec![
                    Edn::UInt(1),
                    Edn::UInt(2),
                    Edn::UInt(3)
                ])))
            )}))
        )
    }
}

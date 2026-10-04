//! `quarry` -- an interactive shell and one-shot query runner.

use quarry::{Context, Result};
use std::io::{self, BufRead, Write};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "\
quarry - a vectorized analytical SQL engine

usage:
  quarry [-t name=file.csv]... [-c 'SQL']
  quarry [-t name=file.csv]...            start an interactive shell

options:
  -t, --table name=path   register a CSV file as a table (repeatable)
  -c, --command SQL       run one statement and exit
  -h, --help              show this message

shell commands:
  .tables                 list registered tables
  .schema [table]         show column types
  .timing on|off          report query durations
  .quit                   exit
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("quarry: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut ctx = Context::new();
    let mut command: Option<String> = None;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            "-t" | "--table" => {
                i += 1;
                let spec = args.get(i).ok_or_else(|| {
                    quarry::Error::Io("-t requires an argument of the form name=path".into())
                })?;
                let (name, path) = spec.split_once('=').ok_or_else(|| {
                    quarry::Error::Io(format!("malformed --table {spec:?}, expected name=path"))
                })?;
                ctx.register_csv(name, path)?;
                eprintln!("registered table {name:?} from {path}");
            }
            "-c" | "--command" => {
                i += 1;
                command = Some(
                    args.get(i)
                        .ok_or_else(|| quarry::Error::Io("-c requires a SQL statement".into()))?
                        .clone(),
                );
            }
            other => {
                return Err(quarry::Error::Io(format!("unknown argument {other:?}\n\n{USAGE}")))
            }
        }
        i += 1;
    }

    if let Some(sql) = command {
        return run_one(&ctx, &sql, false);
    }
    shell(ctx)
}

fn run_one(ctx: &Context, sql: &str, timing: bool) -> Result<()> {
    let start = Instant::now();
    let result = ctx.sql(sql)?;
    let elapsed = start.elapsed();

    println!("{}", result.to_table());
    if result.explain_text().is_none() {
        let n = result.num_rows();
        print!("({n} row{})", if n == 1 { "" } else { "s" });
        if timing {
            print!(" in {:.3?}", elapsed);
        }
        println!();
    }
    Ok(())
}

fn shell(ctx: Context) -> Result<()> {
    println!("quarry -- type .help for commands, .quit to exit");
    if ctx.table_names().is_empty() {
        println!("no tables registered; start with -t name=file.csv");
    }

    let stdin = io::stdin();
    let mut timing = false;
    // Statements may span lines; accumulate until a semicolon or a blank line.
    let mut buffer = String::new();

    loop {
        print!("{}", if buffer.is_empty() { "quarry> " } else { "   ...> " });
        io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            return Ok(());
        }
        let trimmed = line.trim();

        if buffer.is_empty() && trimmed.starts_with('.') {
            match handle_dot_command(&ctx, trimmed, &mut timing) {
                DotResult::Quit => return Ok(()),
                DotResult::Handled => continue,
                DotResult::Unknown => {
                    println!("unknown command {trimmed:?}; try .help");
                    continue;
                }
            }
        }

        if trimmed.is_empty() && buffer.trim().is_empty() {
            buffer.clear();
            continue;
        }

        buffer.push_str(&line);
        // A blank line also submits, so a user who forgets the semicolon is
        // not stuck in continuation mode.
        let submit = trimmed.ends_with(';') || trimmed.is_empty();
        if !submit {
            continue;
        }

        let sql = buffer.trim().trim_end_matches(';').to_string();
        buffer.clear();
        if sql.is_empty() {
            continue;
        }
        // An error in the shell is reported, not fatal.
        if let Err(e) = run_one(&ctx, &sql, timing) {
            println!("{e}");
        }
    }
}

enum DotResult {
    Handled,
    Quit,
    Unknown,
}

fn handle_dot_command(ctx: &Context, cmd: &str, timing: &mut bool) -> DotResult {
    let mut parts = cmd.split_whitespace();
    match parts.next() {
        Some(".quit") | Some(".exit") => DotResult::Quit,
        Some(".help") => {
            print!("{USAGE}");
            DotResult::Handled
        }
        Some(".tables") => {
            let names = ctx.table_names();
            if names.is_empty() {
                println!("(no tables registered)");
            } else {
                for n in names {
                    println!("{n}");
                }
            }
            DotResult::Handled
        }
        Some(".schema") => {
            let targets: Vec<String> = match parts.next() {
                Some(t) => vec![t.to_string()],
                None => ctx.table_names(),
            };
            for t in targets {
                match ctx.schema_of(&t) {
                    Some(s) => {
                        println!("{t}:");
                        for f in s.fields() {
                            println!("  {:<20} {}", f.name, f.data_type);
                        }
                    }
                    None => println!("no table named {t:?}"),
                }
            }
            DotResult::Handled
        }
        Some(".timing") => {
            *timing = parts.next().map(|v| v == "on").unwrap_or(!*timing);
            println!("timing {}", if *timing { "on" } else { "off" });
            DotResult::Handled
        }
        _ => DotResult::Unknown,
    }
}

//! Binary shim around the library: argument parsing and I/O only, no logic
//! of its own — everything decision-shaped stays in lib.rs where it's
//! testable without a terminal.

use {{crate_snake}}::greet;

fn main() {
    // Mock arg handling — swap for `clap` (derive) when real flags arrive.
    let name = std::env::args().nth(1).unwrap_or_else(|| "world".into());
    println!("{}", greet(&name));
}

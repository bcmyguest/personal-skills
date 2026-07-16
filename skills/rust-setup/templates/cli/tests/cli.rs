//! End-to-end test of the real binary. No test dependencies needed: Cargo
//! hands integration tests the built binary's path via `CARGO_BIN_EXE_<name>`.

use std::process::Command;

#[test]
fn binary_greets_the_given_name() {
    let output = Command::new(env!("CARGO_BIN_EXE_{{crate_name}}"))
        .arg("Ferris")
        .output()
        .expect("binary should run");
    assert!(output.status.success());
    assert_eq!(String::from_utf8_lossy(&output.stdout), "Hello, Ferris!\n");
}

#[test]
fn binary_defaults_to_world() {
    let output = Command::new(env!("CARGO_BIN_EXE_{{crate_name}}"))
        .output()
        .expect("binary should run");
    assert!(output.status.success());
    assert_eq!(String::from_utf8_lossy(&output.stdout), "Hello, world!\n");
}

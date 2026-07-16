//! {{description}}
//!
//! Mock hello-world API — replace with real code, keeping the shape: logic
//! lives in the library where it's testable, and every public item carries a
//! unit test next to it.

/// Returns the greeting for `name`.
pub fn greet(name: &str) -> String {
    format!("Hello, {name}!")
}

#[cfg(test)]
mod tests {
    use super::greet;

    #[test]
    fn greets_by_name() {
        assert_eq!(greet("world"), "Hello, world!");
    }
}

// Embed a version string that identifies the exact commit when proseg is built
// from a git checkout that isn't a tagged release, e.g. "3.2.0 (v3.2.0-88-g46b4426)",
// or "3.2.0 (v3.2.0-88-g46b4426-dirty)" with uncommitted changes. A build of the
// tagged release, or one without git (e.g. installed from crates.io), reports
// just the Cargo.toml version.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn main() {
    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap();
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");

    // Only trust git if the repository is this crate's own, not some enclosing
    // repository that happens to contain an unpacked crate.
    let is_own_repo = git(&["rev-parse", "--show-toplevel"])
        .and_then(|top| Path::new(&top).canonicalize().ok())
        .is_some_and(|top| manifest_dir.canonicalize().ok() == Some(top));

    let describe = if is_own_repo {
        git(&["describe", "--tags", "--always", "--dirty", "--match", "v*"])
    } else {
        None
    };

    let version = match describe {
        Some(describe) if describe != format!("v{pkg_version}") => {
            format!("{pkg_version} ({describe})")
        }
        _ => pkg_version,
    };
    println!("cargo:rustc-env=PROSEG_VERSION={version}");

    if is_own_repo {
        // Re-run when HEAD moves, tags/branches change, or the working tree's
        // dirty state might have changed.
        for path in ["HEAD", "index", "refs", "packed-refs"] {
            if let Some(p) = git(&["rev-parse", "--git-path", path]) {
                println!("cargo:rerun-if-changed={p}");
            }
        }
        println!("cargo:rerun-if-changed=src");
        println!("cargo:rerun-if-changed=Cargo.toml");
    }
}

# AGENTS.md

## Commands

```bash
cargo build --release          # build (required for benchmarks/running the binary)
cargo test                     # run all tests
cargo test <name>              # run a single test
cargo clippy                   # lint
cargo clippy --fix             # lint + auto-fix
cargo build --release --features dhat-heap  # heap profiling build
./debug/example.sh             # smoke-test with a small Xenium dataset
```

- Unit tests live in `#[cfg(test)]` modules next to the code they test (find them with `grep -rl '#\[cfg(test)\]' src`); most are in `src/sampler/voxelcheckerboard.rs` and `src/sampler/sparsevec.rs`.
- `cargo test` is expected to build and pass. If a change breaks a test, fix the test or the code in the same change rather than leaving the suite broken.
- `polyagamma::tests::ks_test_against_reference` is `#[ignore]`d by default; run it with `cargo test --release ks_test_against_reference -- --ignored`.
- No CI for tests/lint — only a docker-publish workflow on tag push, so nothing else will catch a broken suite.

## Architecture

- Single Rust crate (edition 2024, MSRV 1.88.0) producing two binaries:
  - `proseg` (entry: `src/main.rs`) — the main segmentation tool
  - `proseg-to-baysor` (entry: `src/to_baysor.rs`) — converts proseg spatialdata zarr to Baysor format
- Core sampling code lives under `src/sampler/`; output/schemas/spatialdata in root `src/`.
- Reading input and writing output happens outside the sampler module.

## Key source files

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI (clap derive), platform presets, main orchestration loop |
| `src/sampler/transcripts.rs` | Transcript datasets, CSV/parquet/zarr/Visium reading |
| `src/sampler/voxelcheckerboard.rs` | Voxel grid init from priors, resolution changes, cell polygon extraction |
| `src/sampler/voxelsampler.rs` | Voxel→cell assignment updates (checkerboard pattern) |
| `src/sampler/paramsampler.rs` | Parameter sampling (volumes, expression, rates) |
| `src/sampler/paramoptimizer.rs` | Parameter optimization (EM) used for the point estimate |
| `src/sampler/transcriptrepo.rs` | Transcript repositioning (diffusion) |
| `src/sampler.rs` | `ModelParams`, `ModelPriors`, shard constants |
| `src/spatialdata_output.rs` | Zarr spatialdata output |
| `src/output.rs` | CSV/parquet/geojson output |
| `src/schemas.rs` | Arrow schemas for output tables |
| `src/sampler/csrmat.rs` | CSR sparse matrix (primary count storage) |
| `src/sampler/sparsevec.rs` | B+-tree sparse count vector backing each `CSRMat` row |
| `src/sampler/transitionmat.rs` | State transition counts (`--record-state-transitions`) |

## Gotchas

- **Parquet gene column must use plain string encoding, not dictionary.** Dictionary-encoded int32 indices break Dask round-trips when gene codes exceed 127. See `debug/SOLUTION.md`.
- **Memory exhaustion manifests as silent crashes or extreme slowdowns.** Increasing voxel size or reducing voxel layers reduces memory.
- **Burnin voxel size must be an integer multiple of final voxel size**, validated in `main.rs` (panics with "Ratio between --burnin-voxel-size and --voxel-size must an integer").
- **`BACKGROUND_CELL = u32::MAX`** — this sentinel appears throughout the codebase. Don't confuse it with cell index 0.
- **f32 throughout** for performance; careful ordering may matter for numerics.
- **`--overwrite` is required** when the zarr output directory already exists, or proseg panics.
- **`--output-path` must point to a directory that already exists**, or proseg panics.
- **At most one platform preset** (`--xenium`, `--cosmx`, `--cosmx-micron`, `--merscope`, `--merfish`, `--visiumhd`) can be set.

## Release process

From `debug/release-checklist.md`:
1. Update CHANGES.md
2. Bump version in `Cargo.toml` and commit
3. `git tag <version>`
4. `cargo publish`
5. Create GitHub release

## Additional context

- `debug/AGENTS.md` contains more detailed architecture docs (duplicated in `debug/CLAUDE.md`).
- `debug/SOLUTION.md` documents the parquet dictionary-encoding fix.
- `extra/` has utility scripts (cellpose integration, CosMx stitching, conversion helpers).
.PHONY: all build test bench fmt lint clean demo doc

all: build

build:
	cargo build --release

test:
	cargo test --all-targets
	cargo test --doc

lint:
	cargo fmt --all -- --check
	cargo clippy --all-targets -- -D warnings

fmt:
	cargo fmt --all

bench:
	cargo bench

doc:
	cargo doc --no-deps --open

demo: build
	./scripts/demo.sh

clean:
	cargo clean

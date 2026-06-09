# Basalt developer tasks.
#
# `make ci` runs exactly what .github/workflows/ci.yml runs, in the same order,
# so a green local run means a green pipeline.

CARGO ?= cargo
BENCH_CRATE ?= basalt-encoding

.DEFAULT_GOAL := help
.PHONY: help build test fmt fmt-check lint doc bench bench-check clean ci

help: ## Show this help
	@echo 'Basalt — make <target>'
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo 'Note: there is no optimizer or executor yet; see README.md.'

build: ## Build every crate in the workspace
	$(CARGO) build --workspace --all-targets

test: ## Run every test, including doctests
	$(CARGO) test --workspace

fmt: ## Format the workspace
	$(CARGO) fmt --all

fmt-check: ## Check formatting without changing anything
	$(CARGO) fmt --all -- --check

lint: ## Clippy over every target, warnings denied
	$(CARGO) clippy --workspace --all-targets -- -D warnings

doc: ## Build the API docs, warnings denied
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --no-deps --workspace

bench: ## Run the codec benchmarks (this takes a few minutes)
	$(CARGO) bench -p $(BENCH_CRATE)

bench-check: ## Compile the benchmarks without running them
	$(CARGO) bench --workspace --no-run

clean: ## Remove build artifacts
	$(CARGO) clean

ci: fmt-check lint test doc bench-check ## Everything CI runs, in CI's order
	@echo 'ci: all checks passed'

BIN := ./target/x86_64-unknown-linux-musl/release/pytorch-trace-tui
TAPE := demo.tape
KEYRING := /etc/apt/keyrings/charm.gpg
LIST := /etc/apt/sources.list.d/charm.list
REC_SHELL := bash

.DEFAULT_GOAL := help
.PHONY: help install-deps build tape gif clean-gif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

install-deps: ## Install vhs + ffmpeg (Debian/Ubuntu, Charm apt repo)
	sudo mkdir -p /etc/apt/keyrings
	curl -fsSL https://repo.charm.sh/apt/gpg.key | sudo gpg --dearmor -o $(KEYRING)
	echo "deb [signed-by=$(KEYRING)] https://repo.charm.sh/apt/ * *" | sudo tee $(LIST)
	sudo apt-get update
	sudo apt-get install -y vhs ffmpeg
	vhs --version && ffmpeg -version | head -1

target/x86_64-unknown-linux-musl/release/pytorch-trace-tui: ## Build the release binary
	cargo build --release

tape: target/x86_64-unknown-linux-musl/release/pytorch-trace-tui ## Record a new demo.tape interactively (Ctrl-D to finish)
	vhs record --shell $(REC_SHELL) > $(TAPE)

export VHS_NO_SANDBOX=1
gif: target/x86_64-unknown-linux-musl/release/pytorch-trace-tui ## Render demo.tape -> usage.gif
	vhs $(TAPE)

clean-gif: ## Remove the generated usage.gif
	rm -f usage.gif

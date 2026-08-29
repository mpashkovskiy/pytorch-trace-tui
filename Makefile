KEYRING := /etc/apt/keyrings/charm.gpg
.PHONY: install-deps
install-deps: ## Install vhs + ffmpeg (Debian/Ubuntu, Charm apt repo)
	sudo mkdir -p /etc/apt/keyrings
	curl -fsSL https://repo.charm.sh/apt/gpg.key | sudo gpg --dearmor -o $(KEYRING)
	echo "deb [signed-by=$(KEYRING)] https://repo.charm.sh/apt/ * *" | sudo tee /etc/apt/sources.list.d/charm.list
	sudo apt-get update
	sudo apt-get install -y vhs ffmpeg
	vhs --version && ffmpeg -version | head -1

target/x86_64-unknown-linux-musl/release/pytorch-trace-tui: ## Build the release binary
	cargo build --release

demo.tape: target/x86_64-unknown-linux-musl/release/pytorch-trace-tui ## Record a new demo.tape interactively (Ctrl-D to finish)
	vhs record --shell bash > demo.tape

LI_FPS := 12
export VHS_NO_SANDBOX=1
demo.gif: target/x86_64-unknown-linux-musl/release/pytorch-trace-tui demo.tape ## Render LinkedIn compatible gif
	vhs demo.tape --output demo-raw.gif
	ffmpeg -y -i demo-raw.gif -vf "fps=$(LI_FPS),palettegen=max_colors=256" /tmp/li-pal.png
	ffmpeg -y -i demo-raw.gif -i /tmp/li-pal.png -lavfi "fps=$(LI_FPS) [x]; [x][1:v] paletteuse" demo.gif
	@rm -f /tmp/li-pal.png demo-raw.gif

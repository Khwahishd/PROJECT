.PHONY: all build test race chaos bench cover lint fmt vet clean demo docker

GO      ?= go
BIN     := bin
PKGS    := ./...
LDFLAGS := -s -w

all: build

build:
	@mkdir -p $(BIN)
	$(GO) build -ldflags "$(LDFLAGS)" -o $(BIN)/raftkvd  ./cmd/raftkvd
	$(GO) build -ldflags "$(LDFLAGS)" -o $(BIN)/raftkvctl ./cmd/raftkvctl
	@echo "built $(BIN)/raftkvd and $(BIN)/raftkvctl"

test:
	$(GO) test $(PKGS) -timeout 10m

short:
	$(GO) test $(PKGS) -short -timeout 3m

race:
	$(GO) test -race $(PKGS) -timeout 20m

# The full randomized sweep. Each seed is an independent, reproducible failure
# history; a failure prints the seed and the event trace that produced it.
chaos:
	CHAOS_SEEDS=1500 $(GO) test ./internal/raftsim/ -run TestChaos -timeout 40m -v 2>&1 | tail -20

bench:
	$(GO) test ./internal/raft/ -run '^$$' -bench . -benchmem

cover:
	$(GO) test $(PKGS) -coverpkg=./internal/... -coverprofile=coverage.out -timeout 10m
	$(GO) tool cover -func=coverage.out | tail -1
	@echo "html report: go tool cover -html=coverage.out"

vet:
	$(GO) vet $(PKGS)

fmt:
	gofmt -l -w .

lint: vet
	@test -z "$$(gofmt -l .)" || (echo "gofmt needed:"; gofmt -l .; exit 1)

demo: build
	./scripts/demo.sh

docker:
	docker build -t raftkv:latest .

clean:
	rm -rf $(BIN) data coverage.out

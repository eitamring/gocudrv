.PHONY: all test vet race lint nocgo help

all: vet test lint

test:
	go test -v -race -count=1 -cover ./...

vet:
	go vet ./...

race:
	go test -v -race -count=1 ./...

lint:
	if command -v golangci-lint >/dev/null; then golangci-lint run; else echo "golangci-lint not found"; fi

nocgo:
	bash scripts/check-no-cgo.sh

help:
	@echo "targets:"
	@echo "  test   - run tests with race and coverage"
	@echo "  vet    - run go vet"
	@echo "  race   - run tests with race detector"
	@echo "  lint   - run golangci-lint"
	@echo "  nocgo  - build and test with CGO_ENABLED=0"
	@echo "  all    - vet, test, lint"

FROM golang:1.24-alpine AS build
WORKDIR /src
# Copy the module files first so dependency resolution is cached independently
# of source changes. (This project has no dependencies, but the layering keeps
# rebuilds fast if that ever changes.)
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags "-s -w" -o /out/raftkvd  ./cmd/raftkvd && \
    CGO_ENABLED=0 go build -ldflags "-s -w" -o /out/raftkvctl ./cmd/raftkvctl

FROM alpine:3.20
RUN adduser -D -u 10001 raftkv
COPY --from=build /out/raftkvd  /usr/local/bin/raftkvd
COPY --from=build /out/raftkvctl /usr/local/bin/raftkvctl
USER raftkv
VOLUME ["/data"]
EXPOSE 9001
ENTRYPOINT ["raftkvd"]

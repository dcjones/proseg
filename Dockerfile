# syntax=docker/dockerfile:1
FROM rust:1.87 AS builder
WORKDIR /app
COPY . .
RUN cargo install --path .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/cargo/bin/proseg /usr/local/bin/proseg
CMD ["proseg"]

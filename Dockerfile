# syntax=docker/dockerfile:1
FROM rust:1.88 AS builder
WORKDIR /app
COPY . .
RUN cargo install --path . \
  && mkdir /proseg_bins \
  && cp /usr/local/cargo/bin/proseg* /proseg_bins/

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates procps && rm -rf /var/lib/apt/lists/*
COPY --from=builder /proseg_bins/ /usr/local/bin/
CMD ["proseg"]

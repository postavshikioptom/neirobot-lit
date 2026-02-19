# Multi-stage build for Neirobot LiT
# Stage 1: Builder
FROM rust:1.75-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    musl-dev \
    openssl-dev \
    openssl-libs-static \
    pkgconfig

WORKDIR /build

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY benches ./benches

# Build with musl target for static linking
RUN cargo build --release --target x86_64-unknown-linux-musl

# Stage 2: Runtime
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    libgcc

# Create non-root user
RUN addgroup -g 1000 neirobot && \
    adduser -D -u 1000 -G neirobot neirobot

# Create necessary directories
RUN mkdir -p /app/bots /app/logs /app/models && \
    chown -R neirobot:neirobot /app

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/neirobot-lit /usr/local/bin/

# Copy configuration templates
COPY global.toml exchange.toml ./

USER neirobot

ENTRYPOINT ["neirobot-lit"]
CMD ["--help"]

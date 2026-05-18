# Dockerfile for the SQLite MCP server.
# Node Express on port 3002. Read-only access to /app/data/loadshare.db.

FROM node:20-slim

WORKDIR /app

# Compile tools for better-sqlite3
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 make g++ wget \
    && rm -rf /var/lib/apt/lists/*

COPY package.json ./
RUN npm install --omit=dev

COPY sqlite-server.js ./

EXPOSE 3002

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD wget -qO- http://localhost:3002/health || exit 1

CMD ["node", "sqlite-server.js"]

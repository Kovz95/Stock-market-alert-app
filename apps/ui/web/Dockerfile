# Build stage: install deps and build Next.js app (monorepo root context)
FROM node:22-alpine AS base

RUN corepack enable && corepack prepare pnpm@latest --activate
RUN apk add --no-cache libc6-compat

WORKDIR /app

# Copy workspace and lockfile
COPY pnpm-workspace.yaml package.json pnpm-lock.yaml* ./

# Copy gen/ts (needed by web app)
COPY gen/ts ./gen/ts

# Copy apps/ui (web and any sibling packages)
COPY apps/ui ./apps/ui

# Install all dependencies
RUN pnpm install

# Build the web app (from repo root so workspace deps resolve)
WORKDIR /app/apps/ui/web
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

# Production stage: run Next.js standalone server
FROM node:22-alpine AS runner

RUN apk add --no-cache libc6-compat
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy standalone output (includes minimal node_modules and server)
COPY --from=base --chown=nextjs:nodejs /app/apps/ui/web/.next/standalone ./
COPY --from=base --chown=nextjs:nodejs /app/apps/ui/web/.next/static ./apps/ui/web/.next/static
COPY --from=base --chown=nextjs:nodejs /app/apps/ui/web/public ./apps/ui/web/public

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "apps/ui/web/server.js"]

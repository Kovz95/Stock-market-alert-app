
# DCW - The Crawl TTRPG Tools

A full-stack monorepo for managing TTRPG game mechanics including effects, skills, and tags. Built with Next.js 16, Go microservices, gRPC, and PostgreSQL.

## Quick Reference

```bash
# Start all services
docker-compose up -d

# Start frontend
pnpm install && pnpm dev

# Generate protobuf code
buf generate

# Generate SQL code
sqlc generate -f database/sqlc.yaml

# View traces
open http://localhost:16686
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js 16)                     │
│                         http://localhost:3000                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │ gRPC-Web (nice-grpc)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Envoy Proxy (Port 80)                       │
│                    Routes by service prefix                      │
└───┬───────────────────────┬───────────────────────┬─────────────┘
    │                       │                       │
    ▼                       ▼                       ▼
┌─────────┐           ┌─────────┐           ┌─────────┐
│ Effects │           │ Skills  │           │  Tags   │
│ Service │           │ Service │           │ Service │
│ :50051  │           │ :50051  │           │ :50051  │
└────┬────┘           └────┬────┘           └────┬────┘
     │                     │                     │
     └──────────────────┬──┴─────────────────────┘
                        ▼
              ┌─────────────────┐
              │   PostgreSQL    │
              │    Port 5432    │
              └─────────────────┘
```

## Project Structure

```
dcw/
├── apps/
│   ├── grpc/                    # Go microservices
│   │   ├── effects_service/     # Effects CRUD service
│   │   ├── skills_service/      # Skills, SkillTypes, BasicSkills
│   │   └── tags_service/        # Tags CRUD service
│   └── ui/
│       └── web/                 # Next.js 16 frontend
├── core/                        # Shared Go packages
│   ├── effects/                 # Effect domain logic & mappers
│   ├── shared/                  # Enum mappers (proto ↔ db)
│   ├── skills/                  # Skills domain logic
│   └── tracing/                 # OpenTelemetry setup
├── database/
│   ├── sql/
│   │   ├── schema.sql           # PostgreSQL types & tables
│   │   └── queries/             # SQL queries for sqlc
│   └── generated/               # Generated Go code from sqlc
├── gen/
│   ├── go/                      # Generated protobuf Go code
│   └── ts/                      # Generated protobuf TypeScript
├── proto/                       # Protocol buffer definitions
│   ├── effects/v1/
│   ├── skills/v1/
│   ├── tags/v1/
│   └── shared/v1/
├── docker-compose.yml           # Local development stack
├── envoy.yaml                   # API Gateway configuration
├── go.work                      # Go workspace
└── pnpm-workspace.yaml          # pnpm monorepo config
```

## Technology Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 16.x | React framework with App Router |
| React | 19.x | UI library |
| TypeScript | 5.x | Type safety |
| Tailwind CSS | 4.x | Styling |
| Jotai | 2.x | Atomic state management |
| React Query | 5.x | Server state & caching |
| nice-grpc | 2.x | gRPC-Web client |
| Clerk | 6.x | Authentication |
| shadcn/ui | - | Component library (Radix-based) |

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Go | 1.25.x | Service implementation |
| gRPC | - | Service communication |
| pgx | v5 | PostgreSQL driver |
| sqlc | - | Type-safe SQL code generation |
| OpenTelemetry | - | Distributed tracing |

### Infrastructure
| Service | Port | Purpose |
|---------|------|---------|
| Next.js | 3000 | Frontend dev server |
| Envoy | 80 | API Gateway |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Caching (future) |
| Jaeger | 16686 | Trace visualization |

## Development Guidelines

### Code Style

#### Go Services
- **Package names**: lowercase, single word (`shared`, `effects`, `tracing`)
- **Types**: PascalCase (`CreateEffectRequest`, `Server`)
- **Methods**: PascalCase (`CreateEffect`, `GetEffect`)
- **Variables**: camelCase (`req`, `pool`, `span`)
- **Error handling**: Use gRPC status codes

```go
// Good: Use gRPC status codes
if req.Name == "" {
    return nil, status.Error(codes.InvalidArgument, "effect name is required")
}

// Good: Record errors in spans for tracing
span := trace.SpanFromContext(ctx)
if err != nil {
    span.RecordError(err)
    return nil, status.Errorf(codes.Internal, "failed to create: %v", err)
}
```

#### TypeScript/React
- **Components**: PascalCase files and exports (`EffectCard.tsx`)
- **Hooks**: `use` prefix, camelCase (`useEffects`, `useCreateEffect`)
- **Atoms**: camelCase with `Atom` suffix (`effectsAtom`, `modalOpenAtom`)
- **Server Actions**: `"use server"` directive at top of file
- **Client Components**: `"use client"` directive when using hooks

```typescript
// Good: Separate concerns - atoms for state, hooks for server communication
// atoms/effect-atoms.ts
export const effectsAtom = atom<Effect[]>([]);
export const modalOpenAtom = atom<boolean>(false);

// hooks/useEffects.ts
export function useEffects() {
    return useQuery({ queryKey: ["effects"], queryFn: getEffects });
}
```

### File Organization

#### Adding a New Entity/Domain

1. **Proto definition** (`proto/<domain>/v1/<domain>.proto`):
```protobuf
syntax = "proto3";
package domain.v1;

service DomainService {
    rpc Create(CreateRequest) returns (Response);
    rpc Get(GetRequest) returns (Response);
    rpc List(ListRequest) returns (ListResponse);
    rpc Update(UpdateRequest) returns (Response);
    rpc Delete(DeleteRequest) returns (google.protobuf.Empty);
}
```

2. **Database schema** (`database/sql/schema.sql`):
```sql
CREATE TYPE domain_enum AS ENUM ('value1', 'value2');

CREATE TABLE domains (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

3. **SQL queries** (`database/sql/queries/domain_queries.sql`):
```sql
-- name: GetDomains :many
SELECT * FROM domains ORDER BY name;

-- name: CreateDomain :one
INSERT INTO domains (name) VALUES ($1) RETURNING *;
```

4. **Go service** (`apps/grpc/<domain>_service/`):
   - `main.go` - Server setup with tracing
   - `server.go` - Server struct definition
   - `handler.go` - RPC implementations

5. **Frontend components** (`apps/ui/web/components/`):
   - `containers/DomainsContainer.tsx` - Data fetching container
   - `cards/DomainCard.tsx` - Display component
   - `modals/DomainFormModal.tsx` - Create/Edit modal

6. **Frontend state** (`apps/ui/web/lib/`):
   - `atoms/domain-atoms.ts` - Jotai atoms
   - `hooks/useDomains.ts` - React Query hooks
   - `actions/domain-actions.ts` - Server actions

### State Management Patterns

#### Jotai Atoms Pattern
```typescript
// lib/atoms/effect-atoms.ts

// 1. Core data atoms
export const effectsAtom = atom<Effect[]>([]);
export const effectAtomsAtom = splitAtom(effectsAtom); // Efficient list rendering

// 2. UI state atoms
export const modalOpenAtom = atom<boolean>(false);
export const loadingAtom = atom<boolean>(true);
export const editingEffectAtom = atom<Effect | null>(null);

// 3. Form field atoms (using focusAtom for nested updates)
export const formDataBaseAtom = atom<EffectFormData>({ name: "", ... });
export const formNameAtom = focusAtom(formDataBaseAtom, (o) => o.prop("name"));

// 4. Action atoms (write-only)
export const openCreateFormAtom = atom(null, (get, set) => {
    set(editingEffectAtom, null);
    set(formDataBaseAtom, defaultFormData);
    set(modalOpenAtom, true);
});
```

#### React Query + Server Actions Pattern
```typescript
// actions/effect-actions.ts
"use server"

export async function getEffects(): Promise<Effect[]> {
    const channel = createChannel(GRPC_ENDPOINT);
    const client = createClient(EffectServiceDefinition, channel);
    const response = await client.listEffects({});
    return response.effects;
}

// hooks/useEffects.ts
"use client"

export function useEffects() {
    return useQuery({
        queryKey: ["effects"],
        queryFn: getEffects,
    });
}

export function useCreateEffect() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: createEffect,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["effects"] });
        },
    });
}
```

### gRPC Service Implementation

#### Handler Pattern with Tracing
```go
func (s *Server) CreateEffect(ctx context.Context, req *effectsv1.CreateEffectRequest) (*effectsv1.Effect, error) {
    // 1. Get span from context (auto-created by gRPC interceptor)
    span := trace.SpanFromContext(ctx)
    span.SetAttributes(attribute.String("rpc.method", "CreateEffect"))

    // 2. Validate request
    if req.Name == "" {
        span.RecordError(fmt.Errorf("name not provided"))
        return nil, status.Error(codes.InvalidArgument, "effect name is required")
    }

    // 3. Acquire database connection
    conn, err := s.pool.Acquire(ctx)
    if err != nil {
        return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
    }
    defer conn.Release()

    // 4. Create queries instance
    queries := db.New(conn)

    // 5. Convert proto types to db types
    params := db.CreateEffectParams{
        Name:    req.Name,
        Quality: shared.QualityFromProto(&req.Quality),
        // ... more conversions
    }

    // 6. Execute with nested span
    _, qrySpan := s.tracer.Start(ctx, "query_create_effect")
    effect, err := queries.CreateEffect(ctx, params)
    qrySpan.End()

    if err != nil {
        span.RecordError(err)
        return nil, status.Errorf(codes.Internal, "failed to create: %v", err)
    }

    // 7. Convert db types back to proto
    return dbEffectToProto(effect), nil
}
```

### Type Mapping

#### Proto ↔ Database Enum Conversion
Located in `core/shared/mappers.go`:

```go
func QualityFromProto(q *pb.QualityType) db.QualityType {
    switch *q {
    case pb.QualityType_QUALITY_TYPE_COMMON:
        return db.SkillQualityCommon
    case pb.QualityType_QUALITY_TYPE_RARE:
        return db.SkillQualityRare
    // ... etc
    }
    return db.SkillQualityCommon
}

func QualityToProto(q db.QualityType) pb.QualityType {
    switch q {
    case db.SkillQualityCommon:
        return pb.QualityType_QUALITY_TYPE_COMMON
    // ... etc
    }
    return pb.QualityType_QUALITY_TYPE_UNSPECIFIED
}
```

### Database Patterns

#### sqlc Query Conventions
```sql
-- Simple queries
-- name: GetEffects :many
SELECT * FROM effects ORDER BY name;

-- name: GetEffect :one
SELECT * FROM effects WHERE id = $1;

-- name: CreateEffect :one
INSERT INTO effects (name, quality, target)
VALUES ($1, $2, $3) RETURNING *;

-- Dynamic filtering (use cardinality check for optional arrays)
-- name: FilterEffects :many
SELECT * FROM effects
WHERE
    (operation::text = ANY($1::text[]) OR cardinality($1) IS NULL)
    AND (quality::text = ANY($2::text[]) OR cardinality($2) IS NULL)
    AND (name ILIKE '%' || $3 || '%' OR $3 IS NULL)
ORDER BY name;
```

### Envoy Routing

Routes are defined by gRPC service prefix in `envoy.yaml`:

```yaml
routes:
  - match: { prefix: "/effects.v1.EffectService/" }
    route: { cluster: effects_service }
  - match: { prefix: "/skills.v1.SkillService/" }
    route: { cluster: skills_service }
  - match: { prefix: "/tags.v1.TagsService/" }
    route: { cluster: tags_service }
```

When adding a new service, add a new route and cluster.

## Common Tasks

### Adding a New gRPC Method

1. Define in proto file
2. Run `buf generate`
3. Implement handler in Go service
4. Add server action in frontend
5. Add React Query hook
6. Update UI components

### Adding a New Database Table

1. Add to `database/sql/schema.sql`
2. Add queries to `database/sql/queries/`
3. Run `sqlc generate -f database/sqlc.yaml`
4. Add type mappers in `core/shared/mappers.go`
5. Update service handlers

### Adding a New Shared Enum

1. Define in `proto/shared/v1/enums.proto`
2. Run `buf generate`
3. Add to `database/sql/schema.sql` as PostgreSQL enum
4. Run `sqlc generate`
5. Add mappers in `core/shared/mappers.go`

### Debugging gRPC Services

```bash
# List available services (requires reflection enabled)
grpcurl -plaintext localhost:50051 list

# List methods on a service
grpcurl -plaintext localhost:50051 list effects.v1.EffectService

# Call a method
grpcurl -plaintext -d '{}' localhost:50051 effects.v1.EffectService/ListEffects
```

## Testing

### Frontend Testing
```bash
pnpm --filter=web test        # Run tests
pnpm --filter=web test:watch  # Watch mode
```

### Backend Testing
```bash
cd apps/grpc/effects_service
go test ./...                 # Run all tests
go test -v ./...              # Verbose output
go test -cover ./...          # With coverage
```

### Integration Testing
```bash
# Start services
docker-compose up -d

# Run integration tests
go test -tags=integration ./...
```

## Troubleshooting

### Common Issues

**gRPC connection refused**
- Ensure Docker Compose is running: `docker-compose ps`
- Check Envoy is healthy: `curl http://localhost:80/health`
- Verify service is registered: `docker-compose logs envoy`

**Database connection issues**
- Check PostgreSQL is running: `docker-compose logs postgres`
- Verify connection string in service logs
- Test connection: `psql -h localhost -U postgres -d the_crawl_ttrpg`

**Proto generation failures**
- Ensure buf is installed: `buf --version`
- Check buf.yaml and buf.gen.yaml syntax
- Validate protos: `buf lint`

**Traces not appearing in Jaeger**
- Verify Jaeger is running: `docker-compose logs jaeger`
- Check OTLP endpoint in service config
- Ensure service is using tracing interceptor

## Environment Variables

### Frontend (`apps/ui/web/.env`)
```env
GRPC_ENDPOINT=localhost:80
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...
```

### Backend (via Docker Compose)
```env
DATABASE_URL=postgres://postgres:postgres@postgres:5432/the_crawl_ttrpg
OTEL_EXPORTER_OTLP_ENDPOINT=jaeger:4317
```

## CI/CD Considerations

### Build Order
1. Generate code: `buf generate && sqlc generate`
2. Build Go services: `go build ./...`
3. Build frontend: `pnpm build`
4. Build Docker images

### Docker Images
- Each service has its own Dockerfile
- Multi-stage builds minimize image size
- Services depend on generated code from `gen/` and `database/generated/`

## Best Practices Checklist

### When Adding Features
- [ ] Define proto contract first
- [ ] Generate code before implementing
- [ ] Add proper error handling with gRPC status codes
- [ ] Include tracing spans for database operations
- [ ] Validate input at service boundary
- [ ] Update Envoy routing if new service
- [ ] Add type mappers for new enums
- [ ] Follow existing component patterns in frontend
- [ ] Invalidate React Query cache on mutations

### When Modifying Existing Code
- [ ] Read existing implementation first
- [ ] Maintain backward compatibility in protos (additive changes)
- [ ] Update all type mappers if changing enums
- [ ] Test with grpcurl before frontend integration
- [ ] Check Jaeger for trace propagation

### Security
- [ ] Never commit `.env` files
- [ ] Validate all user input at service boundary
- [ ] Use parameterized queries (sqlc handles this)
- [ ] Keep Clerk keys secure

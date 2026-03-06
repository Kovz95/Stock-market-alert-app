
# Web App – Next.js, React & UI Guide

Frontend for Stockalert: Next.js 16 (App Router), React 19, Jotai, React Query, and **shadcn UI** for all UI primitives. This guide covers architecture, server actions, state, component design, and when to split components.

---

## Table of Contents

1. [Next.js (App Router)](#nextjs-app-router)
2. [Server Actions](#server-actions)
3. [Jotai](#jotai)
4. [React Query](#react-query)
5. [React Design Patterns](#react-design-patterns)
6. [When to Separate into Components](#when-to-separate-into-components)
7. [shadcn UI](#shadcn-ui)
8. [File & Folder Conventions](#file--folder-conventions)

---

## Next.js (App Router)

### App directory structure

- **`app/`** – Routes and layouts. Prefer **server components** by default; add `"use client"` only where needed (hooks, Jotai, event handlers, browser APIs).
- **Route segments** – Folders define segments; `page.tsx` is the UI, `layout.tsx` wraps children, `loading.tsx` / `error.tsx` for boundaries.
- **Route groups** – Use `(groupName)` (e.g. `(reference)`) to group routes without changing the URL.

### Server vs client components

| Use **server component** (no directive) when: |
|-----------------------------------------------|
| Page only needs data from server (e.g. `getContentBySlug`, `generateMetadata`). |
| No `useState`, `useEffect`, Jotai, React Query, or event handlers. |
| You want to keep the component and its data-fetching on the server. |

| Use **client component** (`"use client"` at top) when: |
|------------------------------------------------------|
| You use hooks: `useState`, `useEffect`, `useAtom`, `useQuery`, etc. |
| You need event handlers: `onClick`, `onSubmit`, `onChange`. |
| You use Jotai atoms or React Query. |
| You use browser APIs or components that depend on them (e.g. modals, sidebars). |

- **Rule of thumb:** Start as server component; add `"use client"` only in the leaf file that needs client behavior. Keep server components as the default for pages that only compose client containers.

### Pages and layouts

- **Root layout** (`app/layout.tsx`) – Wraps the app with `QueryClientAtomProvider` (Jotai + TanStack Query), `ThemeProvider`, `TooltipProvider`, and layout components. Stays a server component; providers are client components.
- **Page components** – Prefer thin server pages that render a single client container when the route is interactive (e.g. `app/(reference)/modifications/page.tsx` just returns `<ModificationsContainer />`). Use async server pages for content routes (e.g. `app/(reference)/rules/[[...slug]]/page.tsx`) with `generateStaticParams` and `generateMetadata` where applicable.
- **Params and searchParams** – In App Router they are **Promises**. Always `await` them in server components: `const { slug } = await params;`.

### Metadata and static generation

- Export `metadata` and/or `viewport` from `layout.tsx` or `page.tsx` for SEO and viewport.
- For dynamic metadata, use `generateMetadata({ params })` and `await params` inside it.
- Use `generateStaticParams()` for static routes (e.g. content slugs); return an array of `{ slug }` (or equivalent) objects.

### Data fetching in Next.js

- **Server components** – `async` and `await` data directly (e.g. `getContentBySlug`, `getContentNavigation`). No `useEffect` for initial load.
- **Client components** – Do **not** fetch in the module body. Use server actions from client (called in `useEffect`, event handlers, or React Query) or use React Query with server actions as `queryFn` / `mutationFn`.

---

## Server Actions

### Where and how

- **Location:** `actions/` (e.g. `effect-actions.ts`, `modification-actions.ts`, `tag-actions.ts`).
- **Directive:** Every action file must start with `"use server"`.
- **Usage:** Call these async functions from client components (or from other server actions). They run on the server and can use secrets, DB, and gRPC.

### Conventions in this project

2. **gRPC** – Create channel and client per call (or reuse as per existing pattern), pass `metadata` from `getGrpcMetadata`. Do not use gRPC or `getGrpcMetadata` from client-only code.
3. **Errors** – Throw descriptive errors; the client will catch and show them (e.g. in modals or toasts). For “Unauthorized”, throw after checking `isAuthenticated`.
4. **Return values** – Return serializable data (plain objects, arrays). Server actions cannot return functions or non-serializable values.

### Example pattern

```typescript
// actions/effect-actions.ts
"use server";

import { getGrpcMetadata } from "@/lib/grpc-metadata";
import { createChannel, createClient } from "nice-grpc";
import * as effects from "@dcw/generated/effects/v1/effects";

const GRPC_ENDPOINT = process.env.GRPC_ENDPOINT || "localhost:80";

export async function getEffects() {
  const { metadata } = await getGrpcMetadata("ListEffects");
  const channel = createChannel(GRPC_ENDPOINT);
  const client = createClient(effects.EffectServiceDefinition, channel);
  const response = await client.listEffects({}, { metadata });
  return response.effects;
}

export async function createEffect(effect: effects.CreateEffectRequest) {
  const { isAuthenticated } = await auth();
  if (!isAuthenticated) throw new Error("Unauthorized");
  const { metadata } = await getGrpcMetadata("CreateEffect");
  const channel = createChannel(GRPC_ENDPOINT);
  const client = createClient(effects.EffectServiceDefinition, channel);
  const response = await client.createEffect(effect, { metadata });
  return response;
}
```

### When to use server actions

- Any server-only operation: gRPC calls, auth, env-only config.
- All mutations and reads that go through gRPC in this app are implemented as server actions and then used from client (directly or via React Query).

---

## Jotai

### Role in the app

- **UI and form state:** Modals open/closed, which item is being edited, form field values, wizard step, loading/error for a given view.
- **Client-side list state:** When the container fetches via server actions and stores results in atoms (e.g. `effectsAtom`, `modificationsAtom`), and list items receive an **atom reference** (e.g. `effectAtom`) so only the relevant item re-renders on update.

### Atom layout (per domain)

Mirror the pattern in `lib/atoms/effect-atoms.ts` and `lib/atoms/modification-atoms.ts`:

1. **Core data** – `itemsAtom = atom<Item[]>([])`, plus `itemAtomsAtom = splitAtom(itemsAtom)` for list rendering.
2. **UI state** – `loadingAtom`, `errorAtom`, `modalOpenAtom`, `editingItemAtom`, `formSubmittingAtom`, and for wizards `wizardStepAtom`, etc.
3. **Form data** – One `formDataBaseAtom` (single object), then **focused atoms** per field with `focusAtom(formDataBaseAtom, (optic) => optic.prop("fieldName"))`. Use `splitAtom` for array fields (e.g. tags) when each item is a separate atom.
4. **Derived** – Read-only atoms: `atom((get) => get(editingItemAtom) !== null)`, `modalTitleAtom`, etc.
5. **Actions** – Write-only or read-write atoms that coordinate multiple updates: `openCreateFormAtom`, `openEditFormAtom`, `closeModalAtom`, `addItemAtom`, `removeItemAtom`, etc.

### focusAtom (jotai-optics)

- Use for **nested form state**: one base atom holding an object, and many atoms that “focus” on a single key. Updating a focused atom updates only that key in the base atom.
- Keeps form state in one place and avoids prop drilling; field components use `useAtom(formNameAtom)` etc.

### splitAtom

- Use for **lists** (e.g. `effectAtomsAtom = splitAtom(effectsAtom)`). Map over the array of atoms and pass each atom to a card/item component. When one item updates, only that item’s atom changes, so only that card re-renders.

### Passing atoms as props

- For list items, pass the **atom** (e.g. `effectAtom`), not the value: `<EffectCard effectAtom={effectAtom} />`. Inside the card, `useAtomValue(effectAtom)` (or `useAtom`). This preserves fine-grained updates.

### When to use Jotai vs React Query

- **Jotai:** UI state, form state, wizard state, and (in current patterns) client-held list state that is fetched via server actions and stored in atoms, with containers managing loading/error and refetch (e.g. debounced filter).
- **React Query:** Use when you want cache-by-key, background refetch, and request deduplication (e.g. `useEffects()`, `useEffect(id)`). Hooks in `hooks/` call server actions as `queryFn`/`mutationFn` and invalidate queries on success. Both patterns coexist: some features use only Jotai + server actions, others use React Query + server actions; follow the existing feature’s pattern.

---

## React Query

- **Provider:** Root layout uses `QueryClientAtomProvider` from `jotai-tanstack-query/react`, which provides both the TanStack Query client and Jotai so atoms and hooks share the same cache.
- **Usage:** Hooks in `hooks/` (e.g. `useEffects`, `useCreateEffect`, `useUpdateEffect`, `useDeleteEffect`) use `useQuery` / `useMutation` with server actions. Mutations call `queryClient.invalidateQueries({ queryKey: ["effects"] })` (or the relevant key) on success.
- **Server actions:** All actual I/O lives in server actions; React Query only calls them and manages cache and loading/error state.

### jotai-tanstack-query

- **Purpose:** Exposes TanStack Query as Jotai atoms so query keys and refetch can depend on Jotai state (e.g. pagination, filters) and any component can read/trigger the same query via `useAtom` without prop drilling.
- **Same cache:** Atoms from `atomWithQuery` / `atomWithInfiniteQuery` / `atomWithMutation` use the same `QueryClient` as `useQuery`/`useMutation`; invalidating a query key updates both hook and atom consumers.
- **When to use atoms vs hooks:** Use **hooks** when you only need loading/error/data in one place and no composition with Jotai. Use **jotai-tanstack-query** when the query key or refetch should depend on Jotai state (e.g. `alertsPaginatedQueryAtom` reading `alertsPageAtom` / `alertsPageSizeAtom`) or you want multiple components to subscribe via atoms. You can migrate one feature at a time; the rest can stay on hooks.
- **Example:** Paginated alerts use `alertsPaginatedQueryAtom` in `lib/store/alerts.ts` (atomWithQuery that reads page/pageSize atoms); the alerts page uses it via `useAlertsPaginated()` which is a thin wrapper around `useAtom(alertsPaginatedQueryAtom)`.

---

## React Design Patterns

### Composition

- **Containers** – Own data flow: fetch (server action in `useEffect` or React Query), Jotai state, and pass data or atoms to presentational children. Example: `EffectsContainer`, `ModificationsContainer`.
- **Modals** – Own dialog open state via Jotai (`modalOpenAtom`, `editingItemAtom`), read form atoms, call server actions on submit, then update list atoms or invalidate queries and close modal.
- **Forms** – Form state lives in Jotai (base + focus atoms). Field components are presentational and bound to atoms; they don’t receive `value`/`onChange` from parent, they `useAtom(formXAtom)`.

### Controlled vs uncontrolled

- **Controlled:** All form inputs in this app are controlled via Jotai. Value comes from `useAtomValue(formXAtom)`, updates via `set` from `useAtom(formXAtom)` or `useSetAtom(formXAtom)`.
- **Uncontrolled:** Use only when you intentionally avoid state (e.g. file input or a one-off ref). Prefer controlled for anything that’s submitted or validated.

### Lists and keys

- When mapping over **atoms** from `splitAtom`, you get an array of atom references. Use a **stable key** if available (e.g. `item.id` from the atom value). If the list order is stable and items are not reordered, index can be used as in existing code, but prefer ID when you have it to avoid subtle bugs with reordering.

### Loading and error

- Handle in the container: set `loadingAtom` / `errorAtom` during fetch, and in the UI branch on loading/error/empty/success. Use a single `renderContent()` or similar to keep the page structure clear.

### Accessibility

- Use semantic HTML and shadcn components (they use Radix). Buttons that only show an icon need `aria-label`. Use `DialogDescription` and `DialogTitle` for modals.

---

## When to Separate into Components

### Extract a new component when:

1. **Reuse** – Same UI or logic used in more than one place (e.g. `EffectCard`, `ModificationCard`, shared form fields).
2. **List items** – Each item is a natural unit (e.g. card per effect). Pass the atom for the item so only that item re-renders.
3. **Form sections** – A logical group of fields (e.g. `EffectBasicFields`, `EffectEnumFields`, `EffectTagsField`) that can be tested and read in isolation.
4. **Modals / dialogs** – Each form modal is its own component; it reads many atoms and dispatches actions.
5. **Wizard steps** – Each step is a component; the wizard container owns step index and submits at the end.
6. **Filters** – Filter bar is its own component, bound to filter atoms.
7. **Readability** – A single component would be long or nested enough to be hard to follow; splitting by responsibility (e.g. “header”, “body”, “footer”) or by domain (e.g. “basic” vs “details”) improves clarity.

### Keep inline when:

1. **Used once** – No reuse and no readability win.
2. **Trivial** – A few lines of JSX with no logic.
3. **Tightly coupled** – The block only makes sense in this one parent and would require many props or atoms to extract.

### Naming and placement

- **Containers:** `components/containers/<Domain>Container.tsx` – data and UI state, composes cards + filter + modal.
- **Cards:** `components/cards/<Entity>Card.tsx` – one list item; receives `entityAtom`.
- **Modals:** `components/modals/<Entity>FormModal.tsx` or wizard in `components/modals/<entity>-wizard/`.
- **Form field groups:** `components/<domain>/<Entity>BasicFields.tsx`, `...EnumFields.tsx`, etc., or under `components/forms/` when shared across domains.
- **Filters:** `components/filters/<Domain>FilterBar.tsx`.

---

## shadcn UI

### Rule: Use shadcn for all UI

- **Do not** introduce other UI libraries for primitives (buttons, inputs, selects, modals, etc.). Use **only** components from `components/ui/` (shadcn) and build on top of them.
- **Adding components:** Use the shadcn CLI (`npx shadcn@latest add <component>`) so styles and behavior stay consistent (Radix-based, Tailwind, CVA).

### Where they live

- `components/ui/` – Button, Input, Label, Textarea, Select, Dialog, Card, Badge, Switch, Sheet, Sidebar, Tooltip, Separator, Skeleton, Combobox, Field, Alert Dialog, Drawer, Sonner (toast), etc.
- Use the existing set; add new primitives via shadcn when needed.

### Styling

- **`cn()`** – From `@/lib/utils`. Use for conditional or merged class names: `cn(buttonVariants({ variant, size }), className)`.
- **Tailwind** – All styling is Tailwind. Prefer design tokens (e.g. `text-muted-foreground`, `bg-primary`, `border-border`) over raw colors.
- **Variants** – Many shadcn components use `class-variance-authority` (e.g. `buttonVariants`) and accept `variant` and `size`. Use these instead of duplicating classes.

### Forms and fields

- **Label + Input / Select / Textarea** – Use `Label` with `htmlFor` and the corresponding shadcn control. For structured forms, you can use the `Field` family (`Field`, `FieldGroup`, `FieldLabel`, etc.) from `@/components/ui/field` for consistency and error display.
- **Validation / errors** – Prefer Field’s error state and/or inline error message components; avoid one-off divs that don’t match the design system.

### Modals and dialogs

- Use `Dialog`, `DialogContent`, `DialogHeader`, `DialogTitle`, `DialogDescription`, `DialogFooter`, `DialogClose` from `@/components/ui/dialog`. Control open state with Jotai (`modalOpenAtom`, `onOpenChange`).

### Icons

- Use `@phosphor-icons/react` for icons (e.g. `Plus`, `PencilSimple`, `Trash`). Keep size and alignment consistent (e.g. `data-icon="inline-start"` where supported, or consistent `className` for icon sizing).

---

## File & Folder Conventions

| Path | Purpose |
|------|--------|
| `app/` | Routes, layouts, server pages. Thin pages that delegate to client containers. |
| `actions/` | Server actions only. One file per domain or resource (e.g. `effect-actions.ts`). |
| `components/containers/` | Top-level client components that own data and state for a page/section. |
| `components/cards/` | List item components; receive atom (e.g. `effectAtom`). |
| `components/modals/` | Form modals and wizards. |
| `components/filters/` | Filter bars bound to filter atoms. |
| `components/effects/`, `components/skills/`, etc. | Domain-specific presentational pieces (e.g. effect form sections). |
| `components/forms/` | Shared form primitives (e.g. `TextInput`, `EnumSelect`) and shared field components. |
| `components/ui/` | shadcn primitives only; do not add app logic here. |
| `lib/atoms/` | Jotai atoms per domain (e.g. `effect-atoms.ts`, `modification-atoms.ts`). |
| `lib/utils.ts` | `cn()` and small helpers. |
| `lib/grpc-metadata.ts` | Server-only; build gRPC metadata with Clerk auth. |
| `hooks/` | React Query (and other) hooks that call server actions. |

### Naming

- **Components:** PascalCase (`EffectCard`, `ModificationsContainer`).
- **Files:** PascalCase for components (`EffectCard.tsx`), kebab-case for utilities or config when the project uses it.
- **Atoms:** camelCase with a descriptive suffix (`effectsAtom`, `modalOpenAtom`, `formNameAtom`).
- **Server actions:** camelCase (`getEffects`, `createEffect`, `searchModifications`).

### Imports

- Prefer `@/` for app imports (`@/components/ui/button`, `@/lib/atoms/effect-atoms`, `@/actions/effect-actions`).
- Use generated types from the workspace package (e.g. `@dcw/generated/...`) for API and proto types.

---

## Quick reference

- **Page is only layout + one client area** → Server page that renders one container.
- **Need hooks or events** → `"use client"` and a client component.
- **Server-side I/O or auth** → Server action in `actions/`, called from client or React Query.
- **Form and UI state** → Jotai atoms; forms use focus atoms and action atoms.
- **List of items with fine-grained updates** → `splitAtom` + pass atom to each item component.
- **Cache and request lifecycle** → React Query in `hooks/` with server actions.
- **Any button, input, modal, card surface** → shadcn from `components/ui/`.
- **New UI primitive** → Add via shadcn CLI; do not add a new UI library.

For **content and MDX**, see `content/CLAUDE.md`.

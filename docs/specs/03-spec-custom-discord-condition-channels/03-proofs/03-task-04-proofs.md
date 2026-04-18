# Proofs: Task 04 - Add Jotai store and TanStack hooks

## Planned evidence

- `ls` output showing `apps/ui/web/lib/store/discord-custom.ts` and `apps/ui/web/lib/hooks/useDiscordCustom.ts`.
- `grep` output confirming `CUSTOM_DISCORD_KEY`, `customDiscordChannelsQueryAtom`, and all five hook exports.
- Short snippet showing each mutation calls `queryClient.invalidateQueries({ queryKey: CUSTOM_DISCORD_KEY })` in `onSuccess`.
- `pnpm --filter web typecheck` output (exit 0).

## Completion notes

(Fill in after implementation)

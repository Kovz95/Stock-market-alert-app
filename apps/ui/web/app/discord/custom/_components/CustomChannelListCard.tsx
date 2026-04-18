"use client";

import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  useDeleteCustomDiscordChannel,
  useSendCustomDiscordChannelTestMessage,
  useUpdateCustomDiscordChannel,
} from "@/lib/hooks/useDiscordCustom";
import type { CustomDiscordChannel } from "@/actions/discord-custom-actions";

interface Props {
  channel: CustomDiscordChannel;
}

export function CustomChannelListCard({ channel }: Props) {
  const updateMutation = useUpdateCustomDiscordChannel();
  const deleteMutation = useDeleteCustomDiscordChannel();
  const testMutation = useSendCustomDiscordChannelTestMessage();

  async function handleToggleEnabled(checked: boolean) {
    const result = await updateMutation.mutateAsync({
      name: channel.name,
      patch: { enabled: checked },
    });
    if (!result.success) {
      toast.error(result.errorMessage ?? "Failed to update channel.");
    }
  }

  async function handleTest() {
    const result = await testMutation.mutateAsync(channel.name);
    if (result.success) {
      toast.success(`Test message sent to "${channel.name}".`);
    } else {
      toast.error(result.errorMessage ?? "Test message failed.");
    }
  }

  async function handleDelete() {
    const result = await deleteMutation.mutateAsync(channel.name);
    if (result.success) {
      toast.success(`Channel "${channel.name}" deleted.`);
    } else {
      toast.error(result.errorMessage ?? "Failed to delete channel.");
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <CardTitle className="text-base">{channel.name}</CardTitle>
            <p className="text-sm text-muted-foreground">{channel.channelName}</p>
          </div>
          <Badge variant={channel.enabled ? "default" : "secondary"}>
            {channel.enabled ? "Enabled" : "Disabled"}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {channel.description && (
          <p className="text-sm text-muted-foreground">{channel.description}</p>
        )}

        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Condition
          </p>
          <code className="block rounded bg-muted px-2.5 py-1.5 text-xs font-mono break-all">
            {channel.condition}
          </code>
        </div>

        <Separator />

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <Checkbox
              id={`enable-${channel.name}`}
              checked={channel.enabled}
              onCheckedChange={handleToggleEnabled}
              disabled={updateMutation.isPending}
            />
            <Label htmlFor={`enable-${channel.name}`} className="cursor-pointer">
              {channel.enabled ? "Enabled" : "Disabled"}
            </Label>
          </div>

          <div className="ml-auto flex gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={handleTest}
              disabled={testMutation.isPending}
            >
              {testMutation.isPending ? "Testing…" : "Test"}
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={handleDelete}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? "Deleting…" : "Delete"}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useCreateCustomDiscordChannel } from "@/lib/hooks/useDiscordCustom";
import { CustomChannelConditionInput } from "./CustomChannelConditionInput";

export function CustomChannelCreateForm() {
  const [name, setName] = useState("");
  const [webhookUrl, setWebhookUrl] = useState("");
  const [description, setDescription] = useState("");
  const [condition, setCondition] = useState("");
  const [enabled, setEnabled] = useState(true);

  const createMutation = useCreateCustomDiscordChannel();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) {
      toast.error("Channel name is required.");
      return;
    }
    if (!condition) {
      toast.error("Please select a condition.");
      return;
    }

    const result = await createMutation.mutateAsync({
      name: name.trim(),
      webhookUrl,
      description,
      condition,
      enabled,
    });

    if (result.success) {
      toast.success(`Custom channel "${name.trim()}" created.`);
      setName("");
      setWebhookUrl("");
      setDescription("");
      setCondition("");
      setEnabled(true);
    } else {
      toast.error(result.errorMessage ?? "Failed to create custom channel.");
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Create custom channel</CardTitle>
        <CardDescription>
          Route alerts that match a specific condition to a dedicated Discord
          webhook.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-2">
            <Label htmlFor="cc-name">Name</Label>
            <Input
              id="cc-name"
              placeholder="e.g. RSI Oversold Alerts"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="cc-webhook">Webhook URL</Label>
            <Input
              id="cc-webhook"
              type="password"
              placeholder="https://discord.com/api/webhooks/..."
              value={webhookUrl}
              onChange={(e) => setWebhookUrl(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="cc-description">Description</Label>
            <Textarea
              id="cc-description"
              placeholder="Optional description for this channel"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
            />
          </div>

          <Separator />

          <div className="space-y-2">
            <Label>Condition</Label>
            <CustomChannelConditionInput
              value={condition}
              onChange={setCondition}
            />
          </div>

          <Separator />

          <div className="flex items-center gap-3">
            <Checkbox
              id="cc-enabled"
              checked={enabled}
              onCheckedChange={(v) => setEnabled(Boolean(v))}
            />
            <Label htmlFor="cc-enabled">Enabled</Label>
          </div>

          <Button
            type="submit"
            disabled={createMutation.isPending}
            className="w-full"
          >
            {createMutation.isPending ? "Creating…" : "Create channel"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

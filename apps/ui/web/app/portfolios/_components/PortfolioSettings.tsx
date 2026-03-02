"use client";

import * as React from "react";
import type { Portfolio } from "../../../../../../gen/ts/alert/v1/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { useUpdatePortfolio, useDeletePortfolio } from "@/lib/hooks/usePortfolios";
import { toast } from "sonner";

type PortfolioSettingsProps = {
  portfolio: Portfolio;
  onDeleted?: () => void;
};

export function PortfolioSettings({ portfolio, onDeleted }: PortfolioSettingsProps) {
  const [name, setName] = React.useState(portfolio.name);
  const [webhook, setWebhook] = React.useState(portfolio.discordWebhook);
  const [enabled, setEnabled] = React.useState(portfolio.enabled);

  const updateMutation = useUpdatePortfolio();
  const deleteMutation = useDeletePortfolio();

  // Sync local state when portfolio prop changes
  React.useEffect(() => {
    setName(portfolio.name);
    setWebhook(portfolio.discordWebhook);
    setEnabled(portfolio.enabled);
  }, [portfolio.portfolioId, portfolio.name, portfolio.discordWebhook, portfolio.enabled]);

  const isDirty =
    name !== portfolio.name ||
    webhook !== portfolio.discordWebhook ||
    enabled !== portfolio.enabled;

  const handleSave = async () => {
    try {
      await updateMutation.mutateAsync({
        portfolioId: portfolio.portfolioId,
        name: name.trim(),
        discordWebhook: webhook.trim(),
        enabled,
      });
      toast.success("Portfolio settings saved");
    } catch (err) {
      toast.error(`Failed to save: ${err}`);
    }
  };

  const handleDelete = async () => {
    try {
      await deleteMutation.mutateAsync(portfolio.portfolioId);
      toast.success("Portfolio deleted");
      onDeleted?.();
    } catch (err) {
      toast.error(`Failed to delete: ${err}`);
    }
  };

  return (
    <div className="space-y-4 rounded-lg border p-4">
      <h3 className="font-semibold text-sm">Portfolio Settings</h3>
      <div className="space-y-3">
        <div className="space-y-1">
          <Label htmlFor="edit-name" className="text-xs">Name</Label>
          <Input
            id="edit-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="h-8"
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="edit-webhook" className="text-xs">Discord Webhook URL</Label>
          <Input
            id="edit-webhook"
            type="password"
            value={webhook}
            onChange={(e) => setWebhook(e.target.value)}
            placeholder="https://discord.com/api/webhooks/..."
            className="h-8"
          />
        </div>
        <div className="flex items-center gap-2">
          <Checkbox
            id="edit-enabled"
            checked={enabled}
            onCheckedChange={(checked) => setEnabled(checked === true)}
          />
          <Label htmlFor="edit-enabled" className="text-xs">
            Enable portfolio alerts
          </Label>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          onClick={handleSave}
          disabled={!isDirty || !name.trim() || updateMutation.isPending}
        >
          {updateMutation.isPending ? "Saving..." : "Save Changes"}
        </Button>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button size="sm" variant="destructive" disabled={deleteMutation.isPending}>
              Delete Portfolio
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete portfolio?</AlertDialogTitle>
              <AlertDialogDescription>
                This will permanently delete &quot;{portfolio.name}&quot; and remove all
                its stock associations. This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction onClick={handleDelete}>
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
}

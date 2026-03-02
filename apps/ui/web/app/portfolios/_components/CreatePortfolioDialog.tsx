"use client";

import * as React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useCreatePortfolio } from "@/lib/hooks/usePortfolios";
import { toast } from "sonner";

type CreatePortfolioDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated?: (portfolioId: string) => void;
};

export function CreatePortfolioDialog({
  open,
  onOpenChange,
  onCreated,
}: CreatePortfolioDialogProps) {
  const [name, setName] = React.useState("");
  const [webhook, setWebhook] = React.useState("");
  const createMutation = useCreatePortfolio();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    try {
      const portfolio = await createMutation.mutateAsync({
        name: name.trim(),
        discordWebhook: webhook.trim(),
      });
      toast.success(`Created portfolio: ${name.trim()}`);
      setName("");
      setWebhook("");
      onOpenChange(false);
      if (portfolio?.portfolioId) {
        onCreated?.(portfolio.portfolioId);
      }
    } catch (err) {
      toast.error(`Failed to create portfolio: ${err}`);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create Portfolio</DialogTitle>
          <DialogDescription>
            Create a new portfolio with an optional Discord webhook for alerts.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="portfolio-name">Portfolio Name</Label>
            <Input
              id="portfolio-name"
              placeholder="e.g., Tech Stocks, Dividend Portfolio"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="portfolio-webhook">Discord Webhook URL</Label>
            <Input
              id="portfolio-webhook"
              type="password"
              placeholder="https://discord.com/api/webhooks/..."
              value={webhook}
              onChange={(e) => setWebhook(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Each portfolio can send alerts to its own Discord channel.
            </p>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={!name.trim() || createMutation.isPending}
            >
              {createMutation.isPending ? "Creating..." : "Create"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

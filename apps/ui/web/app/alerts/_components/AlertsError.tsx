type AlertsErrorProps = {
  message: string;
};

export function AlertsError({ message }: AlertsErrorProps) {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <div className="text-destructive">Failed to load alerts: {message}</div>
    </div>
  );
}

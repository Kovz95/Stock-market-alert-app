import { DashboardSectionCards } from "@/components/dashboard-section-cards";
import { AlertActivityChart } from "@/components/alert-activity-chart";

export default function Page() {
  return (
    <div className="@container/main flex flex-1 flex-col gap-2">
      <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
        <DashboardSectionCards />
        <div className="px-4 lg:px-6">
          <AlertActivityChart />
        </div>
      </div>
    </div>
  );
}

"use client"

import * as React from "react"

import { NavDocuments } from "@/components/nav-documents"
import { NavMain } from "@/components/nav-main"
import { NavSecondary } from "@/components/nav-secondary"
import { NavUser } from "@/components/nav-user"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { LayoutDashboardIcon, ListIcon, ChartBarIcon, FolderIcon, UsersIcon, CameraIcon, FileTextIcon, Settings2Icon, CircleHelpIcon, SearchIcon, DatabaseIcon, FileChartColumnIcon, FileIcon, CommandIcon, BellIcon } from "lucide-react"
import { NavAlerts } from "./nav-alerts"
import Link from "next/link"

const data = {
  user: {
    name: "shadcn",
    email: "m@example.com",
    avatar: "/avatars/shadcn.jpg",
  },
  navSecondary: [
    {
      title: "Settings",
      url: "#",
      icon: (
        <Settings2Icon
        />
      ),
    },
    {
      title: "Get Help",
      url: "#",
      icon: (
        <CircleHelpIcon
        />
      ),
    },
    {
      title: "Search",
      url: "#",
      icon: (
        <SearchIcon
        />
      ),
    },
  ],
  documents: [
    {
      name: "Data Library",
      url: "#",
      icon: (
        <DatabaseIcon
        />
      ),
    },
    {
      name: "Reports",
      url: "#",
      icon: (
        <FileChartColumnIcon
        />
      ),
    },
    {
      name: "Word Assistant",
      url: "#",
      icon: (
        <FileIcon
        />
      ),
    },
  ],
  alerts: [
    {
      title: "Alerts",
      url: "/alerts",
      icon: (
        <BellIcon
        />
      ),
    },
    {
      title: "Add Alerts",
      url: "#",
      icon: (
        <ListIcon
        />
      ),
    },
    {
      title: "Alert Audit",
      url: "#",
      icon: (
        <ChartBarIcon
        />
      ),
    },
    {
      title: "Alert History",
      url: "#",
      icon: (
        <FolderIcon
        />
      ),
    },
  ],
  discord: [
    {
      title: "Hourly Discord Management",
      url: "#",
      icon: (
        <UsersIcon
        />
      ),
    },
    {
      title: "Daily Discord Management",
      url: "#",
      icon: (
        <ChartBarIcon
        />
      ),
    },
    {
      title: "Weekly Discord Management",
      url: "#",
      icon: (
        <FolderIcon
        />
      ),
    },
  ],
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              className="data-[slot=sidebar-menu-button]:p-1.5!"
            >
              <a href="#">
                <CommandIcon className="size-5!" />
                <Link href={'/'}>
                  <span className="text-base font-semibold">Kovich Stock Alerts</span>
                </Link>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.alerts} title="Alerts" />
        <NavMain items={data.discord} title="Discord" />
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter>
    </Sidebar>
  )
}

import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import {
  Activity,
  LayoutDashboard,
  MessageSquareText,
  Network,
  ListTree,
} from "lucide-react";
import { cn } from "@/lib/utils";
import Dashboard from "@/pages/Dashboard";
import Chat from "@/pages/Chat";
import Explorer from "@/pages/Explorer";
import Incidents from "@/pages/Incidents";

const navItems = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/chat", label: "Agent Chat", icon: MessageSquareText },
  { to: "/explorer", label: "Graph Explorer", icon: Network },
  { to: "/incidents", label: "Incidents", icon: ListTree },
];

export default function App() {
  return (
    <div className="flex h-full w-full bg-slate-50 text-slate-900">
      <aside className="flex w-64 flex-col border-r border-slate-200 bg-white">
        <div className="flex items-center gap-2 px-5 py-5 border-b border-slate-200">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-primary text-primary-foreground">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <div className="text-sm font-semibold">IcM GraphRAG</div>
            <div className="text-xs text-muted-foreground">Agentic Reasoning · 2026</div>
          </div>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary/10 text-primary"
                    : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
                )
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="px-5 py-3 border-t border-slate-200 text-[11px] text-muted-foreground">
          400 synthetic incidents · 2026
        </div>
      </aside>
      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/incidents" element={<Incidents />} />
        </Routes>
      </main>
    </div>
  );
}

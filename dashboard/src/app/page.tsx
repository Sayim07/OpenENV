"use client";

import { useState, useEffect, useRef } from "react";
import {
  Activity,
  Inbox,
  Settings,
  CheckCircle2,
  ShieldCheck,
  Mail,
  Command,
  Cpu,
  Zap,
  ChevronRight,
  Monitor,
  Database,
  BarChart3,
  ChevronDown,
  X,
  RefreshCw,
  Shuffle,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// ─── Types ────────────────────────────────────────────────────────────────────
interface Stats {
  total_emails: number;
  processed: number;
  success_rate: number;
  active_agent: string;
  logs: string[];
  recent_trajectories: Trajectory[];
  session_id: string;
}

interface Trajectory {
  id: string;
  task: string;
  score: string;
  time: string;
  actions?: string[];
}

// ─── Sidebar Nav Items ────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { id: "monitor", icon: Monitor, label: "Triage Monitor" },
  { id: "registry", icon: Database, label: "Task Registry" },
  { id: "analytics", icon: BarChart3, label: "Performance" },
  { id: "settings", icon: Settings, label: "Environment" },
];

// ─── View renderers ───────────────────────────────────────────────────────────
function RegistryView({ stats }: { stats: Stats }) {
  return (
    <div className="space-y-6">
      <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-8">
        <h2 className="text-lg font-bold text-slate-100 mb-6 flex items-center gap-2">
          <Database className="w-5 h-5 text-indigo-400" /> Task Registry
        </h2>
        <div className="grid grid-cols-3 gap-4">
          {["easy", "medium", "hard"].map((level) => (
            <div
              key={level}
              className="bg-slate-800/60 border border-slate-700 rounded-2xl p-5 hover:border-indigo-500/40 transition-all cursor-pointer"
            >
              <span className="text-xs font-bold uppercase tracking-widest text-slate-400">
                {level}
              </span>
              <p className="text-2xl font-bold text-slate-100 mt-2">
                {level === "easy" ? 30 : level === "medium" ? 45 : 60}
              </p>
              <p className="text-xs text-slate-500 mt-1">max steps</p>
            </div>
          ))}
        </div>
        <div className="mt-6 space-y-3">
          <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
            Registered Actions
          </p>
          {[
            "ARCHIVE",
            "LABEL_URGENT",
            "LABEL_DELEGATE",
            "DRAFT_REPLY",
            "ESCALATE",
            "FLAG_SPAM",
            "SNOOZE",
            "NO_OP",
          ].map((action) => (
            <div
              key={action}
              className="flex items-center justify-between px-4 py-2 bg-slate-800/40 rounded-xl"
            >
              <span className="text-xs font-mono text-slate-300">{action}</span>
              <span className="text-[10px] text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-md">
                active
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PerformanceView({ stats }: { stats: Stats }) {
  const bars = stats.recent_trajectories.slice(0, 6);
  return (
    <div className="space-y-6">
      <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-8">
        <h2 className="text-lg font-bold text-slate-100 mb-6 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-amber-400" /> Performance Analytics
        </h2>
        <div className="grid grid-cols-3 gap-4 mb-8">
          <div className="bg-slate-800/60 rounded-2xl p-5 border border-slate-700">
            <p className="text-xs text-slate-500 uppercase">Mean Score</p>
            <p className="text-3xl font-bold text-emerald-400 mt-1">
              {stats.success_rate}%
            </p>
          </div>
          <div className="bg-slate-800/60 rounded-2xl p-5 border border-slate-700">
            <p className="text-xs text-slate-500 uppercase">Total Steps</p>
            <p className="text-3xl font-bold text-amber-400 mt-1">
              {stats.processed}
            </p>
          </div>
          <div className="bg-slate-800/60 rounded-2xl p-5 border border-slate-700">
            <p className="text-xs text-slate-500 uppercase">Trajectories</p>
            <p className="text-3xl font-bold text-indigo-400 mt-1">
              {stats.recent_trajectories.length}
            </p>
          </div>
        </div>
        {bars.length > 0 ? (
          <div className="space-y-3">
            {bars.map((t) => (
              <div key={t.id} className="space-y-1">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>TRJ-{t.id}</span>
                  <span>{(parseFloat(t.score) * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${parseFloat(t.score) * 100}%` }}
                    transition={{ duration: 0.6 }}
                    className="h-full bg-emerald-400 rounded-full"
                  />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-slate-600 italic text-sm">
            No trajectory data yet. Run an episode to populate.
          </p>
        )}
      </div>
    </div>
  );
}

function EnvironmentView({ stats }: { stats: Stats }) {
  return (
    <div className="space-y-6">
      <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-8">
        <h2 className="text-lg font-bold text-slate-100 mb-6 flex items-center gap-2">
          <Settings className="w-5 h-5 text-slate-400" /> Environment Config
        </h2>
        <div className="space-y-4">
          {[
            { l: "Active Model", v: stats.active_agent || "None" },
            { l: "API Endpoint", v: "http://127.0.0.1:7860" },
            { l: "Observation Mode", v: "Zero-Shot" },
            { l: "Reward Function", v: "compute_reward()" },
            { l: "Max Inbox Size", v: "20 emails/reset" },
            { l: "Telemetry Interval", v: "1s polling" },
            { l: "Session ID", v: stats.session_id || "—" },
          ].map((row) => (
            <div
              key={row.l}
              className="flex justify-between items-center py-3 border-b border-slate-800 last:border-0"
            >
              <span className="text-sm text-slate-500">{row.l}</span>
              <span className="text-sm text-slate-200 font-mono">{row.v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Switch Algorithm Modal ───────────────────────────────────────────────────
const ALGORITHMS = [
  { id: "zero-shot", label: "Zero-Shot", desc: "Direct prompt → action, no examples" },
  { id: "few-shot", label: "Few-Shot", desc: "2–5 demonstrations primed in context" },
  { id: "chain-of-thought", label: "Chain-of-Thought", desc: "Reason step-by-step before action" },
  { id: "react", label: "ReAct", desc: "Reason + Act interleaved loop" },
];

function AlgorithmModal({
  current,
  onSelect,
  onClose,
}: {
  current: string;
  onSelect: (id: string) => void;
  onClose: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-[#0b1120] border border-slate-700 rounded-3xl p-8 w-[420px] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="font-bold text-slate-100 flex items-center gap-2">
            <Shuffle className="w-4 h-4 text-indigo-400" /> Switch Algorithm
          </h3>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-200 transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="space-y-3">
          {ALGORITHMS.map((algo) => (
            <button
              key={algo.id}
              onClick={() => { onSelect(algo.id); onClose(); }}
              className={`w-full text-left px-4 py-3 rounded-xl border transition-all group ${
                current === algo.id
                  ? "bg-indigo-500/10 border-indigo-500/40 text-indigo-300"
                  : "bg-slate-800/40 border-slate-700 hover:border-indigo-500/30 hover:bg-slate-800"
              }`}
            >
              <p className="text-sm font-semibold text-slate-100 group-hover:text-indigo-300 transition-colors">
                {algo.label}
              </p>
              <p className="text-[11px] text-slate-500 mt-0.5">{algo.desc}</p>
            </button>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function Home() {
  const [activeNav, setActiveNav] = useState("monitor");
  const [status, setStatus] = useState("Checking...");
  const [algorithm, setAlgorithm] = useState("zero-shot");
  const [showAlgoModal, setShowAlgoModal] = useState(false);
  const [expandedTraj, setExpandedTraj] = useState<string | null>(null);
  const [stats, setStats] = useState<Stats>({
    total_emails: 0,
    processed: 0,
    success_rate: 0,
    active_agent: "None",
    logs: [],
    recent_trajectories: [],
    session_id: "",
  });
  const logsEndRef = useRef<HTMLDivElement>(null);
  // Generated only on client to avoid SSR hydration mismatch
  const [sessionId, setSessionId] = useState("");
  useEffect(() => {
    setSessionId(
      typeof crypto !== "undefined"
        ? crypto.randomUUID().slice(0, 8)
        : Math.random().toString(36).slice(2, 10)
    );
  }, []);

  useEffect(() => {
    const checkBackend = () => {
      fetch("http://127.0.0.1:7860/stats")
        .then((res) => res.json())
        .then((data) => {
          setStatus("Operational");
          setStats({
            total_emails: data.total_emails ?? 0,
            processed: data.processed ?? 0,
            success_rate: data.success_rate ?? 0,
            active_agent: data.active_agent ?? "None",
            logs: data.logs ?? [],
            recent_trajectories: data.recent_trajectories ?? [],
            session_id: sessionId,
          });
        })
        .catch(() => setStatus("Offline"));
    };
    checkBackend();
    const interval = setInterval(checkBackend, 1000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll logs to bottom
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [stats.logs]);

  const navLabel =
    NAV_ITEMS.find((n) => n.id === activeNav)?.label ?? "Triage Monitor";

  // ── Breadcrumb task label ──
  const breadcrumbTask =
    status === "Operational"
      ? stats.active_agent !== "None"
        ? "Live Stream"
        : "Idle"
      : "Offline";

  return (
    <div className="flex h-screen bg-[#020617] text-slate-100 font-sans overflow-hidden">
      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside className="w-64 border-r border-slate-800 bg-[#020617]/50 backdrop-blur-xl flex flex-col pt-8">
        <div className="px-6 flex items-center gap-3 mb-10 overflow-hidden">
          <div className="bg-indigo-500 p-2 rounded-lg shadow-[0_0_15px_rgba(99,102,241,0.5)]">
            <Command className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-xl tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
            OpenEnv
          </span>
        </div>

        <nav className="flex-1 px-4 space-y-2">
          {NAV_ITEMS.map((item) => {
            const isActive = activeNav === item.id;
            return (
              <button
                key={item.id}
                id={`nav-${item.id}`}
                onClick={() => setActiveNav(item.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group ${
                  isActive
                    ? "bg-indigo-500/10 text-indigo-400 border border-indigo-500/20"
                    : "text-slate-400 hover:text-slate-100 hover:bg-white/5 border border-transparent"
                }`}
              >
                <item.icon
                  className={`w-4 h-4 ${
                    isActive
                      ? "text-indigo-400"
                      : "group-hover:text-indigo-400 transition-colors"
                  }`}
                />
                <span className="text-sm font-medium">{item.label}</span>
                {isActive && (
                  <div className="ml-auto w-1 h-4 bg-indigo-500 rounded-full" />
                )}
              </button>
            );
          })}
        </nav>

        <div className="p-4 mt-auto mb-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-semibold text-emerald-400 uppercase tracking-widest">
                Secure Build
              </span>
            </div>
            <p className="text-[10px] text-slate-600 leading-relaxed">
              Meta AI Hackathon · OpenEnv 1.0.0
            </p>
          </div>
        </div>
      </aside>

      {/* ── Main Content ─────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Header */}
        <header className="h-16 border-b border-slate-800 px-8 flex items-center justify-between bg-black/10 backdrop-blur-sm z-10 shrink-0">
          <div className="flex items-center gap-4">
            <h1 className="text-lg font-semibold text-slate-100">{navLabel}</h1>
            <div className="h-4 w-[1px] bg-slate-800" />
            <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
              <button
                id="breadcrumb-tasks"
                onClick={() => setActiveNav("registry")}
                className="hover:text-indigo-400 transition-colors cursor-pointer"
              >
                Tasks
              </button>
              <ChevronRight className="w-3 h-3" />
              <button
                id="breadcrumb-stream"
                onClick={() => setActiveNav("monitor")}
                className="text-indigo-400 hover:text-indigo-300 transition-colors cursor-pointer"
              >
                {breadcrumbTask}
              </button>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Session ID */}
            <span
              id="session-display"
              className="text-[10px] text-slate-600 font-mono border border-slate-800 px-2 py-1 rounded-lg"
              title="Runtime session ID (generated on page load)"
            >
              SID: {sessionId}
            </span>

            {/* Status badge */}
            <div
              id="status-badge"
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${
                status === "Operational"
                  ? "bg-emerald-500/5 text-emerald-400 border-emerald-500/20"
                  : status === "Offline"
                  ? "bg-red-500/5 text-red-400 border-red-500/20"
                  : "bg-amber-500/5 text-amber-400 border-amber-500/20"
              }`}
              title={
                status === "Operational"
                  ? "Backend API is reachable"
                  : "Backend API is unreachable — start the FastAPI server"
              }
            >
              <div
                className={`w-1.5 h-1.5 rounded-full ${
                  status === "Operational"
                    ? "bg-emerald-400 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.5)]"
                    : status === "Offline"
                    ? "bg-red-400"
                    : "bg-amber-400 animate-pulse"
                }`}
              />
              <span className="text-[10px] font-bold uppercase tracking-wider">
                {status}
              </span>
            </div>
          </div>
        </header>

        {/* Scrollable workspace */}
        <div className="flex-1 overflow-y-auto p-8 space-y-8">
          {/* ── Non-monitor views ── */}
          {activeNav === "registry" && <RegistryView stats={stats} />}
          {activeNav === "analytics" && <PerformanceView stats={stats} />}
          {activeNav === "settings" && <EnvironmentView stats={stats} />}

          {/* ── Monitor view ── */}
          {activeNav === "monitor" && (
            <>
              {/* Stat Cards */}
              <section className="grid grid-cols-1 md:grid-cols-4 gap-6">
                {[
                  {
                    label: "Inbox Size",
                    val: stats.total_emails,
                    trend: "live",
                    icon: Inbox,
                    color: "text-blue-400",
                    id: "card-inbox",
                  },
                  {
                    label: "Total Steps",
                    val: stats.processed,
                    trend: "live",
                    icon: Zap,
                    color: "text-amber-400",
                    id: "card-steps",
                  },
                  {
                    label: "Success Rate",
                    val: `${stats.success_rate}%`,
                    trend: "live",
                    icon: CheckCircle2,
                    color: "text-emerald-400",
                    id: "card-success",
                  },
                  {
                    label: "Agent",
                    val: stats.active_agent,
                    trend: "llm",
                    icon: Cpu,
                    color: "text-rose-400",
                    id: "card-agent",
                  },
                ].map((s, i) => (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.08 }}
                    key={s.id}
                    id={s.id}
                    className="bg-slate-900/40 border border-slate-800 p-5 rounded-3xl hover:border-slate-700 transition-all hover:translate-y-[-2px] hover:shadow-2xl hover:shadow-indigo-500/5 group cursor-default"
                  >
                    <div className="flex justify-between items-start mb-4">
                      <div className="p-2 rounded-xl bg-slate-800 group-hover:bg-slate-700 transition-colors">
                        <s.icon className={`w-5 h-5 ${s.color}`} />
                      </div>
                      <span className="text-[10px] font-bold text-slate-500 bg-slate-800 px-2 py-0.5 rounded-md uppercase">
                        {s.trend}
                      </span>
                    </div>
                    <h3 className="text-slate-500 text-xs font-semibold mb-1 uppercase tracking-wider">
                      {s.label}
                    </h3>
                    <p className="text-2xl font-bold text-slate-100 truncate">
                      {s.val}
                    </p>
                  </motion.div>
                ))}
              </section>

              {/* Console / Sidebar grid */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* ── Observation Stream ── */}
                <section className="lg:col-span-2 space-y-4">
                  <div className="flex items-center justify-between px-2">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded-full bg-slate-800 flex items-center justify-center">
                        <Inbox className="w-2 h-2 text-slate-400" />
                      </div>
                      <span className="text-sm font-semibold text-slate-300">
                        Observation Stream
                      </span>
                      {status === "Operational" && stats.logs.length > 0 && (
                        <span className="flex items-center gap-1 text-[10px] text-emerald-400">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />
                          LIVE
                        </span>
                      )}
                    </div>
                    <span
                      id="session-obs"
                      className="text-[10px] text-slate-600 font-mono"
                    >
                      Session: {sessionId}
                    </span>
                  </div>

                  <div
                    id="observation-stream"
                    className="bg-[#0b1120] border border-slate-800 rounded-3xl p-6 font-mono text-sm shadow-inner min-h-[400px] relative overflow-hidden"
                  >
                    {/* Window buttons */}
                    <div className="absolute top-0 right-0 p-4 flex gap-1.5 opacity-40 hover:opacity-100 transition-opacity">
                      <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
                      <div className="w-2.5 h-2.5 rounded-full bg-yellow-500" />
                      <div className="w-2.5 h-2.5 rounded-full bg-green-500" />
                    </div>

                    <div className="space-y-2 max-h-[460px] overflow-y-auto pr-1">
                      {stats.logs.length === 0 ? (
                        <p className="text-slate-600 italic">
                          No activity detected. Start the backend server and run
                          an episode to see live logs.
                        </p>
                      ) : (
                        stats.logs.map((log, idx) => (
                          <p
                            key={idx}
                            className={`leading-relaxed ${
                              log.startsWith("#") || log.startsWith("Reset")
                                ? "text-emerald-400"
                                : log.toLowerCase().includes("error") ||
                                  log.toLowerCase().includes("fail")
                                ? "text-rose-400"
                                : log.startsWith("Action:")
                                ? "text-indigo-300 border-l border-indigo-800 pl-3"
                                : "text-slate-300 border-l border-slate-800 pl-3"
                            }`}
                          >
                            <span className="text-slate-600 mr-2 select-none">
                              {String(idx + 1).padStart(3, "0")}
                            </span>
                            {log}
                          </p>
                        ))
                      )}
                      <div ref={logsEndRef} />
                    </div>
                  </div>
                </section>

                {/* ── Right Panel ── */}
                <section className="space-y-6">
                  {/* Agent Config */}
                  <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 blur-3xl pointer-events-none" />
                    <div className="flex items-center gap-3 mb-6">
                      <Cpu className="w-5 h-5 text-indigo-400" />
                      <h3 className="font-semibold text-sm">
                        Agent Configuration
                      </h3>
                    </div>
                    <div className="space-y-4">
                      {[
                        {
                          l: "Model",
                          v:
                            stats.active_agent !== "None"
                              ? stats.active_agent
                              : "—",
                        },
                        {
                          l: "Algorithm",
                          v: ALGORITHMS.find((a) => a.id === algorithm)?.label ?? algorithm,
                        },
                        { l: "Temperature", v: "0.0" },
                        { l: "Budget / Step", v: "30s" },
                      ].map((row, i) => (
                        <div
                          key={i}
                          className="flex justify-between items-center text-xs"
                        >
                          <span className="text-slate-500">{row.l}</span>
                          <span className="text-slate-200 font-medium font-mono">
                            {row.v}
                          </span>
                        </div>
                      ))}
                    </div>
                    <button
                      id="btn-switch-algorithm"
                      onClick={() => setShowAlgoModal(true)}
                      className="w-full mt-6 py-3 bg-white text-black font-bold text-[10px] uppercase tracking-widest rounded-xl hover:bg-indigo-400 hover:text-white transition-all duration-300 flex items-center justify-center gap-2"
                    >
                      <Shuffle className="w-3 h-3" />
                      Switch Algorithm
                    </button>
                  </div>

                  {/* Recent Trajectories */}
                  <div className="bg-slate-900/40 border border-slate-800 rounded-3xl p-6">
                    <div className="flex items-center justify-between mb-6">
                      <h3 className="font-semibold text-sm">
                        Recent Trajectories
                      </h3>
                      <span className="text-[10px] text-slate-600 bg-slate-800 px-2 py-0.5 rounded-md">
                        {stats.recent_trajectories.length} runs
                      </span>
                    </div>
                    <div className="space-y-3">
                      {stats.recent_trajectories.length === 0 ? (
                        <p className="text-[11px] text-slate-600 italic">
                          No trajectories yet. Run an episode to see results.
                        </p>
                      ) : (
                        stats.recent_trajectories.map((t, i) => (
                          <div key={t.id || i}>
                            <button
                              id={`traj-${t.id}`}
                              onClick={() =>
                                setExpandedTraj(
                                  expandedTraj === t.id ? null : t.id
                                )
                              }
                              className="w-full flex items-center justify-between p-3 rounded-2xl bg-slate-800/50 hover:bg-slate-800 transition-colors cursor-pointer border border-transparent hover:border-slate-700 group"
                            >
                              <div className="text-left">
                                <p className="text-xs font-bold text-slate-100 group-hover:text-indigo-400 transition-colors">
                                  TRJ-{t.id}
                                </p>
                                <p className="text-[9px] text-slate-500 uppercase tracking-tighter">
                                  {t.task}
                                </p>
                              </div>
                              <div className="flex items-center gap-3">
                                <div className="text-right">
                                  <p className="text-xs font-bold text-emerald-400">
                                    +{t.score}
                                  </p>
                                  <p className="text-[9px] text-slate-600 uppercase">
                                    {t.time}
                                  </p>
                                </div>
                                <ChevronDown
                                  className={`w-3 h-3 text-slate-500 transition-transform ${
                                    expandedTraj === t.id ? "rotate-180" : ""
                                  }`}
                                />
                              </div>
                            </button>
                            <AnimatePresence>
                              {expandedTraj === t.id && (
                                <motion.div
                                  initial={{ height: 0, opacity: 0 }}
                                  animate={{ height: "auto", opacity: 1 }}
                                  exit={{ height: 0, opacity: 0 }}
                                  transition={{ duration: 0.2 }}
                                  className="overflow-hidden"
                                >
                                  <div className="px-3 py-3 bg-slate-800/30 rounded-b-2xl text-[11px] text-slate-400 space-y-1 border border-t-0 border-slate-800">
                                    <p>
                                      <span className="text-slate-600">
                                        Score:
                                      </span>{" "}
                                      <span className="text-emerald-400 font-mono">
                                        {t.score}
                                      </span>
                                    </p>
                                    <p>
                                      <span className="text-slate-600">
                                        Task level:
                                      </span>{" "}
                                      {t.task}
                                    </p>
                                    <p>
                                      <span className="text-slate-600">
                                        Completed:
                                      </span>{" "}
                                      {t.time}
                                    </p>
                                    {t.actions && t.actions.length > 0 && (
                                      <p>
                                        <span className="text-slate-600">
                                          Actions:
                                        </span>{" "}
                                        {t.actions.join(", ")}
                                      </p>
                                    )}
                                  </div>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </section>
              </div>
            </>
          )}
        </div>
      </main>

      {/* Algorithm Modal */}
      <AnimatePresence>
        {showAlgoModal && (
          <AlgorithmModal
            current={algorithm}
            onSelect={setAlgorithm}
            onClose={() => setShowAlgoModal(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

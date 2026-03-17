import { useState, useEffect, Fragment } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'

// ─── Constants ───────────────────────────────────────────────────────────────

const API = 'https://clearpath-production-2705.up.railway.app'
const CURRENT_MONTH = 3
const MONTH_NAMES = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const MONTH_FULL  = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October', 'November', 'December']

// IBM Carbon palette
const C = {
  blue:    '#0062ff',
  blueLt:  '#4589ff',
  blueDim: 'rgba(0,98,255,0.12)',
  red:     '#da1e28',
  orange:  '#ff832b',
  yellow:  '#f1c21b',
  green:   '#24a148',
  bg:      '#070d1a',
  card:    '#0f1829',
  cardHov: '#111f38',
  border:  'rgba(30,64,120,0.4)',
  borderB: 'rgba(0,98,255,0.4)',
  txt1:    '#e8f0ff',
  txt2:    '#8da8cc',
  txt3:    '#6b83a8',
  txt4:    '#3d5278',
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function urgencyColor(s) {
  if (s >= 9) return C.red
  if (s >= 7) return C.orange
  if (s >= 4) return C.yellow
  return C.green
}
function urgencyLabel(s) {
  if (s >= 9) return 'CRITICAL'
  if (s >= 7) return 'HIGH'
  if (s >= 4) return 'MODERATE'
  return 'LOW'
}
function riskLevelColor(lvl) {
  return { critical: C.red, high: C.orange, moderate: C.yellow, low: C.green }[lvl] ?? C.txt3
}

function computeRiskScore(metrics) {
  const m = metrics.find(r => r.month === CURRENT_MONTH) ?? metrics.at(-1)
  if (!m) return 5
  const n = [
    Math.min(m.patient_fall_rate / 4, 1),
    Math.min(m.medication_error_rate / 5, 1),
    Math.min(m.readmission_rate_30d / 25, 1),
    1 - Math.min(m.handoff_documentation_score / 100, 1),
    Math.min(m.nurse_patient_ratio / 7, 1),
    Math.min(m.sepsis_response_time_minutes / 60, 1),
  ]
  return Math.round((n.reduce((a, b) => a + b, 0) / n.length) * 10)
}

function getLast6Months(metrics) {
  const result = []
  for (let i = 5; i >= 0; i--) {
    const mo = ((CURRENT_MONTH - 1 - i + 12) % 12) + 1
    const rec = metrics.find(r => r.month === mo)
    if (rec) result.push({ ...rec, label: MONTH_NAMES[mo] })
  }
  return result
}

function parseCausalChain(chain) {
  if (!chain) return []
  return chain.split(/\s*(?:→|->)\s*/).map(s => s.trim()).filter(Boolean)
}

// ─── Small shared atoms ───────────────────────────────────────────────────────

function Spinner({ size = 24, color = C.blue }) {
  return (
    <div style={{
      width: size, height: size, flexShrink: 0,
      border: `2px solid ${color}30`,
      borderTopColor: color,
      borderRadius: '50%',
      animation: 'spin 0.7s linear infinite',
      display: 'inline-block',
    }} />
  )
}

function Badge({ label, color, bg }) {
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5,
      color, background: bg ?? `${color}20`,
      border: `1px solid ${color}40`,
      borderRadius: 4, padding: '2px 7px',
    }}>
      {label}
    </span>
  )
}

// ─── Header ───────────────────────────────────────────────────────────────────

function Header({ onDashboard }) {
  return (
    <header style={{
      height: 60,
      borderBottom: `1px solid ${C.border}`,
      padding: '0 28px',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      background: 'rgba(7,13,26,0.96)',
      backdropFilter: 'blur(14px)',
      position: 'sticky', top: 0, zIndex: 100,
    }}>
      <div
        style={{ display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer', userSelect: 'none' }}
        onClick={onDashboard}
      >
        <div style={{
          width: 34, height: 34, borderRadius: 8,
          background: `linear-gradient(135deg, ${C.blue}, ${C.blueLt})`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 15, fontWeight: 800, color: '#fff',
          boxShadow: `0 0 14px ${C.blue}50`,
        }}>CP</div>
        <div>
          <div style={{ fontSize: 17, fontWeight: 700, color: C.txt1, letterSpacing: '-0.3px', lineHeight: 1 }}>ClearPath</div>
          <div style={{ fontSize: 10, color: C.txt3, marginTop: 1 }}>Nursing CE Intelligence Platform</div>
        </div>
      </div>

      <div style={{
        display: 'flex', alignItems: 'center', gap: 7,
        background: C.blueDim,
        border: `1px solid ${C.blue}40`,
        borderRadius: 20, padding: '5px 14px',
        fontSize: 11, color: C.blueLt, fontWeight: 500,
      }}>
        <div style={{
          width: 7, height: 7, borderRadius: '50%', background: C.blue,
          boxShadow: `0 0 7px ${C.blue}`,
        }} />
        Powered by IBM watsonx.ai
      </div>
    </header>
  )
}

// ─── Risk Ring (SVG donut) ────────────────────────────────────────────────────

function RiskRing({ score }) {
  const color = urgencyColor(score)
  const r = 22
  const circ = 2 * Math.PI * r
  return (
    <div style={{ position: 'relative', width: 58, height: 58, flexShrink: 0 }}>
      <svg width="58" height="58" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="29" cy="29" r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="4" />
        <circle cx="29" cy="29" r={r} fill="none" stroke={color} strokeWidth="4"
          strokeDasharray={`${(score / 10) * circ} ${circ}`}
          strokeLinecap="round"
          style={{ transition: 'stroke-dasharray 0.6s ease' }}
        />
      </svg>
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      }}>
        <span style={{ fontSize: 15, fontWeight: 800, color, lineHeight: 1 }}>{score}</span>
        <span style={{ fontSize: 8, color: C.txt3, marginTop: 1, letterSpacing: 0.5 }}>RISK</span>
      </div>
    </div>
  )
}

// ─── Department Card ──────────────────────────────────────────────────────────

function DepartmentCard({ dept, metrics, onAnalyze, analyzing }) {
  const [hov, setHov] = useState(false)
  const score = metrics ? computeRiskScore(metrics) : null
  const m = metrics?.find(r => r.month === CURRENT_MONTH)

  const quickStats = m ? [
    { label: '30d Readmit', value: `${m.readmission_rate_30d}%` },
    { label: 'Handoff Score', value: `${m.handoff_documentation_score}/100` },
    { label: 'Med Errors', value: `${m.medication_error_rate}/1k` },
    { label: 'Nurse Ratio', value: `1:${m.nurse_patient_ratio}` },
  ] : []

  return (
    <div
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: hov ? C.cardHov : C.card,
        border: `1px solid ${hov ? C.borderB : C.border}`,
        borderRadius: 12, padding: 22,
        transition: 'all 0.18s',
        display: 'flex', flexDirection: 'column', gap: 16,
      }}
    >
      {/* Top row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 600, color: C.txt1 }}>{dept.department_name}</div>
          <div style={{ fontSize: 11, color: C.txt3, marginTop: 3 }}>ID: {dept.department_id}</div>
        </div>
        {score !== null
          ? <RiskRing score={score} />
          : <div style={{ width: 58, height: 58, borderRadius: '50%', background: '#1a2540', opacity: 0.5 }}
              className="pulse" />
        }
      </div>

      {/* Quick stats grid */}
      {quickStats.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 7 }}>
          {quickStats.map(({ label, value }) => (
            <div key={label} style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.04)',
              borderRadius: 6, padding: '7px 10px',
            }}>
              <div style={{ fontSize: 10, color: C.txt3 }}>{label}</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: C.txt2, marginTop: 2 }}>{value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Analyze button */}
      <button
        onClick={() => onAnalyze(dept.department_id)}
        disabled={analyzing}
        style={{
          width: '100%', padding: '10px 0',
          background: analyzing ? `${C.blue}18` : hov ? `${C.blue}25` : `${C.blue}15`,
          border: `1px solid ${C.blue}50`,
          borderRadius: 8,
          color: analyzing ? C.blueLt : C.blue,
          fontSize: 13, fontWeight: 600,
          cursor: analyzing ? 'not-allowed' : 'pointer',
          transition: 'all 0.18s',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
        }}
      >
        {analyzing ? (
          <>
            <Spinner size={13} color={C.blueLt} />
            Analyzing with watsonx.ai…
          </>
        ) : '▶  Run AI Analysis'}
      </button>
    </div>
  )
}

// ─── Seasonal Panel ───────────────────────────────────────────────────────────

function SeasonalPanel({ risks }) {
  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 12, padding: 22,
    }}>
      <div style={{
        fontSize: 11, fontWeight: 700, color: C.txt3,
        textTransform: 'uppercase', letterSpacing: 1, marginBottom: 18,
      }}>
        Seasonal Forecast — {MONTH_FULL[CURRENT_MONTH]}
      </div>

      {risks.length === 0
        ? <div style={{ color: C.txt3, fontSize: 13 }}>No active seasonal risks.</div>
        : risks.map((risk, i) => (
          <div key={i} style={{
            borderLeft: `3px solid ${riskLevelColor(risk.risk_level)}`,
            paddingLeft: 14, marginBottom: 20,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: C.txt1 }}>{risk.name}</span>
              <Badge label={risk.risk_level} color={riskLevelColor(risk.risk_level)} />
            </div>
            <div style={{ fontSize: 11, color: C.txt3, marginBottom: 8 }}>
              {risk.affected_departments.join(' · ')}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
              {risk.ce_topic_areas.slice(0, 2).map((topic, j) => (
                <span key={j} style={{
                  fontSize: 10, color: C.blueLt,
                  background: C.blueDim,
                  border: `1px solid ${C.blue}30`,
                  borderRadius: 4, padding: '2px 7px',
                }}>{topic}</span>
              ))}
            </div>
          </div>
        ))
      }
    </div>
  )
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

function Dashboard({ departments, departmentMetrics, seasonalRisks, onAnalyze, analyzingId }) {
  return (
    <div style={{ padding: '32px 28px' }}>
      {/* Page header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700, color: C.txt1 }}>Director Dashboard</h1>
        <p style={{ margin: '6px 0 0', fontSize: 13, color: C.txt3 }}>
          Unit-level risk intelligence for {MONTH_FULL[CURRENT_MONTH]} 2025
          &nbsp;·&nbsp; {departments.length} departments monitored
        </p>
      </div>

      {/* Main layout: department grid + seasonal sidebar */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0,1fr) 300px',
        gap: 20,
        alignItems: 'start',
      }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))',
          gap: 16,
        }}>
          {departments.map(dept => (
            <DepartmentCard
              key={dept.department_id}
              dept={dept}
              metrics={departmentMetrics[dept.department_id]}
              onAnalyze={onAnalyze}
              analyzing={analyzingId === dept.department_id}
            />
          ))}
        </div>
        <SeasonalPanel risks={seasonalRisks} />
      </div>
    </div>
  )
}

// ─── Causal Chain ─────────────────────────────────────────────────────────────

function CausalChain({ chain }) {
  const steps = parseCausalChain(chain)
  if (!steps.length) return null

  const stepColors = [C.red, C.orange, C.yellow, C.green]

  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 12, padding: 22, marginBottom: 20,
    }}>
      <div style={{
        fontSize: 11, fontWeight: 700, color: C.txt3,
        textTransform: 'uppercase', letterSpacing: 1, marginBottom: 16,
      }}>
        Identified Causal Chain
      </div>
      <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
        {steps.map((step, i) => (
          <Fragment key={i}>
            <div style={{
              background: `${stepColors[i % stepColors.length]}15`,
              border: `1px solid ${stepColors[i % stepColors.length]}40`,
              borderRadius: 8, padding: '9px 18px',
              fontSize: 13, fontWeight: 600,
              color: stepColors[i % stepColors.length],
            }}>
              {step}
            </div>
            {i < steps.length - 1 && (
              <div style={{ color: C.txt4, fontSize: 20, lineHeight: 1 }}>→</div>
            )}
          </Fragment>
        ))}
      </div>
    </div>
  )
}

// ─── CE Recommendation Card ───────────────────────────────────────────────────

function CECard({ rec }) {
  const [open, setOpen] = useState(false)
  const color = urgencyColor(rec.urgency_score)

  const timing = {
    immediate:    { label: 'Immediate',    color: C.red },
    this_month:   { label: 'This Month',   color: C.orange },
    next_quarter: { label: 'Next Quarter', color: C.blueLt },
  }[rec.timing] ?? { label: rec.timing, color: C.txt3 }

  return (
    <div
      onClick={() => setOpen(o => !o)}
      style={{
        background: C.card,
        border: `1px solid ${C.border}`,
        borderLeft: `4px solid ${color}`,
        borderRadius: 12, padding: '18px 20px',
        marginBottom: 10, cursor: 'pointer',
        transition: 'background 0.15s',
      }}
      onMouseEnter={e => e.currentTarget.style.background = C.cardHov}
      onMouseLeave={e => e.currentTarget.style.background = C.card}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
        {/* Urgency badge */}
        <div style={{
          minWidth: 46, height: 46, borderRadius: 8,
          background: `${color}15`, border: `1px solid ${color}35`,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
        }}>
          <span style={{ fontSize: 17, fontWeight: 800, color, lineHeight: 1 }}>{rec.urgency_score}</span>
          <span style={{ fontSize: 8, color: `${color}90`, letterSpacing: 0.5 }}>/ 10</span>
        </div>

        {/* Content */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: C.txt1, marginBottom: 8, lineHeight: 1.4 }}>
            {rec.topic}
          </div>
          <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
            <Badge label={timing.label} color={timing.color} />
            <Badge label={`${rec.hours} CE hrs`} color={C.txt3} bg="rgba(255,255,255,0.05)" />
            <Badge label={urgencyLabel(rec.urgency_score)} color={color} />
          </div>
        </div>

        <div style={{ color: C.txt4, fontSize: 11, flexShrink: 0, paddingTop: 2 }}>
          {open ? '▲' : '▼'}
        </div>
      </div>

      {/* Expanded reasoning */}
      {open && (
        <div style={{
          marginTop: 14, paddingTop: 14,
          borderTop: `1px solid ${C.border}`,
          fontSize: 13, color: C.txt2, lineHeight: 1.65,
        }}>
          {rec.reasoning}
        </div>
      )}
    </div>
  )
}

// ─── Metrics Trend Chart ──────────────────────────────────────────────────────

const CHART_SERIES = [
  { key: 'readmission_rate_30d',       label: '30d Readmit %',   color: C.red,    axis: 'L' },
  { key: 'medication_error_rate',       label: 'Med Error /1k',   color: C.orange, axis: 'L' },
  { key: 'handoff_documentation_score', label: 'Handoff Score',   color: C.blue,   axis: 'R' },
]

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#131929',
      border: `1px solid ${C.border}`,
      borderRadius: 8, padding: '10px 14px',
      fontSize: 12, minWidth: 160,
    }}>
      <div style={{ color: C.txt1, fontWeight: 600, marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ display: 'flex', justifyContent: 'space-between', gap: 16, color: p.color, marginBottom: 3 }}>
          <span style={{ color: C.txt3 }}>{CHART_SERIES.find(s => s.key === p.dataKey)?.label}</span>
          <span style={{ fontWeight: 600 }}>{p.value}</span>
        </div>
      ))}
    </div>
  )
}

function MetricsTrend({ metrics }) {
  const data = getLast6Months(metrics)

  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 12, padding: 22, marginBottom: 20,
    }}>
      <div style={{
        fontSize: 11, fontWeight: 700, color: C.txt3,
        textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14,
      }}>
        6-Month EMR Trend
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 18, marginBottom: 16, flexWrap: 'wrap' }}>
        {CHART_SERIES.map(s => (
          <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 14, height: 3, background: s.color, borderRadius: 2 }} />
            <span style={{ fontSize: 11, color: C.txt3 }}>{s.label}</span>
          </div>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="label"
            tick={{ fill: C.txt3, fontSize: 11 }}
            axisLine={false} tickLine={false}
          />
          <YAxis
            yAxisId="L"
            tick={{ fill: C.txt3, fontSize: 11 }}
            axisLine={false} tickLine={false}
            domain={[0, 30]} width={32}
          />
          <YAxis
            yAxisId="R"
            orientation="right"
            tick={{ fill: C.txt3, fontSize: 11 }}
            axisLine={false} tickLine={false}
            domain={[50, 100]} width={36}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine yAxisId="L" y={15} stroke={`${C.red}40`} strokeDasharray="4 4" />
          <Line yAxisId="L" type="monotone" dataKey="readmission_rate_30d" stroke={C.red}    strokeWidth={2} dot={false} />
          <Line yAxisId="L" type="monotone" dataKey="medication_error_rate" stroke={C.orange} strokeWidth={2} dot={false} />
          <Line yAxisId="R" type="monotone" dataKey="handoff_documentation_score" stroke={C.blue} strokeWidth={2} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
      <div style={{ fontSize: 10, color: C.txt4, marginTop: 6 }}>
        Dashed red line = 15% readmission threshold
      </div>
    </div>
  )
}

// ─── Department Detail ────────────────────────────────────────────────────────

function DepartmentDetail({ analysis, metrics, onBack }) {
  return (
    <div style={{ padding: '32px 28px', maxWidth: 920, margin: '0 auto' }}>
      {/* Back */}
      <button
        onClick={onBack}
        style={{
          background: 'none', border: `1px solid ${C.border}`,
          color: C.txt3, borderRadius: 8, padding: '7px 16px',
          cursor: 'pointer', fontSize: 13, marginBottom: 28,
          display: 'inline-flex', alignItems: 'center', gap: 7,
          transition: 'all 0.15s',
        }}
        onMouseEnter={e => { e.currentTarget.style.color = C.txt1; e.currentTarget.style.borderColor = C.borderB }}
        onMouseLeave={e => { e.currentTarget.style.color = C.txt3; e.currentTarget.style.borderColor = C.border }}
      >
        ← Back to Dashboard
      </button>

      {/* Department header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ fontSize: 11, color: C.blue, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 5 }}>
          AI Analysis · {MONTH_FULL[CURRENT_MONTH]} 2025
        </div>
        <h1 style={{ margin: '0 0 14px', fontSize: 26, fontWeight: 700, color: C.txt1 }}>
          {analysis.department_name}
        </h1>

        {/* Risk summary */}
        <div style={{
          background: C.card, border: `1px solid ${C.border}`,
          borderRadius: 12, padding: '18px 22px',
          fontSize: 14, color: C.txt2, lineHeight: 1.7,
        }}>
          {analysis.risk_summary}
        </div>
      </div>

      {/* Causal chain */}
      {analysis.causal_chain && <CausalChain chain={analysis.causal_chain} />}

      {/* 6-month trend */}
      {metrics && <MetricsTrend metrics={metrics} />}

      {/* CE Recommendations */}
      <div>
        <div style={{
          fontSize: 11, fontWeight: 700, color: C.txt3,
          textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14,
        }}>
          CE Recommendations &nbsp;·&nbsp; {analysis.recommendations.length} items
        </div>
        {[...analysis.recommendations]
          .sort((a, b) => b.urgency_score - a.urgency_score)
          .map((rec, i) => <CECard key={i} rec={rec} />)
        }
      </div>
    </div>
  )
}

// ─── Loading / Error screens ──────────────────────────────────────────────────

function LoadingScreen() {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      height: 'calc(100vh - 60px)', gap: 16,
    }}>
      <Spinner size={40} />
      <div style={{ color: C.txt3, fontSize: 14 }}>Connecting to ClearPath backend…</div>
    </div>
  )
}

function ErrorScreen({ message }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: 'calc(100vh - 60px)',
    }}>
      <div style={{
        background: C.card, border: `1px solid ${C.red}40`,
        borderRadius: 12, padding: 36, maxWidth: 420, textAlign: 'center',
      }}>
        <div style={{ fontSize: 36, marginBottom: 14 }}>⚠</div>
        <div style={{ fontSize: 15, fontWeight: 600, color: C.red, marginBottom: 8 }}>Connection Error</div>
        <div style={{ fontSize: 13, color: C.txt3, lineHeight: 1.6 }}>{message}</div>
      </div>
    </div>
  )
}

// ─── Root App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [view, setView]               = useState('dashboard')
  const [departments, setDepartments] = useState([])
  const [deptMetrics, setDeptMetrics] = useState({})
  const [seasonalRisks, setSeasonalRisks] = useState([])
  const [selected, setSelected]       = useState(null)   // { analysis, metrics }
  const [analyzingId, setAnalyzingId] = useState(null)
  const [loading, setLoading]         = useState(true)
  const [error, setError]             = useState(null)

  // Initial data load
  useEffect(() => {
    async function init() {
      try {
        const [depts, seasonal] = await Promise.all([
          fetch(`${API}/departments`).then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json() }),
          fetch(`${API}/seasonal/${CURRENT_MONTH}`).then(r => r.json()),
        ])
        setDepartments(depts)
        setSeasonalRisks(seasonal.seasonal_risks ?? [])

        // Fetch all department metrics in parallel
        const entries = await Promise.all(
          depts.map(d =>
            fetch(`${API}/departments/${d.department_id}/metrics`)
              .then(r => r.json())
              .then(data => [d.department_id, data.metrics ?? []])
              .catch(() => [d.department_id, []])
          )
        )
        setDeptMetrics(Object.fromEntries(entries))
      } catch (err) {
        setError(`Failed to connect to the ClearPath backend (${err.message}). Check that the Railway service is running.`)
      } finally {
        setLoading(false)
      }
    }
    init()
  }, [])

  async function handleAnalyze(deptId) {
    setAnalyzingId(deptId)
    try {
      const res = await fetch(`${API}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ department_id: deptId, current_month: CURRENT_MONTH }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      const analysis = await res.json()
      setSelected({ analysis, metrics: deptMetrics[deptId] ?? [] })
      setView('detail')
    } catch (err) {
      alert(`Analysis failed: ${err.message}`)
    } finally {
      setAnalyzingId(null)
    }
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: C.bg,
      color: C.txt1,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "IBM Plex Sans", sans-serif',
    }}>
      <Header onDashboard={() => setView('dashboard')} />

      {loading ? (
        <LoadingScreen />
      ) : error ? (
        <ErrorScreen message={error} />
      ) : view === 'detail' && selected ? (
        <DepartmentDetail
          analysis={selected.analysis}
          metrics={selected.metrics}
          onBack={() => setView('dashboard')}
        />
      ) : (
        <Dashboard
          departments={departments}
          departmentMetrics={deptMetrics}
          seasonalRisks={seasonalRisks}
          onAnalyze={handleAnalyze}
          analyzingId={analyzingId}
        />
      )}

      <style>{`
        @keyframes spin  { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:.3 } 50% { opacity:.7 } }

        @media (max-width: 768px) {
          /* Stack seasonal panel below department grid */
          .dashboard-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  )
}

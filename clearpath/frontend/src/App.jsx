import { useState, useEffect, Fragment } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'

// ─── Constants ───────────────────────────────────────────────────────────────

const API = 'https://clearpath-production-2705.up.railway.app'
const MONTH_NAMES = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const MONTH_FULL  = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                         'July', 'August', 'September', 'October', 'November', 'December']

const KNOWN_STANDARDS = [
  'I-PASS', 'AHRQ', 'NQF', 'CMS', 'ANA', 'Leapfrog', 'Joint Commission',
  'SEP-1', 'HRRP', 'NPSG', 'TeamSTEPPS', 'CUSP', 'CLABSI', 'SBAR',
  'Safe Practice', 'Morse', 'NEWS2', 'MEWS', 'SOFA', 'GINA',
]

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

function computeRiskScore(metrics, month) {
  const m = metrics.find(r => r.month === month) ?? metrics.at(-1)
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

function getLast6Months(metrics, currentMonth) {
  const result = []
  for (let i = 5; i >= 0; i--) {
    const mo = ((currentMonth - 1 - i + 12) % 12) + 1
    const rec = metrics.find(r => r.month === mo)
    if (rec) result.push({ ...rec, label: MONTH_NAMES[mo] })
  }
  return result
}

function parseCausalChain(chain) {
  if (!chain) return []
  return chain.split(/\s*(?:→|->)\s*/).map(s => s.trim()).filter(Boolean)
}

function getWorstMetric(m, benchmarks) {
  if (!m || !benchmarks) return null
  let worst = null
  let worstGap = 0
  for (const [key, bench] of Object.entries(benchmarks)) {
    const val = m[key]
    if (val == null) continue
    const gap = key === 'handoff_documentation_score'
      ? bench.target - val
      : val - bench.target
    if (gap > worstGap) {
      worstGap = gap
      worst = { key, label: bench.label, value: val, target: bench.target, gap, source: bench.source, higherIsBetter: key === 'handoff_documentation_score' }
    }
  }
  return worst
}

function findReferenceBadges(text) {
  if (!text) return []
  const found = []
  for (const std of KNOWN_STANDARDS) {
    if (text.includes(std)) found.push(std)
  }
  return [...new Set(found)]
}

function key_format(key, val) {
  if (key === 'readmission_rate_30d') return `${val}%`
  if (key === 'medication_error_rate') return `${val}/1k`
  if (key === 'nurse_patient_ratio') return `1:${val}`
  if (key === 'sepsis_response_time_minutes') return `${val} min`
  if (key === 'handoff_documentation_score') return `${val}/100`
  return `${val}`
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

function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 11, fontWeight: 700, color: C.txt3,
      textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14,
    }}>
      {children}
    </div>
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

// ─── Orchestrate Banner ──────────────────────────────────────────────────────

function OrchestrateBanner() {
  return (
    <div style={{
      background: `linear-gradient(135deg, ${C.blue}12, ${C.blueLt}08)`,
      border: `1px solid ${C.blue}25`,
      borderRadius: 10,
      padding: '12px 20px',
      display: 'flex', alignItems: 'center', gap: 12,
      marginBottom: 24,
    }}>
      <div style={{
        width: 32, height: 32, borderRadius: 8,
        background: `${C.blue}18`,
        border: `1px solid ${C.blue}30`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 14, flexShrink: 0,
      }}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path d="M8 1C4.134 1 1 4.134 1 8s3.134 7 7 7 7-3.134 7-7-3.134-7-7-7zm0 12.5A5.506 5.506 0 012.5 8 5.506 5.506 0 018 2.5 5.506 5.506 0 0113.5 8 5.506 5.506 0 018 13.5z" fill={C.blueLt}/>
          <path d="M8 4.5a1 1 0 00-1 1v3a1 1 0 001 1h2.5a1 1 0 000-2H9V5.5a1 1 0 00-1-1z" fill={C.blueLt}/>
        </svg>
      </div>
      <div>
        <div style={{ fontSize: 13, color: C.txt1, fontWeight: 500 }}>
          Conversational interface powered by IBM watsonx Orchestrate — available via dedicated agent portal
        </div>
      </div>
    </div>
  )
}

// ─── Month Selector ──────────────────────────────────────────────────────────

function MonthSelector({ currentMonth, onChange }) {
  return (
    <div style={{
      display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 24,
    }}>
      {MONTH_NAMES.slice(1).map((name, i) => {
        const mo = i + 1
        const active = mo === currentMonth
        return (
          <button
            key={mo}
            onClick={() => onChange(mo)}
            style={{
              padding: '6px 14px',
              borderRadius: 20,
              border: active ? `1px solid ${C.blue}` : `1px solid ${C.border}`,
              background: active ? `${C.blue}20` : 'transparent',
              color: active ? C.blue : C.txt3,
              fontSize: 12, fontWeight: active ? 700 : 500,
              cursor: 'pointer',
              transition: 'all 0.15s',
            }}
          >
            {name}
          </button>
        )
      })}
    </div>
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

// ─── Department Card ─────────────────────────────────────────────────────────

function DepartmentCard({ dept, metrics, benchmarks, onAnalyze, analyzing, currentMonth }) {
  const [hov, setHov] = useState(false)
  const score = metrics ? computeRiskScore(metrics, currentMonth) : null
  const m = metrics?.find(r => r.month === currentMonth)
  const worst = m && benchmarks ? getWorstMetric(m, benchmarks) : null

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
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 600, color: C.txt1 }}>{dept.department_name}</div>
          <div style={{ fontSize: 11, color: C.txt3, marginTop: 3 }}>ID: {dept.department_id}</div>
        </div>
        {score !== null
          ? <RiskRing score={score} />
          : <div style={{ width: 58, height: 58, borderRadius: '50%', background: '#1a2540', opacity: 0.5 }}
              className="pulse" />
        }
      </div>

      {worst && (
        <div style={{
          background: `${worst.gap > 0 ? C.red : C.green}08`,
          border: `1px solid ${worst.gap > 0 ? C.red : C.green}25`,
          borderRadius: 8, padding: '10px 14px',
        }}>
          <div style={{ fontSize: 11, color: C.txt3, marginBottom: 4 }}>{worst.label}</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span style={{ fontSize: 18, fontWeight: 700, color: worst.gap > 0 ? C.red : C.green }}>
              {worst.higherIsBetter ? `${worst.value}/100` : key_format(worst.key, worst.value)}
            </span>
            <span style={{ fontSize: 11, color: C.txt3 }}>
              target: {worst.higherIsBetter ? `${worst.target}/100` : key_format(worst.key, worst.target)}
            </span>
          </div>
          <div style={{ fontSize: 10, color: C.txt4, marginTop: 3 }}>{worst.source}</div>
        </div>
      )}

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
            Generating report via watsonx.ai…
          </>
        ) : 'Generate CE Report'}
      </button>
    </div>
  )
}

// ─── Seasonal Panel ──────────────────────────────────────────────────────────

function SeasonalPanel({ risks, currentMonth }) {
  return (
    <div style={{ marginTop: 8 }}>
      <SectionLabel>Seasonal Forecast — {MONTH_FULL[currentMonth]}</SectionLabel>

      {risks.length === 0
        ? <div style={{ color: C.txt3, fontSize: 14 }}>No active seasonal risks.</div>
        : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: 14,
          }}>
            {risks.map((risk, i) => (
              <div key={i} style={{
                background: C.card,
                border: `1px solid ${C.border}`,
                borderLeft: `4px solid ${riskLevelColor(risk.risk_level)}`,
                borderRadius: 10, padding: '16px 18px',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 14, fontWeight: 600, color: C.txt1 }}>{risk.name}</span>
                  <Badge label={risk.risk_level} color={riskLevelColor(risk.risk_level)} />
                </div>
                <div style={{ fontSize: 13, color: C.txt2, lineHeight: 1.5 }}>
                  {risk.description.length > 120 ? risk.description.slice(0, 120) + '…' : risk.description}
                </div>
              </div>
            ))}
          </div>
        )
      }
    </div>
  )
}

// ─── Summary Banner ──────────────────────────────────────────────────────────

function SummaryBanner({ departments, departmentMetrics, seasonalRisks, currentMonth }) {
  const elevatedCount = departments.filter(dept => {
    const metrics = departmentMetrics[dept.department_id]
    return metrics && computeRiskScore(metrics, currentMonth) >= 6
  }).length

  const totalCEHours = departments.reduce((sum, dept) => {
    const metrics = departmentMetrics[dept.department_id]
    if (!metrics) return sum
    const score = computeRiskScore(metrics, currentMonth)
    return sum + (score >= 7 ? 8 : score >= 4 ? 4 : 2)
  }, 0)

  return (
    <div style={{
      display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24,
    }}>
      {[
        { value: elevatedCount, label: 'departments at elevated risk', color: elevatedCount > 0 ? C.orange : C.green },
        { value: seasonalRisks.length, label: 'seasonal alerts active', color: seasonalRisks.length > 2 ? C.orange : C.blue },
        { value: `${totalCEHours}+`, label: 'CE hours recommended', color: C.blue },
      ].map(({ value, label, color }) => (
        <div key={label} style={{
          background: `${color}10`,
          border: `1px solid ${color}30`,
          borderRadius: 10, padding: '12px 20px',
          display: 'flex', alignItems: 'center', gap: 10,
          flex: '1 1 200px',
        }}>
          <span style={{ fontSize: 22, fontWeight: 800, color }}>{value}</span>
          <span style={{ fontSize: 13, color: C.txt2 }}>{label}</span>
        </div>
      ))}
    </div>
  )
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

function Dashboard({ departments, departmentMetrics, departmentBenchmarks, seasonalRisks, onAnalyze, analyzingId, currentMonth, onMonthChange }) {
  return (
    <div style={{ padding: '32px 28px', maxWidth: 1200, margin: '0 auto' }}>
      {/* Page header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ margin: 0, fontSize: 28, fontWeight: 700, color: C.txt1 }}>Director Dashboard</h1>
        <p style={{ margin: '6px 0 0', fontSize: 14, color: C.txt3 }}>
          Unit-level risk intelligence for {MONTH_FULL[currentMonth]} 2025
          &nbsp;·&nbsp; {departments.length} departments monitored
        </p>
      </div>

      {/* Orchestrate banner */}
      <OrchestrateBanner />

      {/* Month selector */}
      <MonthSelector currentMonth={currentMonth} onChange={onMonthChange} />

      {/* Summary stats */}
      <SummaryBanner
        departments={departments}
        departmentMetrics={departmentMetrics}
        seasonalRisks={seasonalRisks}
        currentMonth={currentMonth}
      />

      {/* Department cards grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
        gap: 16,
      }}>
        {departments.map(dept => (
          <DepartmentCard
            key={dept.department_id}
            dept={dept}
            metrics={departmentMetrics[dept.department_id]}
            benchmarks={departmentBenchmarks[dept.department_id]}
            onAnalyze={onAnalyze}
            analyzing={analyzingId === dept.department_id}
            currentMonth={currentMonth}
          />
        ))}
      </div>

      {/* Seasonal forecast */}
      <div style={{ marginTop: 32 }}>
        <SeasonalPanel risks={seasonalRisks} currentMonth={currentMonth} />
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
      <SectionLabel>Identified Causal Chain</SectionLabel>
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
  const refBadges = findReferenceBadges(rec.reasoning)

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
        <div style={{
          minWidth: 46, height: 46, borderRadius: 8,
          background: `${color}15`, border: `1px solid ${color}35`,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
        }}>
          <span style={{ fontSize: 17, fontWeight: 800, color, lineHeight: 1 }}>{rec.urgency_score}</span>
          <span style={{ fontSize: 8, color: `${color}90`, letterSpacing: 0.5 }}>/ 10</span>
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: C.txt1, marginBottom: 8, lineHeight: 1.4 }}>
            {rec.topic}
          </div>
          <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>
            <Badge label={timing.label} color={timing.color} />
            <Badge label={urgencyLabel(rec.urgency_score)} color={color} />
          </div>
        </div>

        <div style={{ color: C.txt4, fontSize: 11, flexShrink: 0, paddingTop: 2 }}>
          {open ? '▲' : '▼'}
        </div>
      </div>

      {open && (
        <div style={{
          marginTop: 16, paddingTop: 16,
          borderTop: `1px solid ${C.border}`,
        }}>
          <div style={{ fontSize: 14, color: C.txt2, lineHeight: 1.75 }}>
            {rec.reasoning}
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
            <Badge label={`${rec.hours} CE hrs`} color={C.txt3} bg="rgba(255,255,255,0.05)" />
          </div>
          {refBadges.length > 0 && (
            <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginTop: 10 }}>
              {refBadges.map(ref => (
                <span key={ref} style={{
                  fontSize: 10, fontWeight: 600,
                  color: C.blueLt,
                  background: `${C.blue}15`,
                  border: `1px solid ${C.blue}30`,
                  borderRadius: 4, padding: '2px 8px',
                }}>
                  {ref}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Benchmark Comparison Bars ───────────────────────────────────────────────

function BenchmarkBars({ metrics, benchmarks, currentMonth }) {
  const m = metrics?.find(r => r.month === currentMonth)
  if (!m || !benchmarks || Object.keys(benchmarks).length === 0) return null

  const items = Object.entries(benchmarks).map(([key, bench]) => {
    const actual = m[key]
    if (actual == null) return null
    const higherIsBetter = key === 'handoff_documentation_score'
    const passing = higherIsBetter ? actual >= bench.target : actual <= bench.target
    return { key, ...bench, actual, passing, higherIsBetter }
  }).filter(Boolean)

  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 12, padding: 22, marginBottom: 20,
    }}>
      <SectionLabel>Benchmark Comparison</SectionLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {items.map(item => {
          const maxVal = item.higherIsBetter
            ? Math.max(item.actual, item.target, 100)
            : Math.max(item.actual, item.target) * 1.3
          const actualPct = (item.actual / maxVal) * 100
          const targetPct = (item.target / maxVal) * 100

          return (
            <div key={item.key}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                <span style={{ fontSize: 13, color: C.txt2 }}>{item.label}</span>
                <span style={{ fontSize: 12, color: C.txt3 }}>
                  {key_format(item.key, item.actual)} vs {key_format(item.key, item.target)} ({item.source})
                </span>
              </div>
              <div style={{
                position: 'relative', height: 8,
                background: 'rgba(255,255,255,0.04)',
                borderRadius: 4, overflow: 'visible',
              }}>
                <div style={{
                  height: '100%',
                  width: `${Math.min(actualPct, 100)}%`,
                  background: item.passing ? C.green : C.red,
                  borderRadius: 4,
                  transition: 'width 0.4s ease',
                  opacity: 0.7,
                }} />
                <div style={{
                  position: 'absolute',
                  left: `${Math.min(targetPct, 100)}%`,
                  top: -3, height: 14, width: 2,
                  background: C.txt3,
                  borderRadius: 1,
                }} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── SOAP Notes Section ──────────────────────────────────────────────────────

function SOAPNotesSection({ notes }) {
  const [expanded, setExpanded] = useState(false)
  if (!notes || notes.length === 0) return null

  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 12, padding: 22, marginBottom: 20,
    }}>
      <div
        onClick={() => setExpanded(e => !e)}
        style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
      >
        <SectionLabel>Clinical Documentation Samples · {notes.length} notes</SectionLabel>
        <span style={{ color: C.txt4, fontSize: 11 }}>{expanded ? '▲' : '▼'}</span>
      </div>

      {expanded && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 8 }}>
          {notes.map(note => (
            <div key={note.note_id} style={{
              border: `1px solid ${C.border}`,
              borderRadius: 10, padding: 18,
              background: 'rgba(255,255,255,0.015)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
                <div>
                  <span style={{ fontSize: 13, fontWeight: 600, color: C.txt1 }}>{note.note_type}</span>
                  <span style={{ fontSize: 12, color: C.txt3, marginLeft: 10 }}>{note.note_id}</span>
                </div>
                <div style={{ fontSize: 12, color: C.txt3 }}>
                  {note.date} · {note.provider}
                </div>
              </div>

              {[
                { label: 'S', content: note.subjective, color: C.blue },
                { label: 'O', content: note.objective, color: C.orange },
                { label: 'A', content: note.assessment, color: C.yellow },
                { label: 'P', content: note.plan, color: C.green },
              ].map(({ label, content, color }) => (
                <div key={label} style={{ marginBottom: 10 }}>
                  <div style={{
                    display: 'inline-block',
                    fontSize: 10, fontWeight: 800,
                    color, background: `${color}15`,
                    border: `1px solid ${color}30`,
                    borderRadius: 3, padding: '1px 6px',
                    marginBottom: 4,
                  }}>
                    {label}
                  </div>
                  <div style={{ fontSize: 13, color: C.txt2, lineHeight: 1.6, paddingLeft: 4 }}>
                    {content}
                  </div>
                </div>
              ))}
            </div>
          ))}
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

function MetricsTrend({ metrics, currentMonth }) {
  const data = getLast6Months(metrics, currentMonth)

  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 12, padding: 22, marginBottom: 20,
    }}>
      <SectionLabel>6-Month EMR Trend</SectionLabel>

      <div style={{ display: 'flex', gap: 18, marginBottom: 16, flexWrap: 'wrap' }}>
        {CHART_SERIES.map(s => (
          <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 14, height: 3, background: s.color, borderRadius: 2 }} />
            <span style={{ fontSize: 11, color: C.txt3 }}>{s.label}</span>
          </div>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis dataKey="label" tick={{ fill: C.txt3, fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis yAxisId="L" tick={{ fill: C.txt3, fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, 30]} width={32} />
          <YAxis yAxisId="R" orientation="right" tick={{ fill: C.txt3, fontSize: 11 }} axisLine={false} tickLine={false} domain={[50, 100]} width={36} />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine yAxisId="L" y={15} stroke={`${C.red}40`} strokeDasharray="4 4" />
          <Line yAxisId="L" type="monotone" dataKey="readmission_rate_30d" stroke={C.red} strokeWidth={2} dot={false} />
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

function DepartmentDetail({ analysis, metrics, soapNotes, benchmarks, onBack, currentMonth }) {
  return (
    <div style={{ padding: '32px 28px', maxWidth: 960, margin: '0 auto' }}>
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

      <div style={{ marginBottom: 28 }}>
        <div style={{ fontSize: 11, color: C.blue, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 5 }}>
          AI Analysis · {MONTH_FULL[currentMonth]} 2025
        </div>
        <h1 style={{ margin: '0 0 16px', fontSize: 28, fontWeight: 700, color: C.txt1 }}>
          {analysis.department_name}
        </h1>

        <div style={{
          background: C.card, border: `1px solid ${C.border}`,
          borderRadius: 12, padding: '20px 24px',
          fontSize: 14, color: C.txt2, lineHeight: 1.8,
        }}>
          {analysis.risk_summary}
        </div>
      </div>

      {analysis.causal_chain && <CausalChain chain={analysis.causal_chain} />}

      {benchmarks && <BenchmarkBars metrics={metrics} benchmarks={benchmarks} currentMonth={currentMonth} />}

      {metrics && <MetricsTrend metrics={metrics} currentMonth={currentMonth} />}

      <SOAPNotesSection notes={soapNotes} />

      <div style={{ marginTop: 8 }}>
        <SectionLabel>CE Recommendations · {analysis.recommendations.length} items</SectionLabel>
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
        <div style={{ fontSize: 36, marginBottom: 14 }}>!</div>
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
  const [deptBenchmarks, setDeptBenchmarks] = useState({})
  const [deptSoapNotes, setDeptSoapNotes] = useState({})
  const [seasonalRisks, setSeasonalRisks] = useState([])
  const [selected, setSelected]       = useState(null)
  const [analyzingId, setAnalyzingId] = useState(null)
  const [loading, setLoading]         = useState(true)
  const [error, setError]             = useState(null)
  const [currentMonth, setCurrentMonth] = useState(3)

  useEffect(() => {
    async function init() {
      try {
        const [depts, seasonal] = await Promise.all([
          fetch(`${API}/departments`).then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json() }),
          fetch(`${API}/seasonal/${currentMonth}`).then(r => r.json()),
        ])
        setDepartments(depts)
        setSeasonalRisks(seasonal.seasonal_risks ?? [])

        const [metricsEntries, benchmarkEntries, soapEntries] = await Promise.all([
          Promise.all(
            depts.map(d =>
              fetch(`${API}/departments/${d.department_id}/metrics`)
                .then(r => r.json())
                .then(data => [d.department_id, data.metrics ?? []])
                .catch(() => [d.department_id, []])
            )
          ),
          Promise.all(
            depts.map(d =>
              fetch(`${API}/departments/${d.department_id}/benchmarks`)
                .then(r => r.json())
                .then(data => [d.department_id, data.benchmarks ?? {}])
                .catch(() => [d.department_id, {}])
            )
          ),
          Promise.all(
            depts.map(d =>
              fetch(`${API}/departments/${d.department_id}/soap-notes`)
                .then(r => r.json())
                .then(data => [d.department_id, data.soap_notes ?? []])
                .catch(() => [d.department_id, []])
            )
          ),
        ])
        setDeptMetrics(Object.fromEntries(metricsEntries))
        setDeptBenchmarks(Object.fromEntries(benchmarkEntries))
        setDeptSoapNotes(Object.fromEntries(soapEntries))
      } catch (err) {
        setError(`Failed to connect to the ClearPath backend (${err.message}). Check that the Railway service is running.`)
      } finally {
        setLoading(false)
      }
    }
    init()
  }, [])

  useEffect(() => {
    if (loading) return
    fetch(`${API}/seasonal/${currentMonth}`)
      .then(r => r.json())
      .then(data => setSeasonalRisks(data.seasonal_risks ?? []))
      .catch(() => {})
  }, [currentMonth, loading])

  async function handleAnalyze(deptId) {
    setAnalyzingId(deptId)
    try {
      const res = await fetch(`${API}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ department_id: deptId, current_month: currentMonth }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      const analysis = await res.json()
      setSelected({
        analysis,
        metrics: deptMetrics[deptId] ?? [],
        soapNotes: deptSoapNotes[deptId] ?? [],
        benchmarks: deptBenchmarks[deptId] ?? {},
      })
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
          soapNotes={selected.soapNotes}
          benchmarks={selected.benchmarks}
          onBack={() => setView('dashboard')}
          currentMonth={currentMonth}
        />
      ) : (
        <Dashboard
          departments={departments}
          departmentMetrics={deptMetrics}
          departmentBenchmarks={deptBenchmarks}
          seasonalRisks={seasonalRisks}
          onAnalyze={handleAnalyze}
          analyzingId={analyzingId}
          currentMonth={currentMonth}
          onMonthChange={setCurrentMonth}
        />
      )}

      <style>{`
        @keyframes spin  { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:.3 } 50% { opacity:.7 } }
      `}</style>
    </div>
  )
}

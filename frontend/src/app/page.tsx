'use client';

import { useState } from 'react';
import { AlertTriangle, ChevronRight, ChevronLeft, Upload, RotateCcw, Maximize2, Box } from 'lucide-react';

const issues = [
  { id: 'ISS-041', severity: 'HIGH', floor: 'Floor 1',  room: 'HVAC Room',        desc: 'Beams placed at wrong angle — 12° offset from spec' },
  { id: 'ISS-040', severity: 'MED',  floor: 'Floor 3',  room: 'East Stairwell',   desc: 'Rebar spacing non-compliant, 8" vs required 6"' },
  { id: 'ISS-039', severity: 'HIGH', floor: 'Floor 2',  room: 'Utility Chase',    desc: 'Conduit penetration missing fire-stop sealant' },
  { id: 'ISS-038', severity: 'LOW',  floor: 'Roof',     room: 'Parapet Section',  desc: 'Anchor bolt torque below spec on 4 of 12 bolts' },
  { id: 'ISS-037', severity: 'MED',  floor: 'Floor 1',  room: 'Lobby',            desc: 'Concrete pour — honeycombing on column C4' },
  { id: 'ISS-036', severity: 'HIGH', floor: 'Basement', room: 'Pump Room',        desc: 'Waterproofing membrane gap, 2.3m section' },
  { id: 'ISS-035', severity: 'LOW',  floor: 'Floor 4',  room: 'Office Zone B',    desc: 'Ceiling grid misaligned, 15mm cumulative drift' },
  { id: 'ISS-034', severity: 'MED',  floor: 'Floor 2',  room: 'Mechanical Room',  desc: 'Pipe support bracket missing at grid line 7' },
];

const SEV = {
  HIGH: { color: '#b91c1c', bg: 'rgba(185,28,28,0.08)', border: 'rgba(185,28,28,0.25)', label: '#b91c1c' },
  MED:  { color: '#b45309', bg: 'rgba(180,83,9,0.08)',  border: 'rgba(180,83,9,0.25)',  label: '#b45309' },
  LOW:  { color: '#1a56db', bg: 'rgba(26,86,219,0.07)', border: 'rgba(26,86,219,0.22)', label: '#1a56db' },
};

const cards = [
  {
    label: 'Floor 1 — HVAC Room · Beam Alignment',
    before: { tag: 'EXPECTED', color: '#0a7c4e', desc: 'Structural drawing: correct 90° beam placement per spec' },
    after:  { tag: 'ACTUAL',   color: '#b91c1c', desc: 'Site capture: beam at 12° offset — visible misalignment' },
  },
  {
    label: 'Floor 3 — East Stairwell · Rebar Spacing',
    before: { tag: 'SPEC',     color: '#1a56db', desc: 'Blueprint standard: 6" rebar centre-to-centre spacing' },
    after:  { tag: 'CAPTURED', color: '#b45309', desc: 'Captured image: 8" spacing — non-compliant by 33%' },
  },
  {
    label: 'Basement — Pump Room · Waterproofing',
    before: { tag: 'DESIGN',   color: '#0a7c4e', desc: 'Design document: continuous waterproof membrane coverage' },
    after:  { tag: 'FLAGGED',  color: '#b91c1c', desc: 'Drone scan: 2.3m gap detected in membrane' },
  },
  {
    label: 'Floor 2 — Column C4 · Concrete Quality',
    before: { tag: 'REFERENCE', color: '#0a7c4e', desc: 'Reference standard: smooth column face, zero voids' },
    after:  { tag: 'ISSUE',     color: '#b91c1c', desc: 'Inspection photo: honeycombing pattern on C4 face' },
  },
];

// ─── tokens ───────────────────────────────────────────────────────────────────
const T = {
  bg:        'rgba(255,255,255,0.52)',
  bgStrong:  'rgba(255,255,255,0.78)',
  border:    'rgba(0,40,100,0.14)',
  borderMd:  'rgba(0,40,100,0.22)',
  borderHard:'rgba(0,40,100,0.32)',
  text:      '#0a1628',
  mid:       '#3a4a62',
  dim:       '#6a7a92',
  faint:     '#aab8cc',
  accent:    '#e85d04',
  blue:      '#1a56db',
  mono:      "'Courier Prime', monospace" as const,
  cond:      "'Barlow Condensed', sans-serif" as const,
  body:      "'Barlow', sans-serif" as const,
};

export default function Home() {
  const [cardIdx, setCardIdx] = useState(0);
  const [activeIssue, setActiveIssue] = useState(0);
  const card = cards[cardIdx];

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', fontFamily: T.body }}>

      {/* ── SIDEBAR ─────────────────────────────────────────────────────── */}
      <aside style={{
        width: 268, flexShrink: 0,
        background: 'rgba(240,244,250,0.82)',
        borderRight: `1px solid ${T.borderHard}`,
        backdropFilter: 'blur(10px)',
        display: 'flex', flexDirection: 'column',
      }}>

        {/* Sidebar header */}
        <div style={{
          padding: '14px 16px 12px',
          borderBottom: `1px solid ${T.borderMd}`,
          background: 'rgba(255,255,255,0.6)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 3 }}>
            <div style={{ width: 8, height: 8, background: T.accent, transform: 'rotate(45deg)' }} />
            <span style={{ fontFamily: T.mono, fontSize: 10, fontWeight: 700, letterSpacing: '0.16em', color: T.mid, textTransform: 'uppercase' }}>
              Issue Log
            </span>
          </div>
          <div style={{ display: 'flex', gap: 14, marginTop: 6 }}>
            {(['HIGH','MED','LOW'] as const).map(s => (
              <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <div style={{ width: 5, height: 5, background: SEV[s].color, transform: 'rotate(45deg)' }} />
                <span style={{ fontFamily: T.mono, fontSize: 8, color: SEV[s].color, letterSpacing: '0.06em' }}>
                  {issues.filter(i => i.severity === s).length} {s}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Issue list */}
        <div style={{ overflowY: 'auto', flex: 1, padding: '6px' }}>
          {issues.map((iss, i) => {
            const s = SEV[iss.severity as keyof typeof SEV];
            const active = activeIssue === i;
            return (
              <button
                key={iss.id}
                onClick={() => setActiveIssue(i)}
                style={{
                  width: '100%', textAlign: 'left',
                  background: active ? 'rgba(255,255,255,0.85)' : 'transparent',
                  border: `1px solid ${active ? T.borderHard : 'transparent'}`,
                  borderLeft: `3px solid ${active ? s.color : 'transparent'}`,
                  padding: '9px 11px', marginBottom: 2,
                  cursor: 'pointer', transition: 'all 0.1s',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontFamily: T.mono, fontSize: 8, color: T.faint, letterSpacing: '0.1em' }}>{iss.id}</span>
                  <span style={{
                    fontFamily: T.mono, fontSize: 7, fontWeight: 700,
                    padding: '1px 5px',
                    background: s.bg, color: s.color,
                    border: `1px solid ${s.border}`,
                    letterSpacing: '0.08em',
                  }}>{iss.severity}</span>
                </div>
                <p style={{ fontFamily: T.cond, fontSize: 12, fontWeight: 600, color: active ? T.text : T.mid, margin: '0 0 3px', letterSpacing: '0.03em' }}>
                  {iss.floor} · {iss.room}
                </p>
                <p style={{ fontFamily: T.body, fontSize: 10, color: T.dim, margin: 0, lineHeight: 1.5 }}>
                  {iss.desc}
                </p>
              </button>
            );
          })}
        </div>
      </aside>

      {/* ── MAIN ────────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Header */}
        <header style={{
          padding: '0 24px',
          height: 52,
          borderBottom: `1px solid ${T.borderHard}`,
          background: 'rgba(240,244,250,0.9)',
          backdropFilter: 'blur(12px)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          flexShrink: 0,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
            {/* Logo */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ width: 22, height: 22, border: `2px solid ${T.accent}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ width: 10, height: 10, background: T.accent }} />
              </div>
              <span style={{ fontFamily: T.mono, fontSize: 14, fontWeight: 700, letterSpacing: '0.12em', color: T.text }}>SITEVAULT</span>
            </div>
            <div style={{ width: 1, height: 20, background: T.borderMd }} />
            <span style={{ fontFamily: T.mono, fontSize: 9, color: T.dim, letterSpacing: '0.14em' }}>
              SITE-047 · DOWNTOWN TOWER PROJECT · FEB 21 2026
            </span>
          </div>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {/* Live indicator */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '4px 12px',
              border: `1px solid ${T.borderMd}`,
              background: 'rgba(255,255,255,0.5)',
            }}>
              <span style={{ width: 5, height: 5, background: '#0a7c4e', display: 'inline-block', animation: 'pulse 2s infinite' }} />
              <span style={{ fontFamily: T.mono, fontSize: 8, color: '#0a7c4e', letterSpacing: '0.14em' }}>LIVE MONITORING</span>
            </div>

            <button className="cad-btn" style={{ fontFamily: T.mono }}>
              ↓ EXPORT
            </button>

            <button className="cad-btn cad-btn-primary" style={{ fontFamily: T.mono }}>
              <Upload size={11} /> UPLOAD VIDEO
            </button>
          </div>
        </header>

        {/* Body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px 22px', display: 'flex', flexDirection: 'column', gap: 18 }}>

          {/* ── SECTION 1: 3D Models ── */}
          <section>
            <SectionHeader
              tag="SEC.01"
              title="3D SITE MODELS"
              sub="BIM Reference vs. As-Built Point Cloud — synchronized rotation"
              right={<Badge text="DESIGN vs AS-BUILT" color={T.blue} />}
            />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
              <ModelPlaceholder label="DESIGN MODEL" tag="BIM REFERENCE" accentColor={T.blue} />
              <ModelPlaceholder label="AS-BUILT SCAN" tag="POINT CLOUD"  accentColor={T.accent} />
            </div>
          </section>

          {/* ── SECTION 2: Photo Comparison ── */}
          <section style={{ paddingBottom: 16 }}>
            <SectionHeader
              tag="SEC.02"
              title="ISSUE EVIDENCE"
              sub={`${cardIdx + 1} / ${cards.length} — ${card.label}`}
              right={
                <div style={{ display: 'flex', gap: 6 }}>
                  <NavBtn onClick={() => setCardIdx(i => Math.max(0, i - 1))} disabled={cardIdx === 0} dir="left" />
                  <NavBtn onClick={() => setCardIdx(i => Math.min(cards.length - 1, i + 1))} disabled={cardIdx === cards.length - 1} dir="right" primary />
                </div>
              }
            />

            {/* Card */}
            <div style={{ border: `1px solid ${T.borderHard}`, background: T.bg, backdropFilter: 'blur(8px)' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
                <PhotoSlot tag={card.before.tag} color={card.before.color} desc={card.before.desc} side="left" />
                <PhotoSlot tag={card.after.tag}  color={card.after.color}  desc={card.after.desc}  side="right" />
              </div>
              {/* Footer */}
              <div style={{
                padding: '8px 16px',
                borderTop: `1px solid ${T.borderMd}`,
                background: 'rgba(240,244,250,0.6)',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
                <span style={{ fontFamily: T.mono, fontSize: 8, color: T.dim, letterSpacing: '0.1em' }}>{card.label}</span>
                <div style={{ display: 'flex', gap: 4 }}>
                  {cards.map((_, i) => (
                    <button key={i} className={`dot-nav ${i === cardIdx ? 'active' : ''}`} onClick={() => setCardIdx(i)} />
                  ))}
                </div>
              </div>
            </div>
          </section>

        </div>
      </div>
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function SectionHeader({ tag, title, sub, right }: { tag: string; title: string; sub: string; right?: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 8 }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'baseline' }}>
        <span style={{ fontFamily: T.mono, fontSize: 8, color: T.accent, letterSpacing: '0.14em' }}>{tag}</span>
        <span style={{ fontFamily: T.cond, fontSize: 13, fontWeight: 700, letterSpacing: '0.14em', color: T.text }}>{title}</span>
        <span style={{ fontFamily: T.body, fontSize: 10, color: T.dim }}>{sub}</span>
      </div>
      {right}
    </div>
  );
}

function Badge({ text, color }: { text: string; color: string }) {
  return (
    <span style={{
      fontFamily: T.mono, fontSize: 8, letterSpacing: '0.1em',
      padding: '2px 8px',
      border: `1px solid ${color}55`,
      background: `${color}10`,
      color,
    }}>{text}</span>
  );
}

function NavBtn({ onClick, disabled, dir, primary }: { onClick: () => void; disabled: boolean; dir: 'left' | 'right'; primary?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        width: 30, height: 28,
        border: `1px solid ${disabled ? T.border : primary ? T.accent : T.borderHard}`,
        background: disabled ? 'transparent' : primary ? T.accent : 'rgba(255,255,255,0.6)',
        color: disabled ? T.faint : primary ? '#fff' : T.mid,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'all 0.1s',
      }}
    >
      {dir === 'left' ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
    </button>
  );
}

function ModelPlaceholder({ label, tag, accentColor }: { label: string; tag: string; accentColor: string }) {
  return (
    <div style={{
      position: 'relative',
      height: 270,
      border: `1px solid ${T.borderHard}`,
      background: 'rgba(255,255,255,0.38)',
      backdropFilter: 'blur(6px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      overflow: 'hidden',
    }}>
      {/* Perspective grid overlay */}
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: `
          linear-gradient(${accentColor}18 1px, transparent 1px),
          linear-gradient(90deg, ${accentColor}18 1px, transparent 1px)
        `,
        backgroundSize: '24px 24px',
      }} />

      {/* Scanline */}
      <div className="scanline" style={{ background: `linear-gradient(90deg, transparent, ${accentColor}, transparent)` }} />

      {/* Corner ticks */}
      {[
        { top: 0,    left: 0,    borderTop: `2px solid ${accentColor}`, borderLeft:  `2px solid ${accentColor}` },
        { top: 0,    right: 0,   borderTop: `2px solid ${accentColor}`, borderRight: `2px solid ${accentColor}` },
        { bottom: 0, left: 0,    borderBottom: `2px solid ${accentColor}`, borderLeft:  `2px solid ${accentColor}` },
        { bottom: 0, right: 0,   borderBottom: `2px solid ${accentColor}`, borderRight: `2px solid ${accentColor}` },
      ].map((style, i) => (
        <div key={i} style={{ position: 'absolute', width: 16, height: 16, ...style }} />
      ))}

      {/* Crosshair */}
      <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
        <div style={{ width: 1, height: '100%', background: `${accentColor}20`, position: 'absolute' }} />
        <div style={{ height: 1, width: '100%', background: `${accentColor}20`, position: 'absolute' }} />
      </div>

      {/* Center content */}
      <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
        <div style={{ position: 'relative' }}>
          <Box size={60} color={accentColor} strokeWidth={0.6} style={{ opacity: 0.3 }} />
          <div style={{ position: 'absolute', inset: 0, background: `radial-gradient(circle, ${accentColor}22, transparent 70%)` }} />
        </div>
        <div style={{ textAlign: 'center' }}>
          <p style={{ fontFamily: T.mono, fontSize: 10, color: accentColor, letterSpacing: '0.14em', marginBottom: 3 }}>{label}</p>
          <p style={{ fontFamily: T.mono, fontSize: 8, color: T.faint, letterSpacing: '0.1em' }}>// 3D VIEWER INTEGRATION POINT</p>
        </div>
      </div>

      {/* Tag top-center */}
      <div style={{ position: 'absolute', top: 10, left: '50%', transform: 'translateX(-50%)' }}>
        <Badge text={tag} color={accentColor} />
      </div>

      {/* Controls bottom-right */}
      <div style={{ position: 'absolute', bottom: 10, right: 10, display: 'flex', gap: 4 }}>
        {[RotateCcw, Maximize2].map((Icon, i) => (
          <button key={i} style={{
            width: 24, height: 24,
            border: `1px solid ${T.borderMd}`,
            background: 'rgba(255,255,255,0.5)',
            color: T.dim,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer',
          }}>
            <Icon size={11} />
          </button>
        ))}
      </div>
    </div>
  );
}

function PhotoSlot({ tag, color, desc, side }: { tag: string; color: string; desc: string; side: 'left' | 'right' }) {
  return (
    <div style={{
      borderRight: side === 'left' ? `1px solid ${T.borderMd}` : 'none',
      background: 'rgba(255,255,255,0.3)',
      minHeight: 220,
      display: 'flex', flexDirection: 'column',
    }}>
      {/* Tag bar */}
      <div style={{
        padding: '7px 14px',
        borderBottom: `1px solid ${T.borderMd}`,
        background: `${color}0c`,
        display: 'flex', alignItems: 'center', gap: 7,
      }}>
        <div style={{ width: 5, height: 5, background: color, transform: 'rotate(45deg)' }} />
        <span style={{ fontFamily: T.mono, fontSize: 8, fontWeight: 700, color, letterSpacing: '0.14em' }}>{tag}</span>
      </div>

      {/* Image area */}
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexDirection: 'column', gap: 10, padding: 20,
        backgroundImage: `repeating-linear-gradient(45deg, ${color}06 0px, ${color}06 1px, transparent 1px, transparent 10px)`,
      }}>
        {/* Placeholder frame */}
        <div style={{
          width: 52, height: 44,
          border: `1px dashed ${color}55`,
          background: `${color}06`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          position: 'relative',
        }}>
          {/* Corner marks */}
          {[{top:0,left:0},{top:0,right:0},{bottom:0,left:0},{bottom:0,right:0}].map((s, i) => (
            <div key={i} style={{
              position: 'absolute', width: 6, height: 6,
              borderTop: (s as any).top === 0 ? `1.5px solid ${color}` : 'none',
              borderBottom: (s as any).bottom === 0 ? `1.5px solid ${color}` : 'none',
              borderLeft: (s as any).left === 0 ? `1.5px solid ${color}` : 'none',
              borderRight: (s as any).right === 0 ? `1.5px solid ${color}` : 'none',
              ...s,
            }} />
          ))}
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" opacity={0.45}>
            <rect x="3" y="3" width="18" height="18" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
        </div>
        <p style={{ fontFamily: T.body, fontSize: 10, color: T.dim, textAlign: 'center', lineHeight: 1.6, maxWidth: 180 }}>
          {desc}
        </p>
      </div>
    </div>
  );
}
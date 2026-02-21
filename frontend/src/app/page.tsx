'use client';

import { useState } from 'react';
import {
  LayoutGrid, Play, LayoutList, Diamond, Clock,
  Settings, Film, Search
} from 'lucide-react';
import StatCard from '../components/StatCard';

const inventoryData = [
  { name: 'Portland Cement Type I', category: 'MATERIAL', stock: 412, max: 500, unit: 'bags',    status: 'GOOD', updated: 'Today, 09:14' },
  { name: 'Rebar #5 (5/8")',        category: 'MATERIAL', stock: 38,  max: 200, unit: 'bundles', status: 'LOW',  updated: 'Today, 11:02' },
  { name: 'Safety Harnesses',       category: 'SAFETY',   stock: 54,  max: 100, unit: 'units',   status: 'MED',  updated: 'Today, 07:30' },
  { name: 'Concrete Mixer HM-350',  category: 'EQUIPMENT',stock: 6,   max: 8,   unit: 'machines',status: 'GOOD', updated: 'Feb 20, 16:45'},
  { name: 'Plywood Sheets 4x8',     category: 'MATERIAL', stock: 12,  max: 150, unit: 'sheets',  status: 'LOW',  updated: 'Today, 10:55' },
];

const statusColors: Record<string, { bg: string; text: string; bar: string }> = {
  GOOD: { bg: '#1a3a2a', text: '#3ddc84', bar: '#3ddc84' },
  LOW:  { bg: '#3a1a1a', text: '#ff5c5c', bar: '#ff5c5c' },
  MED:  { bg: '#3a2a10', text: '#f5a623', bar: '#f5a623' },
};

const categoryColors: Record<string, string> = {
  MATERIAL: '#5ba4f5',
  SAFETY: '#c084fc',
  EQUIPMENT: '#2dd4bf',
};

type FilterType = 'All' | 'Low' | 'Tools' | 'Material';

export default function Home() {
  const [filter, setFilter] = useState<FilterType>('All');
  const [search, setSearch] = useState('');

  const filtered = inventoryData.filter(item => {
    const matchSearch = item.name.toLowerCase().includes(search.toLowerCase());
    const matchFilter =
      filter === 'All' ? true :
      filter === 'Low' ? item.status === 'LOW' :
      filter === 'Tools' ? item.category === 'EQUIPMENT' :
      filter === 'Material' ? item.category === 'MATERIAL' : true;
    return matchSearch && matchFilter;
  });

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#0d0d0d' }}>

      {/* Sidebar */}
      <aside style={{
        width: 56,
        background: '#111',
        borderRight: '1px solid rgba(255,255,255,0.06)',
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        padding: '16px 0', gap: 6, flexShrink: 0,
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
          {/* Logo mark */}
          <div style={{
            width: 32, height: 32, background: '#f5c542', borderRadius: 8,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            marginBottom: 12
          }}>
            <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 10, fontWeight: 700, color: '#000' }}>SV</span>
          </div>
          <button className="sidebar-btn active"><LayoutGrid size={16} /></button>
          <button className="sidebar-btn"><Play size={16} /></button>
          <button className="sidebar-btn"><LayoutList size={16} /></button>
          <button className="sidebar-btn"><Diamond size={16} /></button>
          <button className="sidebar-btn"><Clock size={16} /></button>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
          <button className="sidebar-btn"><Settings size={16} /></button>
          <div style={{
            width: 30, height: 30, borderRadius: '50%',
            background: 'linear-gradient(135deg, #5ba4f5, #3ddc84)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 10, fontWeight: 700, color: '#000'
          }}>JD</div>
        </div>
      </aside>

      {/* Main content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Top bar */}
        <header style={{
          padding: '16px 28px',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between'
        }}>
          <div>
            <h1 className="mono" style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.01em', margin: 0 }}>IronCheck</h1>
            <p style={{ fontSize: 10, color: '#555', fontFamily: 'Space Mono, monospace', letterSpacing: '0.15em', textTransform: 'uppercase', marginTop: 2 }}>
              PROGRESS
            </p>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            
            <button style={{
              background: 'transparent', border: '1px solid rgba(255,255,255,0.12)',
              color: '#eee', padding: '7px 16px', borderRadius: 8, fontSize: 11,
              fontWeight: 600, cursor: 'pointer'
            }}>↓ Export</button>
            <button style={{
              background: '#f5c542', color: '#000',
              padding: '7px 18px', borderRadius: 8, fontSize: 11,
              fontWeight: 700, cursor: 'pointer', border: 'none'
            }}>+ Upload Video</button>
          </div>
        </header>

        {/* Body */}
        <div style={{ flex: 1, padding: '24px 28px', overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 20 }}>

          {/* Stat cards row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14 }}>
            <StatCard label="Total Inventory" value="1,284" sub="ITEMS TRACKED" badge="+4.2%" badgeColor="#3ddc84" valueColor="#f5c542" />
            <StatCard label="Low Stock Alerts" value="17" sub="ITEMS CRITICAL" badge="+3 TODAY" badgeColor="#ff5c5c" valueColor="#ff5c5c" />
            <StatCard label="Videos Uploaded" value="342" sub="THIS MONTH" badge="+12%" badgeColor="#3ddc84" valueColor="#3ddc84" />
            <StatCard label="Storage Used" value="2.4TB" sub="OF 5TB QUOTA" badge="48%" badgeColor="#5ba4f5" valueColor="#5ba4f5" />
          </div>

          {/* Lower two-panel row */}
          <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 14, flex: 1 }}>

            {/* Video Uploader */}
            <div style={{ background: '#161616', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 14, padding: 20, display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="mono" style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#ccc' }}>Video Uploader</span>
                <span style={{ fontSize: 9, color: '#555', fontFamily: 'Space Mono, monospace', border: '1px solid #2a2a2a', padding: '2px 8px', borderRadius: 5 }}>DRAG &amp; DROP</span>
              </div>

              {/* Drop zone */}
              <div className="upload-area">
                <div style={{ background: '#2a2a2a', borderRadius: 10, padding: '10px 12px' }}>
                  <Film size={28} color="#666" />
                </div>
                <p style={{ fontSize: 12, fontWeight: 600, color: '#ccc', margin: 0 }}>Drop site footage here</p>
                <p style={{ fontSize: 10, color: '#555', margin: 0 }}>or click to browse files</p>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'center', marginTop: 4 }}>
                  {['MP4','MOV','AVI','MKV','WEBM'].map(f => (
                    <span key={f} style={{ fontSize: 9, color: '#555', background: '#1e1e1e', border: '1px solid #2a2a2a', padding: '2px 7px', borderRadius: 4, fontFamily: 'Space Mono, monospace' }}>{f}</span>
                  ))}
                </div>
              </div>

              {/* File rows */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <FileRow name="zone-a-inspection-02..." size="1.2 GB" progress={100} complete />
                <FileRow name="crane-ops-north-feb..." size="3.8 GB" progress={45} />
              </div>
            </div>

            {/* Inventory Tracker */}
            <div style={{ background: '#161616', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 14, padding: 20, display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
                <span className="mono" style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#ccc' }}>Inventory Tracker</span>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  {/* Search */}
                  <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                    <Search size={12} color="#555" style={{ position: 'absolute', left: 10 }} />
                    <input
                      value={search}
                      onChange={e => setSearch(e.target.value)}
                      placeholder="Search item..."
                      style={{
                        background: '#111', border: '1px solid #2a2a2a', borderRadius: 8,
                        padding: '6px 10px 6px 28px', color: '#ccc', fontSize: 11,
                        outline: 'none', width: 160
                      }}
                    />
                  </div>
                  {(['All','Low','Tools','Material'] as FilterType[]).map(f => (
                    <button
                      key={f}
                      onClick={() => setFilter(f)}
                      style={{
                        fontSize: 10, fontWeight: 700, padding: '5px 12px', borderRadius: 7,
                        border: filter === f ? '1px solid #f5c542' : '1px solid #2a2a2a',
                        background: filter === f ? 'rgba(245,197,66,0.1)' : 'transparent',
                        color: filter === f ? '#f5c542' : '#666',
                        cursor: 'pointer', fontFamily: 'Space Mono, monospace'
                      }}
                    >{f}</button>
                  ))}
                </div>
              </div>

              {/* Table */}
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      {['ITEM','CATEGORY','STOCK','UNIT','STATUS','LAST UPDATED'].map(h => (
                        <th key={h} style={{ fontSize: 9, fontFamily: 'Space Mono, monospace', letterSpacing: '0.1em', color: '#444', padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((item, i) => {
                      const sc = statusColors[item.status];
                      const pct = Math.round((item.stock / item.max) * 100);
                      return (
                        <tr key={i} className="inv-row" style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', transition: 'background 0.15s' }}>
                          <td style={{ padding: '12px', fontSize: 13, fontWeight: 600, color: '#eee' }}>{item.name}</td>
                          <td style={{ padding: '12px' }}>
                            <span style={{ fontSize: 9, fontFamily: 'Space Mono, monospace', color: categoryColors[item.category] || '#888', letterSpacing: '0.05em' }}>{item.category}</span>
                          </td>
                          <td style={{ padding: '12px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <span style={{ fontSize: 12, fontWeight: 600, color: '#ccc', minWidth: 32 }}>{item.stock}</span>
                              <div className="progress-bar" style={{ width: 80 }}>
                                <div className="progress-bar-fill" style={{ width: `${pct}%`, background: sc.bar }} />
                              </div>
                            </div>
                          </td>
                          <td style={{ padding: '12px', fontSize: 11, color: '#666' }}>{item.unit}</td>
                          <td style={{ padding: '12px' }}>
                            <span className="status-badge" style={{ background: sc.bg, color: sc.text }}>{item.status}</span>
                          </td>
                          <td style={{ padding: '12px', fontSize: 10, color: '#555', fontFamily: 'Space Mono, monospace' }}>{item.updated}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}

function FileRow({ name, size, progress, complete }: { name: string; size: string; progress: number; complete?: boolean }) {
  return (
    <div className="file-row">
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontSize: 11, color: '#bbb', fontFamily: 'Space Mono, monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '65%' }}>{name}</span>
        <span style={{ fontSize: 11, color: '#555', fontFamily: 'Space Mono, monospace' }}>{size}</span>
      </div>
      <div className="progress-bar">
        <div className="progress-bar-fill" style={{ width: `${progress}%`, background: complete ? '#3ddc84' : '#f5c542' }} />
      </div>
      {complete && <p style={{ fontSize: 9, color: '#3ddc84', fontFamily: 'Space Mono, monospace', marginTop: 4, margin: 0 }}>✓ {size}</p>}
    </div>
  );
}

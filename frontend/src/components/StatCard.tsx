'use client';

interface StatCardProps {
  label: string;
  value: string;
  sub: string;
  badge?: string;
  badgeColor?: string;
  valueColor?: string;
}

export default function StatCard({ label, value, sub, badge, badgeColor = '#3ddc84', valueColor = '#fff' }: StatCardProps) {
  return (
    <div className="stat-card">
      {badge && (
        <span
          className="badge"
          style={{
            background: `${badgeColor}22`,
            color: badgeColor,
          }}
        >
          {badge}
        </span>
      )}
      <p className="label">{label}</p>
      <p className="value mono" style={{ color: valueColor }}>{value}</p>
      <p className="sub">{sub}</p>
    </div>
  );
}

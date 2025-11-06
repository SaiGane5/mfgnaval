import React from 'react'

// Simple horizontal gauge that maps any numeric value to a 0..1 fill using
// normalized = abs(v) / (1 + abs(v)) so it behaves for large/small values.
export default function Gauge({ value = 0, height = 12 }){
  const v = Number(value) || 0
  const norm = Math.abs(v) / (1 + Math.abs(v))
  const percent = Math.round(norm * 100)

  let bg = 'bg-green-500'
  if (norm > 0.66) bg = 'bg-red-500'
  else if (norm > 0.33) bg = 'bg-yellow-500'

  return (
    <div className="w-full">
      <div className="w-full bg-gray-200 rounded" style={{ height }}>
        <div className={`${bg} h-full rounded`} style={{ width: `${percent}%` }} />
      </div>
      <div className="text-xs text-gray-600 mt-1">{percent}%</div>
    </div>
  )
}

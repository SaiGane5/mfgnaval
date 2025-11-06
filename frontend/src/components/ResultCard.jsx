import React from 'react'

// Interpret the numeric decay coefficient into a simple status and recommendation.
// Assumption: larger absolute coefficient => worse decay (more degraded). We normalize
// with a soft mapping so values of any scale produce a 0..1 score.
function interpretValue(v){
  // Use domain-aware thresholds for decay coefficients which are near 1.0.
  // We report degradation as (1 - value) and use different sensible cutoffs so
  // small deviations from 1.0 show up as Monitor/Action when appropriate.
  const n = Number(v)
  if (!Number.isFinite(n)) return (_)=>({ status: 'Unknown', color: 'text-gray-600', recommendation: 'No prediction available.' })

  // default thresholds
  let goodCut = 0.99
  let monitorCut = 0.96

  return (valKey)=>{
    // valKey is optional hint about which output this is (compressor/turbine)
    if (valKey && /turbine/i.test(valKey)){
      // turbine tends to have tighter range in dataset
      goodCut = 0.995
      monitorCut = 0.975
    }
    if (n >= goodCut) return { status: 'Good', color: 'text-green-600', recommendation: 'No immediate action required.' }
    if (n >= monitorCut) return { status: 'Monitor', color: 'text-yellow-600', recommendation: 'Monitor regularly; consider maintenance if trend increases.' }
    return { status: 'Action needed', color: 'text-red-600', recommendation: 'Schedule inspection and diagnostic checks.' }
  }
}

export default function ResultCard({ title, value, hint }){
  const numeric = Number(value)
  const display = Number.isFinite(numeric) ? numeric.toFixed(4) : '—'
  // Decide which thresholds to use by inspecting the title for 'turbine' keyword
  const interpFn = interpretValue(numeric)
  const interp = interpFn && interpFn(title) || { status: 'Unknown', color: 'text-gray-600', recommendation: '' }
  const degradationPct = Number.isFinite(numeric) ? (1 - numeric) * 100 : null

  return (
    <div className="card">
      <div className="flex items-start gap-4">
        <div style={{ flex: 1 }}>
          <div className="text-sm text-gray-500">{title}</div>
          <div className="text-2xl font-semibold mt-1">{display}</div>
          {hint && <div className="text-xs text-gray-500 mt-1">{hint}</div>}
          <div className="mt-3">
            <div className="flex items-center gap-2">
              <div className={`font-medium ${interp.color}`}>{interp.status}</div>
              {degradationPct !== null && (
                <div className="text-xs text-gray-600">Degradation: {degradationPct.toFixed(2)}%</div>
              )}
            </div>
            <div className="text-xs text-gray-600 mt-1">{interp.recommendation}</div>
          </div>
        </div>
      </div>
    </div>
  )
}

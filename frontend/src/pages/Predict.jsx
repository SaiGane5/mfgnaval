import React, { useState } from 'react'
import api from '../services/api'
import InputField from '../components/InputField'
import ResultCard from '../components/ResultCard'
import { useEffect } from 'react'

const initial = {
  lp: 0, v: 0, GTT: 0, gtn:0, ggn:0, ts:0, tp:0, t48:0, t1:0, t2:0, p48:0, p1:0, p2:0, pexh:0, tic:0, mf:0
}

// start with a fallback mapping; we'll attempt to fetch metadata from the backend
const fallbackPretty = {
  lp: 'Lever position (lp)',
  v: 'Ship speed (v) [knots]',
  GTT: 'Gas Turbine shaft torque (GTT) [kN m]',
  gtn: 'Gas Turbine rate of revolutions (gtn) [rpm]',
  ggn: 'Gas Generator rate of revolutions (ggn) [rpm]',
  ts: 'Starboard Propeller Torque (Ts) [kN]',
  tp: 'Port Propeller Torque (Tp) [kN]',
  t48: 'HP Turbine exit temperature (T48) [°C]',
  t1: 'GT Compressor inlet air temperature (T1) [°C]',
  t2: 'GT Compressor outlet air temperature (T2) [°C]',
  p48: 'HP Turbine exit pressure (P48) [bar]',
  p1: 'GT Compressor inlet air pressure (P1) [bar]',
  p2: 'GT Compressor outlet air pressure (P2) [bar]',
  pexh: 'Gas Turbine exhaust gas pressure (Pexh) [bar]',
  tic: 'Turbine Injection Control (TIC) [%]',
  mf: 'Fuel flow (mf) [kg/s]'
}

export default function Predict(){
  const [form, setForm] = useState(initial)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [prettyLabels, setPrettyLabels] = useState(fallbackPretty)
  const [outputHints, setOutputHints] = useState({})

  function handleChange(e){
    const { name, value } = e.target
    setForm(f=>({...f, [name]: value === '' ? '' : Number(value)}))
  }

  async function handleSubmit(e){
    e.preventDefault()
    setLoading(true); setError(null); setResult(null)
    try{
      const res = await api.post('/api/predict', form)
      setResult(res.data)
    }catch(err){
      setError(err?.response?.data?.detail || err.message)
    }finally{ setLoading(false) }
  }

  useEffect(()=>{
    // Try to fetch metadata from backend to populate labels and hints
    let mounted = true
    api.get('/api/metadata').then(res=>{
      if (!mounted) return
      const map = {}
      const hints = {}
      (res.data.features || []).forEach(f=>{ map[f.key] = `${f.label}${f.unit? ` [${f.unit}]`: ''}` })
      (res.data.outputs || []).forEach(o=>{ hints[o.key] = o.description })
      setPrettyLabels(p=>({ ...p, ...map }))
      setOutputHints(hints)
    }).catch(()=>{
      // ignore, keep fallbackPretty
    })
    return ()=>{ mounted = false }
  }, [])

  // A couple of small example presets to help users get started
  function applyPreset(name){
    // Real-world inspired presets based on the Naval GT dataset ranges.
    // 1) Cruise (Nominal) - normal operation
    if (name === 'cruise'){
      setForm({ ...initial,
        lp: 5, v: 15, GTT: 2.5, gtn: 3200, ggn: 1500, ts:1.2, tp:1.1,
        t48:550, t1:20, t2:120, p48:1.2, p1:1.0, p2:1.5, pexh:1.1, tic:70, mf:5.5
      })
    }
    // 2) High speed / heavy load - maximum demand scenario
    else if (name === 'high'){
      setForm({ ...initial,
        lp: 9, v: 27, GTT: 8.0, gtn: 4500, ggn: 2200, ts:3.0, tp:3.0,
        t48:700, t1:30, t2:180, p48:2.0, p1:1.2, p2:2.2, pexh:1.8, tic:95, mf:20.0
      })
    }
    // 3) Aged compressor / degraded condition - same operational profile but signs of decay
    else if (name === 'degraded'){
      setForm({ ...initial,
        lp: 6, v: 18, GTT: 4.0, gtn: 3500, ggn: 1800, ts:1.8, tp:1.7,
        t48:620, t1:35, t2:140, p48:1.4, p1:0.95, p2:1.3, pexh:1.25, tic:80, mf:8.0
      })
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold">Predict</h2>
        <div className="space-x-2">
          <button onClick={()=>applyPreset('cruise')} className="px-3 py-1 bg-gray-100 rounded text-sm">cruise</button>
          <button onClick={()=>applyPreset('high')} className="px-3 py-1 bg-gray-100 rounded text-sm">high</button>
          <button onClick={()=>applyPreset('degraded')} className="px-3 py-1 bg-gray-100 rounded text-sm">degraded</button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
          {Object.keys(initial).map(k=> (
            <div key={k} className="card">
              <InputField name={k} label={prettyLabels[k] || k} value={form[k]} onChange={handleChange} />
            </div>
          ))}
          <div className="md:col-span-2 flex items-center gap-4">
            <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded" disabled={loading}>{loading? 'Running...':'Run Predict'}</button>
            {error && <div className="text-red-600">{error}</div>}
          </div>
        </div>

        <div className="space-y-4">
          <div className="card">
            <h3 className="font-medium text-gray-700">Quick help</h3>
            <p className="text-sm text-gray-600 mt-2">Fill the numeric inputs on the left and press <strong>Run Predict</strong>. Results show numeric predictions, percent degradation, and an action recommendation.</p>
          </div>

          {result ? (
            <div className="space-y-3">
              <ResultCard title="GT Compressor decay state coefficient" value={result.gt_c_decay} hint={outputHints.gt_c_decay || 'GT Compressor decay state coefficient (unitless)'} />
              <ResultCard title="GT Turbine decay state coefficient" value={result.gt_t_decay} hint={outputHints.gt_t_decay || 'GT Turbine decay state coefficient (unitless)'} />
            </div>
          ) : (
            <div className="card text-sm text-gray-600">Results will appear here after you run a prediction.</div>
          )}
        </div>
      </form>
    </div>
  )
}

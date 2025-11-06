import React, { useEffect, useState } from 'react'
import api from '../services/api'

export default function Dashboard(){
  const [status, setStatus] = useState('loading')
  useEffect(()=>{
    api.get('/api/health')
      .then(()=>setStatus('ok'))
      .catch(()=>setStatus('error'))
  },[])

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-4">Dashboard</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card">
          <h3 className="font-medium">API Status</h3>
          <p className="text-sm text-gray-600">{status}</p>
        </div>
        <div className="card">
          <h3 className="font-medium">Quick Actions</h3>
          <p className="text-sm text-gray-600">Use the Predict page to run example predictions.</p>
        </div>
      </div>
    </div>
  )
}

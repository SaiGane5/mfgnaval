import React from 'react'
import { Routes, Route, Link } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'
import './App.css'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-xl font-semibold">MFG Naval Model</h1>
          <div className="space-x-4">
            <Link to="/" className="text-sm text-gray-600">Dashboard</Link>
            <Link to="/predict" className="text-sm text-gray-600">Predict</Link>
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto p-4">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
        </Routes>
      </main>
    </div>
  )
}

export default App

import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
// Import the JSX file explicitly to avoid accidentally resolving the legacy
// placeholder `App.js` (which exports null). Vite's resolver can prefer .js
// over .jsx when the extension is omitted, causing a blank page.
import App from './App.jsx'
import './index.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
)

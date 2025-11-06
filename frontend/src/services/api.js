import axios from 'axios'

// Support both Vite (`VITE_API_URL`) and Create-React-App (`REACT_APP_API_URL`) env var names.
// Accessing `import.meta.env` at runtime in a non-Vite environment throws, so
// we guard with a try/catch. Prefer Vite value if available (build-time),
// otherwise use CRA's `process.env.REACT_APP_API_URL`, otherwise fallback.
let viteApiUrl
try {
  // In Vite-built bundles `import.meta.env` is available. Attempt access
  // inside try/catch so code doesn't blow up under react-scripts.
  viteApiUrl = import.meta?.env?.VITE_API_URL
} catch (err) {
  viteApiUrl = undefined
}

const baseURL = viteApiUrl || process.env.REACT_APP_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL,
  headers: { 'Content-Type': 'application/json' }
})

export default api

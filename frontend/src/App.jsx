import { useState } from 'react'
import Robot from './Robot'

export default function App() {
  const [tab, setTab] = useState('video')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  function handleFile(e) {
    const f = e.target.files[0]
    setFile(f)
    setResult(null)
    if (!f) return setPreview(null)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  async function submit() {
    if (!file) return
    setLoading(true)
    const form = new FormData()
    form.append('file', file)

    const endpoint = tab === 'video' ? '/predict' : '/predict_image'

    try {
      const res = await fetch(`http://localhost:8000${endpoint}`, { method: 'POST', body: form })
      const json = await res.json()
      setResult(json)
    } catch (e) {
      setResult({ error: "Make sure the backend is running! " + String(e) })
    } finally {
      setLoading(false)
    }
  }

  function ProbBar({ fake }) {
    const pctFake = Math.round(fake * 100)
    return (
      <div style={{ width: '100%', background: 'rgba(255, 255, 255, 0.05)', height: 18, borderRadius: 9, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
        <div style={{ width: `${pctFake}%`, height: '100%', background: 'linear-gradient(to right, #a855f7, #ec4899)', boxShadow: '0 0 10px rgba(168, 85, 247, 0.5)' }} />
      </div>
    )
  }

  return (
    <div className="card" style={{ maxWidth: 1000, width: '90%', margin: '40px auto' }}>
      <h1>TruthLens AI</h1>
      <p style={{ color: '#94a3b8', marginBottom: 30, fontSize: '1.1em' }}>Neural Integrity Verification System</p>

      <div style={{ display: 'flex', gap: 12, marginBottom: 30 }}>
        <button
          onClick={() => setTab('video')}
          style={{
            background: tab === 'video' ? '#a855f7' : 'rgba(255,255,255,0.05)',
            border: tab === 'video' ? 'none' : '1px solid rgba(255,255,255,0.1)',
            padding: '12px 24px',
            color: 'white'
          }}>
          📸 Video Analysis
        </button>
        <button
          onClick={() => setTab('image')}
          style={{
            background: tab === 'image' ? '#a855f7' : 'rgba(255,255,255,0.05)',
            border: tab === 'image' ? 'none' : '1px solid rgba(255,255,255,0.1)',
            padding: '12px 24px',
            color: 'white'
          }}>
          🖼️ Image Analysis
        </button>
      </div>

      <div style={{ display: 'flex', gap: 40, flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 300 }}>
          <div style={{
            background: 'rgba(0,0,0,0.2)',
            border: '2px dashed rgba(168, 85, 247, 0.3)',
            padding: 40,
            borderRadius: 20,
            textAlign: 'center',
            cursor: 'pointer',
            position: 'relative',
            overflow: 'hidden'
          }} onClick={() => document.getElementById('fileInput').click()}>
            <input id="fileInput" type="file" accept={tab === 'video' ? "video/*" : "image/*"} onChange={handleFile} style={{ display: 'none' }} />
            <div style={{ fontSize: 40, marginBottom: 10 }}>☁️</div>
            <div style={{ fontWeight: 600, color: '#e2e8f0' }}>{file ? file.name : `Select ${tab} to analyze`}</div>
            <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 8 }}>Drag and drop or click to browse</div>
          </div>

          <div style={{ marginTop: 24 }}>
            <button
              onClick={submit}
              disabled={!file || loading}
              style={{
                width: '100%',
                fontSize: '1.1em',
                padding: '16px',
                background: (file && !loading) ? 'linear-gradient(to right, #a855f7, #6366f1)' : '#333'
              }}>
              {loading ? '🔬 Processing Neural Framework...' : '🚀 Authenticate Media'}
            </button>
          </div>

          {result && !result.error && (
            <div style={{
              marginTop: 30,
              padding: 24,
              backgroundColor: 'rgba(255,255,255,0.03)',
              borderRadius: 20,
              border: `1px solid ${result.prediction === 'FAKE' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(34, 197, 94, 0.3)'}`
            }}>
              <h2 style={{ margin: 0, fontSize: '1.2em', color: '#94a3b8' }}>VERIFICATION STATUS</h2>
              <div style={{
                fontSize: '2.5em',
                fontWeight: 900,
                color: result.prediction === 'FAKE' ? '#ef4444' : '#22c55e',
                letterSpacing: 2,
                marginTop: 10
              }}>
                {result.prediction}
              </div>
              <div style={{ fontSize: 14, color: '#94a3b8', marginTop: 4 }}>Confidence level: {(result.confidence * 100).toFixed(1)}%</div>

              <div style={{ marginTop: 20 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 8, color: '#e2e8f0' }}>
                  <span>Manipulation Probability</span>
                  <span>{(result.probabilities.fake * 100).toFixed(1)}%</span>
                </div>
                <ProbBar fake={result.probabilities.fake} />
              </div>

              <p style={{ fontSize: 13, color: '#64748b', marginTop: 20, lineHeight: 1.6 }}>
                {result.prediction === 'FAKE'
                  ? "Neural analysis detected synthetic artifacts and frame inconsistency patterns typical of Al-generated content."
                  : "Scanning complete. Media shows high neural consistency and natural texture distribution patterns."}
              </p>
            </div>
          )}

          {result && result.error && (
            <div style={{
              marginTop: 20,
              padding: 16,
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.2)',
              borderRadius: 12,
              color: '#fca5a5',
              fontSize: 14
            }}>
              ⚠️ Alert: {result.error}
            </div>
          )}
        </div>

        <div style={{ width: 400, minHeight: 300, background: 'rgba(0,0,0,0.3)', borderRadius: 24, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {preview ? (tab === 'image' ? (
            <img src={preview} alt="preview" style={{ maxWidth: '100%', maxHeight: 400, display: 'block' }} />
          ) : (
            <video src={preview} controls style={{ maxWidth: '100%', maxHeight: 400, display: 'block' }} />
          )) : (
            <div style={{ color: '#475569', fontSize: '1.1em', fontWeight: 500 }}>Media Preview</div>
          )}
        </div>
      </div>
      <Robot />
    </div>
  )
}

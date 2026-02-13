import { useState } from 'react'

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
      const res = await fetch(endpoint, { method: 'POST', body: form })
      const json = await res.json()
      setResult(json)
    } catch (e) {
      setResult({ error: String(e) })
    } finally {
      setLoading(false)
    }
  }
  function ProbBar({fake, real}){
    const pctFake = Math.round(fake*100)
    return (
      <div style={{width:'100%', background:'#eee', height:14, borderRadius:7, overflow:'hidden'}}>
        <div style={{width:`${pctFake}%`, height:'100%', background:'#e85'}} />
      </div>
    )
  }

  return (
    <div style={{padding:20, maxWidth:900, margin:'0 auto'}}>
      <h1>Deepfake Detector</h1>
      <div style={{display:'flex', gap:8, marginBottom:12}}>
        <button onClick={()=>setTab('video')} style={{fontWeight: tab==='video' ? '700' : '400'}}>Video</button>
        <button onClick={()=>setTab('image')} style={{fontWeight: tab==='image' ? '700' : '400'}}>Image</button>
      </div>

      <div style={{display:'flex', gap:20}}>
        <div style={{flex:1}}>
          <input type="file" accept={tab==='video'?"video/*":"image/*"} onChange={handleFile} />
          <div style={{marginTop:12}}>
            <button onClick={submit} disabled={!file || loading}>{loading? 'Analyzing...' : 'Analyze'}</button>
          </div>

          {result && !result.error && (
            <div style={{marginTop:16}}>
              <h3 style={{margin:0}}>Prediction: <span style={{color: result.prediction==='FAKE' ? '#c00' : '#0a0'}}>{result.prediction}</span></h3>
              <div style={{fontSize:12, color:'#666'}}>{(result.confidence*100).toFixed(1)}% confidence</div>
              <div style={{marginTop:8}}>
                <ProbBar fake={result.probabilities.fake} real={result.probabilities.real} />
                <div style={{display:'flex', justifyContent:'space-between', fontSize:12, marginTop:6}}>
                  <div style={{color:'#c00'}}>Fake: {(result.probabilities.fake*100).toFixed(1)}%</div>
                  <div style={{color:'#0a0'}}>Real: {(result.probabilities.real*100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          )}

          {result && result.error && (
            <div style={{marginTop:12, color:'#900'}}>Error: {String(result.error)}</div>
          )}
        </div>
        <div style={{width:420}}>
          {preview && (tab==='image' ? (
            <img src={preview} alt="preview" style={{maxWidth:'100%'}} />
          ) : (
            <video src={preview} controls style={{maxWidth:'100%'}} />
          ))}
        </div>
      </div>
    </div>
  )
}

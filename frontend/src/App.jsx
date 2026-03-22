import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Play, UploadCloud, Camera, Eye, FileText, Layers, ShieldCheck, AlertTriangle, Mic, Type } from 'lucide-react';
import ScannerView from './components/ScannerView';
import CaptureView from './components/CaptureView';

const API_BASE = "http://127.0.0.1:8000";

// --- STARRY BACKGROUND ---
function StarBackground() {
  const stars = Array.from({ length: 70 }, (_, i) => ({
    id: i,
    left: Math.random() * 100,
    top: Math.random() * 100,
    size: Math.random() * 3 + 1,
    delay: Math.random() * 5,
  }));

  return (
    <div className="star-bg">
      {stars.map((s) => (
        <div
          key={s.id}
          className="star"
          style={{
            left: `${s.left}%`,
            top: `${s.top}%`,
            width: s.size,
            height: s.size,
            animationDelay: `${s.delay}s`,
          }}
        />
      ))}
    </div>
  );
}

export default function App() {
  const [view, setView] = useState('home'); // home, scanning, result
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isCaptureViewOpen, setIsCaptureViewOpen] = useState(false);
  const [selectedModality, setSelectedModality] = useState(null);

  const startAnalysis = async (inputData, modality) => {
    setSelectedModality(modality);
    setView('scanning');
    try {
      const formData = new FormData();
      let endpoint = '';
      if (modality === 'text') {
        endpoint = '/predict_text';
        formData.append('current_text', inputData);
      } else {
        endpoint = modality === 'image' ? '/predict_image' : 
                   modality === 'audio' ? '/predict_audio' : '/predict';
        formData.append('file', inputData);
      }
      
      const response = await axios.post(`${API_BASE}${endpoint}`, formData);
      setAnalysisResult(response.data);
    } catch (err) {
      console.error(err);
      alert("Analysis Failed: " + (err.response?.data?.detail || err.message));
      setView('home');
    }
  };

  const handleFileUpload = (e, modality) => {
    const f = e.target.files[0];
    if (f) startAnalysis(f, modality);
  };

  return (
    <div className="min-h-screen relative overflow-hidden" style={{ backgroundColor: '#000', color: '#fff' }}>
      <StarBackground />

      {/* ─── NAVBAR (Matching Reference Image) ─── */}
      <nav className="relative z-50 flex items-center justify-between px-10 py-6">
        {/* LOGO */}
        <div 
          className="flex items-center gap-4 cursor-pointer" 
          onClick={() => setView('home')}
        >
          <div className="w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold text-black" style={{ backgroundColor: '#7ec8a0', boxShadow: '0 0 20px rgba(126, 200, 160, 0.6)' }}>
            DF
          </div>
          <span className="text-3xl font-bold tracking-wide" style={{ color: '#7ec8a0', textShadow: '0 0 10px rgba(126, 200, 160, 0.4)' }}>
            DeepFake
          </span>
        </div>

        {/* LINKS */}
        <div className="hidden md:flex items-center gap-8">
          <div className="nav-link active">Home</div>
          <div className="nav-link">About</div>
          <div className="nav-link">Login or Signup</div>
          <div className="nav-link">Contact Us</div>
        </div>
      </nav>

      {/* ─── MAIN CONTENT ─── */}
      <main className="relative z-10 w-full h-[85vh] flex items-center justify-center px-10">
        
        {view === 'home' && (
          <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            
            {/* LEFT SIDE: 3D Visualization */}
            <div className="flex justify-center items-center">
              <div 
                className="relative w-80 h-96 border rounded-2xl flex items-center justify-center mesh-glow overflow-hidden" 
                style={{ backgroundColor: 'rgba(20, 20, 20, 0.5)', borderColor: 'rgba(126, 200, 160, 0.3)' }}
              >
                {/* Simulated 3D Head background */}
                <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full opacity-30" style={{ color: '#7ec8a0' }}>
                   <path fill="none" stroke="currentColor" strokeWidth="0.5" d="M30 20 Q50 0 70 20 T70 60 Q50 90 30 60 T30 20 M50 0 V90 M30 40 H70 M35 60 H65 M40 20 H60" />
                   <circle cx="50" cy="40" r="4" fill="currentColor" className="animate-pulse" />
                   <circle cx="35" cy="40" r="1.5" fill="currentColor" />
                   <circle cx="65" cy="40" r="1.5" fill="currentColor" />
                </svg>
                {/* Scanning line */}
                <div className="absolute top-0 left-0 right-0 h-1" style={{ background: '#7ec8a0', boxShadow: '0 0 15px #7ec8a0', animation: 'scan-line 3s linear infinite' }} />
              </div>
            </div>

            {/* RIGHT SIDE: Text and Actions */}
            <div className="text-left space-y-6">
              {/* Glowing Blur Behind Text */}
              <div className="absolute top-1/2 right-1/4 w-96 h-32 blur-[100px] pointer-events-none rounded-full" style={{ background: 'rgba(126, 200, 160, 0.25)' }} />

              <h1 className="text-5xl lg:text-7xl font-bold tracking-tight glow-text leading-tight">
                Deepfake Image & <br/> Video Detection
              </h1>
              
              <p className="text-lg text-gray-300 max-w-xl">
                Using Deep Learning for forensic analysis and digital evidence validation.
              </p>

              <div className="flex flex-wrap gap-4 pt-8">
                <button 
                  className="action-btn flex items-center gap-2"
                  onClick={() => document.getElementById('video-upload').click()}
                >
                  <Play size={20} /> Analyze Video
                </button>
                <input id="video-upload" type="file" accept="video/*" className="hidden" onChange={(e) => handleFileUpload(e, 'video')} />
                
                <button 
                  className="action-btn flex items-center gap-2"
                  onClick={() => document.getElementById('image-upload').click()}
                >
                  <UploadCloud size={20} /> Upload Image
                </button>
                <input id="image-upload" type="file" accept="image/*" className="hidden" onChange={(e) => handleFileUpload(e, 'image')} />

                <button 
                  className="action-btn flex items-center gap-2"
                  onClick={() => document.getElementById('audio-upload').click()}
                >
                  <Mic size={20} /> Analyze Audio
                </button>
                <input id="audio-upload" type="file" accept="audio/*" className="hidden" onChange={(e) => handleFileUpload(e, 'audio')} />

                <button 
                  className="action-btn flex items-center gap-2"
                  onClick={() => {
                    const txt = prompt("Paste your text for semantic AI analysis:");
                    if (txt) startAnalysis(txt, 'text');
                  }}
                >
                  <Type size={20} /> Verify Text
                </button>

                <button 
                  className="action-btn flex items-center gap-2"
                  onClick={() => { setSelectedModality('image'); setIsCaptureViewOpen(true); }}
                >
                  <Camera size={20} /> Live Snapshot
                </button>
              </div>

            </div>
          </div>
        )}

        {/* ─── SCANNING & CAPTURE & RESULTS ─── */}
        {view === 'scanning' && selectedModality && (
          <ScannerView
            modality={selectedModality}
            onComplete={() => setView('result')}
          />
        )}

        {isCaptureViewOpen && (
          <CaptureView
            modality={selectedModality}
            onCapture={(file) => {
              setIsCaptureViewOpen(false);
              startAnalysis(file, selectedModality);
            }}
            onClose={() => setIsCaptureViewOpen(false)}
          />
        )}

        {view === 'result' && analysisResult && (
          <div className="w-full max-w-6xl glass-panel p-10 rounded-[40px] fade-up space-y-8 h-[85vh] overflow-y-auto custom-scrollbar relative block mt-8 mb-8" style={{ border: '1px solid rgba(126, 200, 160, 0.15)', background: 'rgba(10, 10, 10, 0.85)', boxShadow: '0 30px 60px rgba(0,0,0,0.8)' }}>
            <button
              onClick={() => setView('home')}
              className="absolute top-8 right-8 flex items-center gap-2 text-gray-400 hover:text-white transition-all uppercase tracking-widest text-xs font-bold bg-black/40 px-5 py-3 rounded-full border border-gray-800 hover:border-[#7ec8a0]/50 hover:bg-[#7ec8a0]/10 z-50"
            >
              ← Back to Scanner
            </button>

            {/* Header: Score and Verdict */}
            <div className="flex flex-col md:flex-row items-center gap-12 border-b border-gray-800/50 pb-10">
              {/* Animated Circular Ring for Confidence */}
              <div className="relative flex items-center justify-center w-56 h-56 shrink-0 shrink">
                <div className="absolute inset-0 rounded-full animate-pulse blur" style={{ background: analysisResult.prediction === 'FAKE' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(34, 197, 94, 0.15)' }} />
                <svg className="absolute inset-0 w-full h-full transform -rotate-90 drop-shadow-lg">
                  <circle cx="112" cy="112" r="96" stroke="rgba(255,255,255,0.05)" strokeWidth="14" fill="none" />
                  <circle
                    cx="112" cy="112" r="96"
                    stroke={analysisResult.prediction === 'FAKE' ? '#ef4444' : '#22c55e'}
                    strokeWidth="14" fill="none"
                    strokeDasharray="603"
                    strokeDashoffset="603"
                    strokeLinecap="round"
                    style={{ animation: `fill-ring 1.5s cubic-bezier(0.4, 0, 0.2, 1) forwards` }}
                  />
                  <style>{`
                    @keyframes fill-ring {
                      from { stroke-dashoffset: 603; }
                      to { stroke-dashoffset: ${603 - (603 * analysisResult.confidence)}; }
                    }
                  `}</style>
                </svg>
                <div className="text-center relative z-10 flex flex-col items-center">
                  <span className={`text-6xl font-black tracking-tighter ${analysisResult.prediction === 'FAKE' ? 'text-red-500' : 'text-green-500'}`}>
                    {Math.round(analysisResult.confidence * 100)}<span className="text-3xl opacity-70">%</span>
                  </span>
                  <span className="text-[10px] text-white/50 font-bold uppercase tracking-[0.3em] mt-1">Confidence</span>
                </div>
              </div>

              <div className="flex-1 mt-4 md:mt-0 text-center md:text-left">
                <span className="text-xs font-black uppercase tracking-[0.4em] text-white/40 flex items-center justify-center md:justify-start gap-3">
                  <Layers size={14} className="text-[#7ec8a0]" /> Neural Analysis Protocol
                </span>
                <h1 className={`text-7xl font-black mt-3 tracking-tighter uppercase leading-none ${analysisResult.prediction === 'FAKE' ? 'text-red-500 drop-shadow-[0_0_30px_rgba(239,68,68,0.5)]' : 'text-green-500 drop-shadow-[0_0_30px_rgba(34,197,94,0.5)]'}`}>
                  {analysisResult.prediction} DECTECTED
                </h1>
                <p className="text-gray-400 mt-5 max-w-lg text-sm leading-relaxed mx-auto md:mx-0 font-medium">
                  Deep neural forensics indicate a <strong className="text-white text-base">{Math.round(analysisResult.confidence * 100)}%</strong> probability that the provided media stream contains <strong className="text-white text-base">{analysisResult.prediction.toLowerCase()}</strong> elements. The underlying biometric matrix has been validated against known baseline structures.
                </p>
              </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Visual Heatmap */}
              <div className="p-7 rounded-[30px] border border-gray-800/50 bg-black/60 hover:bg-black/80 transition-all duration-300 group shadow-lg">
                <div className="flex items-center gap-3 mb-5">
                  <div className="p-2 rounded-xl bg-[#7ec8a0]/10 text-[#7ec8a0]">
                    <Eye size={18} />
                  </div>
                  <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-white/80">Anomaly Heatmap</span>
                </div>
                <div className="h-56 rounded-2xl overflow-hidden border border-white/5 bg-[#050505] flex items-center justify-center relative group-hover:border-[#7ec8a0]/30 transition-colors">
                    {analysisResult.forensics?.heatmap ? (
                        <>
                          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent z-10" />
                          <img src={`data:image/jpeg;base64,${analysisResult.forensics.heatmap}`} className="w-full h-full object-cover opacity-70 group-hover:opacity-100 transition-all group-hover:scale-110 duration-700 ease-out" alt="Heatmap" />
                        </>
                    ) : (
                        <div className="text-center opacity-40">
                            <Camera size={32} className="mx-auto mb-3" />
                            <span className="text-[10px] tracking-[0.2em] uppercase">Visual Cortex Offline</span>
                        </div>
                    )}
                </div>
              </div>

              {/* Neural Metrics / Statistics */}
              <div className="col-span-1 lg:col-span-2 p-7 rounded-[30px] border border-gray-800/50 bg-black/60 hover:bg-black/80 transition-all duration-300 shadow-lg">
                <div className="flex items-center justify-between mb-5">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-xl bg-[#7ec8a0]/10 text-[#7ec8a0]">
                      <AlertTriangle size={18} />
                    </div>
                    <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-white/80">Diagnostic Telemetry</span>
                  </div>
                  {(analysisResult.emotion || analysisResult.forensics?.vocal_jitter) && (
                     <span className="px-3 py-1 rounded-full bg-white/5 text-[10px] uppercase tracking-widest text-[#7ec8a0] border border-white/5 animate-pulse">
                        Sub-routine Active
                     </span>
                  )}
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 h-[224px] content-start">
                  {analysisResult.neural_metrics ? (
                    Object.entries(analysisResult.neural_metrics).map(([key, val], idx) => (
                      <div key={key} className="bg-white/[0.02] p-4 rounded-2xl border border-white/5 relative overflow-hidden group hover:bg-white/[0.04] transition-colors" style={{ animation: `fade-up 0.4s ${idx * 0.1 + 0.2}s both` }}>
                        <div className="absolute inset-0 bg-gradient-to-r from-[#7ec8a0]/10 to-transparent w-full h-full opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                        <div className="text-[10px] text-white/40 font-bold uppercase tracking-[0.2em] mb-3 relative z-10 truncate">{key.replace(/_/g, ' ')}</div>
                        <div className="flex items-center gap-4 relative z-10">
                          <div className="text-2xl font-black text-white w-16">{(val * 100).toFixed(1)}<span className="text-sm text-white/50">%</span></div>
                          <div className="h-1.5 flex-grow bg-gray-900 rounded-full overflow-hidden shadow-inner">
                            <div className="h-full bg-gradient-to-r from-[#7ec8a0]/50 to-[#7ec8a0] rounded-full relative" style={{ width: `${val * 100}%` }}>
                              <div className="absolute top-0 right-0 bottom-0 w-4 bg-white/30 blur-[2px]" />
                            </div>
                          </div>
                        </div>
                      </div>
                    ))
                  ) : analysisResult.forensics?.source_attribution ? (
                    <>
                      <div className="bg-white/[0.02] p-5 rounded-2xl border border-white/5 flex flex-col justify-center">
                        <div className="text-[10px] text-white/40 font-bold uppercase tracking-[0.2em] mb-2">Estimated Generation Engine</div>
                        <div className="text-2xl font-black text-[#7ec8a0] tracking-tight">{analysisResult.forensics.source_attribution.most_likely}</div>
                        <div className="text-xs text-white/30 mt-2">Signature matched against known GAN/Diffusion models.</div>
                      </div>
                      <div className="bg-white/[0.02] p-5 rounded-2xl border border-white/5 flex flex-col justify-center">
                        <div className="text-[10px] text-white/40 font-bold uppercase tracking-[0.2em] mb-2">Biometric Mesh Integrity</div>
                        <div className="flex items-end gap-3">
                          <div className="text-5xl font-black text-white leading-none">{(analysisResult.forensics.mesh_integrity?.integrity_score * 100).toFixed(0)}<span className="text-2xl text-white/30">%</span></div>
                        </div>
                        <div className="h-1.5 w-full bg-gray-900 rounded-full overflow-hidden mt-4 shadow-inner">
                           <div className="h-full bg-gradient-to-r from-[#7ec8a0]/50 to-[#7ec8a0]" style={{ width: `${analysisResult.forensics.mesh_integrity?.integrity_score * 100}%` }} />
                        </div>
                      </div>
                    </>
                  ) : analysisResult.forensics && Object.keys(analysisResult.forensics).filter(k => !['findings', 'heatmap', 'ela', 'fft', 'noise', 'metadata', 'source_attribution', 'mesh_integrity'].includes(k)).length > 0 ? (
                    Object.entries(analysisResult.forensics).filter(([k,v]) => !['findings', 'heatmap', 'ela', 'fft', 'noise', 'metadata', 'source_attribution', 'mesh_integrity'].includes(k)).map(([key, val], idx) => (
                      <div key={key} className="bg-white/[0.02] p-4 rounded-2xl border border-white/5 relative overflow-hidden group hover:bg-white/[0.04] transition-colors" style={{ animation: `fade-up 0.4s ${idx * 0.1 + 0.2}s both` }}>
                        <div className="absolute inset-0 bg-gradient-to-r from-[#7ec8a0]/10 to-transparent w-full h-full opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                        <div className="text-[10px] text-white/40 font-bold uppercase tracking-[0.2em] mb-3 relative z-10 truncate">{key.replace(/_/g, ' ')}</div>
                        <div className="flex items-center gap-4 relative z-10">
                          <div className="text-xl font-black text-white">{typeof val === 'number' ? val.toFixed(4) : typeof val === 'boolean' ? (val ? 'TRUE' : 'FALSE') : String(val)}</div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="col-span-2 flex flex-col items-center justify-center h-full text-white/20 uppercase tracking-[0.3em] text-xs gap-3">
                       <span className="animate-spin w-4 h-4 border-2 border-white/20 border-t-[#7ec8a0] rounded-full"/>
                       Aggregating Metrics...
                    </div>
                  )}
                </div>
              </div>

              {/* Terminal Logs */}
              <div className="col-span-1 lg:col-span-3 p-7 rounded-[30px] border border-gray-800/50 bg-[#050505] overflow-hidden shadow-[inset_0_4px_20px_rgba(0,0,0,0.5)]">
                <div className="flex items-center justify-between mb-5 border-b border-white/5 pb-4">
                  <div className="flex items-center gap-3">
                    <FileText size={18} className="text-[#7ec8a0]" />
                    <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-white/80">Live Execution Terminal</span>
                  </div>
                  <div className="flex gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/50" />
                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50" />
                    <div className="w-2.5 h-2.5 rounded-full bg-green-500/50" />
                  </div>
                </div>
                <div className="space-y-4 font-mono text-[11px] tracking-wide text-white/60 bg-transparent h-48 overflow-y-auto custom-scrollbar pr-4">
                    {/* Fake typing effect for logs */}
                    {["Initializing forensic sub-routines...", 
                      "Loading weights into VRAM...",
                      "Executing spatial consistency checks...", 
                      "Calculating pixel-level deviations...",
                      ...(analysisResult.forensics?.findings?.length > 0 ? analysisResult.forensics.findings : ["No advanced topological anomalies recorded."]),
                      "Process _terminating with code 0."
                    ].map((log, i) => (
                        <div key={i} className="flex gap-4 items-start" style={{ animation: `fade-up 0.5s ${i * 0.15 + 0.5}s both` }}>
                            <span className="text-[#7ec8a0]/60 mt-px select-none shrink-0 border-r border-white/10 pr-4">
                              {new Date(Date.now() + i*753).toISOString().split('T')[1].substring(0, 12)}
                            </span>
                            <span className={`${log.includes('ANOMALIES') || log.includes('AI') || log.includes('Suspicious') ? 'text-red-400 bg-red-500/10 px-2 py-0.5 rounded' : 'text-gray-300'}`}>
                              <span className="mr-2 text-white/30">{'>'}</span>{log}
                            </span>
                        </div>
                    ))}
                </div>
              </div>

            </div>
          </div>
        )}
      </main>
      
      {/* GLOBAL KEYFRAMES */}
      <style>{`
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #7ec8a0; border-radius: 4px; }
        @keyframes scan-line { 0% { top: 0%; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
        .fade-up { animation: fade-up 0.6s ease forwards; }
        @keyframes fade-up { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}

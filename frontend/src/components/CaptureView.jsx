import React, { useRef, useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Camera, X, Circle, Square, Mic, Volume2 } from 'lucide-react';

export default function CaptureView({ modality, onCapture, onClose }) {
    const videoRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const [isRecording, setIsRecording] = useState(false);
    const [stream, setStream] = useState(null);
    const chunksRef = useRef([]);

    useEffect(() => {
        startStream();
        return () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, [modality]);

    const startStream = async () => {
        try {
            const constraints = {
                video: modality === 'image' || modality === 'video' ? { facingMode: 'user' } : false,
                audio: modality === 'audio' || modality === 'video' ? true : false,
            };

            const newStream = await navigator.mediaDevices.getUserMedia(constraints);
            setStream(newStream);
            if (videoRef.current) {
                videoRef.current.srcObject = newStream;
            }
        } catch (err) {
            console.error("Error accessing media devices:", err);
            alert("Neural Interface Error: Unable to access camera or microphone.");
            onClose();
        }
    };

    const captureImage = () => {
        if (!videoRef.current || videoRef.current.readyState < 2) {
            console.warn("Video stream not ready for capture");
            return;
        }

        console.log("Capturing image from stream...");
        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0);

        canvas.toBlob((blob) => {
            if (!blob) {
                console.error("Failed to generate blob from canvas");
                return;
            }
            console.log("Blob generated, size:", blob.size);
            const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
            onCapture(file);
        }, 'image/jpeg');
    };

    const startRecording = () => {
        if (!stream) {
            console.warn("No active stream to record");
            return;
        }

        chunksRef.current = [];

        // Find best supported mime type
        let mimeType = '';
        if (modality === 'audio') {
            mimeType = ['audio/webm', 'audio/mp4', 'audio/wav'].find(type => MediaRecorder.isTypeSupported(type));
        } else {
            mimeType = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4'].find(type => MediaRecorder.isTypeSupported(type));
        }

        const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                chunksRef.current.push(e.data);
            }
        };

        recorder.onstop = () => {
            const blob = new Blob(chunksRef.current, { type: recorder.mimeType || (modality === 'audio' ? 'audio/webm' : 'video/webm') });
            const filename = modality === 'audio' ? "capture.webm" : "capture.webm";
            const file = new File([blob], filename, { type: blob.type });
            onCapture(file);
        };

        recorder.start();
        setIsRecording(true);
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    return (
        <div className="fixed inset-0 z-[100] bg-dark/95 backdrop-blur-3xl flex items-center justify-center p-6">
            <div className="relative w-full max-w-4xl glass-morphism rounded-[3rem] border-primary/20 overflow-hidden shadow-2xl">
                {/* Header */}
                <div className="p-6 flex justify-between items-center border-b border-white/5">
                    <div className="flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                        <span className="text-xs font-black uppercase tracking-[0.3em] text-primary">Live Neural Capture</span>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full text-dim hover:text-white transition-all">
                        <X size={20} />
                    </button>
                </div>

                {/* Viewport */}
                <div className="relative aspect-video bg-black flex items-center justify-center">
                    {modality !== 'audio' ? (
                        <video
                            ref={videoRef}
                            autoPlay
                            muted
                            playsInline
                            className="w-full h-full object-cover"
                        />
                    ) : (
                        <div className="flex flex-col items-center gap-6">
                            <div className="w-32 h-32 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center relative">
                                <Mic className="text-primary" size={48} />
                                <motion.div
                                    animate={{ scale: [1, 1.2, 1] }}
                                    transition={{ repeat: Infinity, duration: 1.5 }}
                                    className="absolute inset-0 rounded-full border border-primary/30"
                                />
                            </div>
                            <span className="text-dim text-sm font-bold uppercase tracking-widest">Audio Stream Active</span>
                        </div>
                    )}

                    {isRecording && (
                        <div className="absolute top-6 left-6 flex items-center gap-2 px-3 py-1 rounded-full bg-danger/20 border border-danger/40">
                            <div className="w-2 h-2 rounded-full bg-danger animate-ping" />
                            <span className="text-[10px] font-black uppercase tracking-widest text-danger">Recording...</span>
                        </div>
                    )}
                </div>

                {/* Controls */}
                <div className="p-12 flex flex-col items-center gap-8 bg-gradient-to-t from-dark/50 to-transparent">
                    {modality === 'image' ? (
                        <div className="flex flex-col items-center gap-4">
                            <button
                                onClick={captureImage}
                                className="w-24 h-24 rounded-full bg-white flex items-center justify-center text-dark hover:scale-110 transition-transform shadow-[0_0_40px_rgba(255,255,255,0.4)] border-4 border-primary/20"
                            >
                                <Camera size={38} />
                            </button>
                            <span className="text-sm font-black uppercase tracking-widest text-white">Snap & Analyze</span>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center gap-4">
                            <button
                                onClick={isRecording ? stopRecording : startRecording}
                                className={`w-24 h-24 rounded-full flex items-center justify-center transition-all shadow-2xl border-4 ${isRecording ? 'bg-white text-dark scale-90 border-danger/20' : 'bg-danger text-white hover:scale-110 border-white/20'}`}
                            >
                                {isRecording ? <Square size={38} /> : <Circle size={38} fill="currentColor" />}
                            </button>
                            <span className={`text-sm font-black uppercase tracking-widest ${isRecording ? 'text-danger animate-pulse' : 'text-white'}`}>
                                {isRecording ? 'Stop Recording' : 'Start Feed Capture'}
                            </span>
                        </div>
                    )}

                    <p className="text-dim text-xs font-bold uppercase tracking-widest opacity-40">
                        {modality === 'image' ? "Press to capture frame" : isRecording ? "Press to stop recording" : "Press to start stream capture"}
                    </p>
                </div>
            </div>
        </div>
    );
}

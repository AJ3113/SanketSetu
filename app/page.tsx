'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { authClient } from '@/lib/auth-client'

export default function Page() {
  const { data: session, isPending } = authClient.useSession()
  const [menu, setMenu] = useState(false)
  const [cameraOn, setCameraOn] = useState(false)
  const [status, setStatus] = useState('Ready')
  const [phrase, setPhrase] = useState<string[]>([])
  const [confidence, setConfidence] = useState<number | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    if (!cameraOn) return
    let stream: MediaStream | null = null
    navigator.mediaDevices?.getUserMedia({ video: true }).then((nextStream) => {
      stream = nextStream
      if (videoRef.current) videoRef.current.srcObject = nextStream
      setStatus('Camera ready')
    }).catch(() => { setStatus('Camera permission needed'); setCameraOn(false) })
    return () => { stream?.getTracks().forEach((track) => track.stop()) }
  }, [cameraOn])

  function startDetection() { setStatus('Starting'); setCameraOn(true) }
  function stopDetection() {
    const stream = videoRef.current?.srcObject as MediaStream | null
    stream?.getTracks().forEach((track) => track.stop())
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraOn(false); setStatus('Translation ended'); setConfidence(null)
  }
  function addDemoSign() { setPhrase((current) => [...current, current.length % 2 ? 'THANK YOU' : 'HELLO']); setConfidence(96) }
  const name = session?.user?.name || 'friend'
  return <main className="app-shell">
    <header className="app-header"><Link className="brand" href="/"><Image src="/logo.png" alt="SanketSetu Logo" width={55} height={30} className="brand-mark" priority />SanketSetu</Link><nav className={menu ? 'nav-open' : ''}><a href="#translate">Translate</a><Link href="/conversation">Conversation</Link><Link href="/learn">Learn Signs</Link><Link href="/help">Help</Link>{session?.user ? <button onClick={() => authClient.signOut({ fetchOptions: { onSuccess: () => location.reload() } })}>Log out</button> : <Link className="nav-login" href="/sign-in">Log in</Link>}</nav><button className="mobile-menu" onClick={() => setMenu(!menu)} aria-label="Open menu">☰</button></header>
    <section className="genz-hero"><div><p className="kicker">SANKETSETU / BETA DROP</p><h1>Real-Time Indian Sign Language <span>to Text & Voice Translator.</span></h1><p className="hero-copy">{isPending ? 'Loading your space...' : session?.user ? `Welcome back, ${name}.` : 'A louder, faster, more human way to communicate.'}</p><div className="hero-actions"><a className="primary-button" href="#translate">Open translator <span>↗</span></a><Link className="ghost-button" href="/conversation">Conversation mode</Link></div><div className="stack-tags"><span>React JS</span><span>FastAPI</span><span>MediaPipe</span><span>gTTS / STT</span><span>SQLite DB</span><span>AI Tutor</span></div></div><div className="hero-terminal"><div className="terminal-bar"><i /><i /><i /><span>isl_detection.exe</span></div><div className="terminal-content"><p><b>›</b> sanketsetu boot --live</p><p className="green">✓ frontend online</p><p className="cyan">✓ camera pipeline ready</p><p className="pink">! classifier: awaiting model</p><div className="cursor-line">&gt; <span /></div></div></div></section>
    <section id="translate" className="workspace"><div className="section-head"><div><p className="kicker">01 / TRANSLATE</p><h2>Hands in. <span>Words out.</span></h2></div><p>Build a phrase from each recognized sign. The session transcript stays together until you clear it.</p></div><div className="translate-grid"><article className="camera-card"><div className="card-top"><span className="live-dot" /> <span>{cameraOn ? 'LIVE CAMERA' : 'CAMERA OFF'}</span><span className="status-pill">{status}</span></div><div className="camera-preview">{cameraOn ? <video ref={videoRef} autoPlay muted playsInline /> : <div className="camera-placeholder"><div className="wave-symbol">⌁</div><strong>Camera is off</strong><span>Start translation when you are ready.</span></div>}</div><div className="camera-actions">{cameraOn ? <><button className="danger-button" onClick={stopDetection}>End translation <span>■</span></button><button className="ghost-button" onClick={addDemoSign}>Add detected sign</button></> : <button className="primary-button" onClick={startDetection}>Activate camera <span>↗</span></button>}<Link className="ghost-button" href="/conversation">Open conversation</Link></div></article><article className="output-card"><p className="kicker">SESSION PHRASE</p><div className="output-symbol">⌁</div><h3>{phrase.length ? phrase.join(' ') : 'Your phrase appears here.'}</h3><p>{cameraOn ? 'Recognized signs will be appended into one live sentence.' : 'Activate the camera to begin building a phrase.'}</p><div className="confidence-row"><span>CONFIDENCE</span><span>{confidence ? `${confidence}%` : '--'}</span></div><div className="confidence-bar"><span style={{ width: `${confidence || 0}%` }} /></div><div className="phrase-actions"><button className="text-button" onClick={() => { setPhrase([]); setConfidence(null) }}>Clear session</button><button className="text-button" onClick={() => phrase.length && window.speechSynthesis?.speak(new SpeechSynthesisUtterance(phrase.join(' ')))}>Speak phrase</button></div></article></div></section>
    <section id="learn" className="learn-strip"><div><p className="kicker">02 / LEARN SIGNS</p><h2>Practice it. <span>Own it.</span></h2><p>Pick a sign, compare your attempt, and get instant feedback.</p></div><Link className="ghost-button" href="/learn">Browse sign library <span>↗</span></Link></section><footer><span>SanketSetu © 2026</span><span>Built for clearer conversations.</span><Link href="/help">Help center</Link></footer>
  </main>
}

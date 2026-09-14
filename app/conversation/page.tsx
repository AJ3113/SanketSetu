'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'

type Message = { from: string; text: string; kind?: 'sign' | 'text' }

export default function ConversationPage() {
  const [cameraOn, setCameraOn] = useState(false)
  const [cameraStatus, setCameraStatus] = useState('Ready')
  const [listening, setListening] = useState(false)
  const [draft, setDraft] = useState('')
  const [phrase, setPhrase] = useState<string[]>([])
  const [messages, setMessages] = useState<Message[]>([{ from: 'SanketSetu', text: 'Conversation mode is ready. Start the camera for sign input, or type to reply.' }])
  const videoRef = useRef<HTMLVideoElement>(null)
  const recognition = useRef<{ stop: () => void } | null>(null)

  useEffect(() => {
    if (!cameraOn) return
    let stream: MediaStream | null = null
    navigator.mediaDevices?.getUserMedia({ video: true }).then((nextStream) => { stream = nextStream; if (videoRef.current) videoRef.current.srcObject = nextStream; setCameraStatus('Camera ready') }).catch(() => { setCameraStatus('Permission needed'); setCameraOn(false) })
    return () => stream?.getTracks().forEach((track) => track.stop())
  }, [cameraOn])

  function toggleCamera() { if (cameraOn) { const stream = videoRef.current?.srcObject as MediaStream | null; stream?.getTracks().forEach((track) => track.stop()); if (videoRef.current) videoRef.current.srcObject = null; setCameraOn(false); setCameraStatus('Translation ended') } else { setCameraStatus('Starting'); setCameraOn(true) } }
  function addDetectedSign() { const sign = phrase.length % 2 ? 'THANK YOU' : 'HELLO'; setPhrase((current) => [...current, sign]); setMessages((current) => [...current, { from: 'Sign user', text: [...phrase, sign].join(' '), kind: 'sign' }]) }
  function toggleListening() { if (listening) { recognition.current?.stop(); setListening(false); return }; const speechWindow = window as Window & { SpeechRecognition?: new () => any; webkitSpeechRecognition?: new () => any }; const SpeechRecognitionClass = speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition; if (!SpeechRecognitionClass) return; const next = new SpeechRecognitionClass(); next.lang = 'en-IN'; next.continuous = false; next.onresult = (event: any) => setDraft(event.results[0][0].transcript); next.onend = () => setListening(false); recognition.current = next; next.start(); setListening(true) }
  function sendMessage() { const text = draft.trim(); if (!text) return; setMessages((current) => [...current, { from: 'You', text, kind: 'text' }]); setDraft('') }
  return <main className="app-shell"><header className="app-header"><Link className="brand" href="/"><Image src="/logo.png" alt="SanketSetu Logo" width={55} height={30} className="brand-mark" priority />SanketSetu</Link><nav><Link href="/">Translate</Link><Link className="active-link" href="/conversation">Conversation</Link><Link href="/learn">Learn Signs</Link><Link href="/help">Help</Link></nav></header><section className="conversation-shell"><div className="conversation-intro"><p className="kicker">02 / CONVERSATION</p><h1>Signs <span>↔</span> Speech.</h1><p>Sign users can send a live phrase from the camera. Non-sign users can reply by typing or speaking in the same chat.</p></div><div className="conversation-layout"><article className="sign-input-card"><div className="card-top"><span className="live-dot" /><span>{cameraOn ? 'LIVE CAMERA' : 'CAMERA OFF'}</span><span className="status-pill">{cameraStatus}</span></div><div className="conversation-camera">{cameraOn ? <video ref={videoRef} autoPlay muted playsInline /> : <div className="camera-placeholder"><div className="wave-symbol">⌁</div><strong>Sign input is off</strong><span>Activate camera to send signs.</span></div>}</div><div className="camera-actions">{cameraOn ? <><button className="danger-button" onClick={toggleCamera}>End translation <span>■</span></button><button className="ghost-button" onClick={addDetectedSign}>Add detected sign</button></> : <button className="primary-button" onClick={toggleCamera}>Activate camera <span>↗</span></button>}</div><div className="phrase-mini"><small>LIVE PHRASE</small><strong>{phrase.length ? phrase.join(' ') : 'No signs in this session yet.'}</strong><button className="text-button" onClick={() => setPhrase([])}>Clear phrase</button></div></article><div className="chat-card"><div className="chat-head"><strong>LIVE CHAT</strong><span><i className="live-dot" /> Private session</span></div><div className="message-list">{messages.map((message, index) => <div className={`message ${message.from === 'You' ? 'mine' : ''}`} key={`${message.from}-${index}`}><small>{message.from}{message.kind === 'sign' ? ' · TRANSLATED' : ''}</small><p>{message.text}</p><button onClick={() => window.speechSynthesis?.speak(new SpeechSynthesisUtterance(message.text))}>Speak aloud</button></div>)}</div><div className="conversation-composer"><button className={listening ? 'mic-button listening' : 'mic-button'} onClick={toggleListening} aria-label={listening ? 'Stop listening' : 'Start listening'}>{listening ? <span className="mic-glyph mic-stop" aria-hidden="true" /> : <span className="mic-glyph" aria-hidden="true" />}</button><input className="conversation-input" value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.nativeEvent.isComposing && event.keyCode !== 229) sendMessage() }} placeholder="Reply with text or use your mic..." /><button className="primary-button" onClick={sendMessage}>Send <span>↗</span></button></div><div className="chat-options"><span>Auto-speak: OFF</span><button onClick={() => setMessages([])}>Clear session</button></div></div></div></section></main>
}

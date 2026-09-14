'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'

const signs = [
  ['HELLO', 'Basics', 'Open palm, friendly wave'], ['THANK YOU', 'Basics', 'Touch fingertips to chin, then move forward'], ['PLEASE', 'Basics', 'Rub open palm over chest'], ['YES', 'Basics', 'Make a fist and nod it up and down'], ['NO', 'Basics', 'Tap index and middle fingers against thumb'], ['GOOD MORNING', 'Basics', 'Good sign followed by sunrise gesture'], ['GOOD NIGHT', 'Basics', 'Good sign followed by sleep gesture'], ['I', 'Basics', 'Point gently toward yourself'], ['NAME', 'Basics', 'Use both hands to show name cards'], ['WATER', 'Daily', 'W hand moves toward mouth'], ['FOOD', 'Daily', 'Bring fingertips toward mouth'], ['BATHROOM', 'Daily', 'Closed fist with thumb between fingers'], ['HOSPITAL', 'Emergency', 'Draw a cross over the palm'], ['DOCTOR', 'Emergency', 'Doctor gesture at wrist'], ['EMERGENCY', 'Emergency', 'Raise both hands to signal urgency'], ['STOP', 'Emergency', 'Open palm held forward'], ['HELP', 'Needs & Feelings', 'One fist lifts on an open palm'], ['NEED', 'Needs & Feelings', 'Flat hands press down firmly'], ['WANT', 'Needs & Feelings', 'Hands pull toward the body'], ['SORRY', 'Needs & Feelings', 'Fist circles over the chest'],
]
const categories = ['All', 'Basics', 'Daily', 'Emergency', 'Needs & Feelings']

export default function LearnPage() {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('All')
  const [selected, setSelected] = useState<(typeof signs)[number] | null>(null)
  const visibleSigns = useMemo(() => signs.filter(([name, group]) => (category === 'All' || group === category) && name.toLowerCase().includes(query.toLowerCase())), [category, query])
  return <main className="library-shell">
    <header className="simple-header"><Link className="brand" href="/"><Image src="/logo.png" alt="SanketSetu Logo" width={55} height={30} className="brand-mark" priority />SanketSetu</Link><nav className="library-nav"><Link href="/">Translate</Link><Link href="/conversation">Conversation</Link><Link className="active-link" href="/learn">Learn Signs</Link><Link href="/help">Help</Link></nav></header>
    <section className="library-hero"><div><p className="kicker">02 / LEARN SIGNS</p><h1>Build your <span>sign fluency.</span></h1><p>Explore 20 commonly used ISL signs, then practice one at a time with a clear visual cue.</p></div><div className="library-count"><strong>{visibleSigns.length}</strong><span>signs in view</span></div></section>
    <section className="library-tools"><label className="search-box"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search signs" aria-label="Search signs" /></label><div className="category-tabs">{categories.map((item) => <button key={item} className={category === item ? 'selected' : ''} onClick={() => setCategory(item)}>{item}</button>)}</div></section>
    <section className="sign-grid">{visibleSigns.map((sign, index) => <button className="sign-card" key={sign[0]} onClick={() => setSelected(sign)}><span className="sign-number">{String(index + 1).padStart(2, '0')}</span><span className="sign-glyph">{sign[0].slice(0, 1)}</span><span className="sign-name">{sign[0]}</span><span className="sign-category">{sign[1]}</span><span className="sign-learn">Learn sign ↗</span></button>)}</section>
    {selected && <div className="sign-modal" role="dialog" aria-modal="true"><div className="sign-modal-card"><button className="modal-close" onClick={() => setSelected(null)} aria-label="Close sign details">×</button><p className="kicker">SIGN PRACTICE</p><div className="modal-glyph">{selected[0].slice(0, 1)}</div><h2>{selected[0]}</h2><p className="modal-category">{selected[1]}</p><p>{selected[2]}.</p><Link href="/#translate" className="primary-button" onClick={() => setSelected(null)}>Try in translator ↗</Link></div></div>}
  </main>
}

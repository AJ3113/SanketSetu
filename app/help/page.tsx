import Link from 'next/link'
import Image from 'next/image'

const faqs = [
  ['What is SanketSetu?', 'SanketSetu is a webcam-based Indian Sign Language (ISL) communication assistant that recognizes supported signs and converts them into text and speech. It also provides AI-guided practice for learning supported signs.'],
  ['How does SanketSetu work?', 'Your webcam captures the hand gesture → hand landmarks are extracted → the ML model identifies the sign → a confidence check validates the prediction → the result is displayed as text and, where applicable, converted to speech.'],
  ['What signs are currently supported?', 'The prototype currently supports commonly used ISL signs such as:\n\nBasics — HELLO, THANK YOU, PLEASE, YES, NO, GOOD MORNING, GOOD NIGHT, I, NAME\nDaily — WATER, FOOD, BATHROOM\nEmergency — HOSPITAL, DOCTOR, EMERGENCY, STOP\nNeeds & Feelings — HELP, NEED, WANT, SORRY'],
  ['What happens if the system is unsure about a sign?', 'SanketSetu uses a confidence-aware recognition mechanism. If the prediction does not meet the required confidence level, the system does not force a translation and instead asks the user to repeat the sign.'],
]
export default function HelpPage() { return <main className="help-shell"><header className="simple-header"><Link className="brand" href="/"><Image src="/logo.png" alt="SanketSetu Logo" width={55} height={30} className="brand-mark" priority />SanketSetu</Link><Link href="/" className="text-button">Back to app</Link></header><section className="help-hero"><p className="kicker">HELP CENTER</p><h1>Quick answers, no tech dump.</h1><p>Learn what SanketSetu does and how the sign-to-speech flow works.</p></section><section className="faq-list">{faqs.map(([question, answer]) => <article key={question}><h2>{question}</h2><p>{answer}</p></article>)}</section><Link href="/" className="primary-button help-cta">Back to translator <span>↗</span></Link></main> }

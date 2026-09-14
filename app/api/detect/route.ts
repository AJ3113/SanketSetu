import { NextResponse } from 'next/server'
export async function GET() { return NextResponse.json({ status: 'model_unconfigured', message: 'Connect the FastAPI service and trained MediaPipe classifier to enable live detection.', stack: ['React', 'FastAPI', 'MediaPipe', 'gTTS/STT', 'SQLite'] }) }

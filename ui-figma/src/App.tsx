import { useState, useEffect, useRef } from 'react'

/* ─────────────────────────────────────────────────────────────
   TYPES
───────────────────────────────────────────────────────────── */
type Screen     = 'home' | 'sentence' | 'story'
type AudioState = 'bear-speaking' | 'child-speaking' | 'feedback'

/* ─────────────────────────────────────────────────────────────
   3D TEDDY BEAR — SVG with gradients, shading, depth
───────────────────────────────────────────────────────────── */
function TeddyBear({ size = 200 }: { size?: number }) {
  const id = `bear-${size}` // stable id per size
  return (
    <svg
      width={size}
      height={size * 1.08}
      viewBox="0 0 220 238"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ filter: 'drop-shadow(0 12px 24px rgba(180,120,60,0.28))' }}
    >
      <defs>
        {/* Main fur gradient */}
        <radialGradient id={`fur-${id}`} cx="42%" cy="35%" r="58%">
          <stop offset="0%"   stopColor="#F5C98A" />
          <stop offset="55%"  stopColor="#E8A85A" />
          <stop offset="100%" stopColor="#C47A30" />
        </radialGradient>
        {/* Lighter tummy / inner ear */}
        <radialGradient id={`tummy-${id}`} cx="50%" cy="40%" r="55%">
          <stop offset="0%"   stopColor="#FDE8C2" />
          <stop offset="100%" stopColor="#F5C98A" />
        </radialGradient>
        {/* Nose sheen */}
        <radialGradient id={`nose-${id}`} cx="35%" cy="30%" r="60%">
          <stop offset="0%"   stopColor="#6B3A2A" />
          <stop offset="100%" stopColor="#3D1F10" />
        </radialGradient>
        {/* Eye gradient */}
        <radialGradient id={`eye-${id}`} cx="35%" cy="28%" r="60%">
          <stop offset="0%"   stopColor="#5C3317" />
          <stop offset="100%" stopColor="#1A0A00" />
        </radialGradient>
        {/* Head top-light */}
        <radialGradient id={`head-hl-${id}`} cx="40%" cy="25%" r="50%">
          <stop offset="0%"   stopColor="rgba(255,255,255,0.38)" />
          <stop offset="100%" stopColor="rgba(255,255,255,0)" />
        </radialGradient>
        {/* Body shadow */}
        <radialGradient id={`body-shadow-${id}`} cx="50%" cy="80%" r="55%">
          <stop offset="0%"   stopColor="rgba(0,0,0,0)" />
          <stop offset="100%" stopColor="rgba(0,0,0,0.18)" />
        </radialGradient>
        {/* Drop shadow ellipse */}
        <radialGradient id={`shadow-${id}`} cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stopColor="rgba(140,80,20,0.35)" />
          <stop offset="100%" stopColor="rgba(140,80,20,0)" />
        </radialGradient>
        {/* Scarf gradient */}
        <linearGradient id={`scarf-${id}`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%"   stopColor="#FF7BAC" />
          <stop offset="100%" stopColor="#E8507A" />
        </linearGradient>
        {/* Cheek blush */}
        <radialGradient id={`blush-${id}`} cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stopColor="#FFB3C6" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#FF8FAB" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* ── Ground shadow ────────────────────────────── */}
      <ellipse cx="110" cy="234" rx="58" ry="9" fill={`url(#shadow-${id})`} />

      {/* ── Left ear ─────────────────────────────────── */}
      <circle cx="46"  cy="60" r="28" fill={`url(#fur-${id})`} />
      <circle cx="46"  cy="60" r="17" fill={`url(#tummy-${id})`} />
      <circle cx="44"  cy="58" r="9"  fill="#F5D5A8" opacity="0.7" />

      {/* ── Right ear ────────────────────────────────── */}
      <circle cx="174" cy="60" r="28" fill={`url(#fur-${id})`} />
      <circle cx="174" cy="60" r="17" fill={`url(#tummy-${id})`} />
      <circle cx="172" cy="58" r="9"  fill="#F5D5A8" opacity="0.7" />

      {/* ── Head ─────────────────────────────────────── */}
      <circle cx="110" cy="98" r="72" fill={`url(#fur-${id})`} />
      {/* Ambient occlusion under ears */}
      <circle cx="52"  cy="90" r="22" fill="rgba(180,100,30,0.12)" />
      <circle cx="168" cy="90" r="22" fill="rgba(180,100,30,0.12)" />
      {/* Head highlight dome */}
      <ellipse cx="96" cy="72" rx="38" ry="26" fill={`url(#head-hl-${id})`} transform="rotate(-18,96,72)" />

      {/* ── Snout ────────────────────────────────────── */}
      <ellipse cx="110" cy="118" rx="30" ry="22" fill={`url(#tummy-${id})`} />
      <ellipse cx="106" cy="113" rx="14" ry="9"  fill="rgba(255,255,255,0.28)" />

      {/* ── Eyes ─────────────────────────────────────── */}
      {/* Left eye */}
      <circle cx="84"  cy="92" r="13" fill={`url(#eye-${id})`} />
      <circle cx="84"  cy="92" r="10" fill="#1A0A00" />
      <circle cx="79"  cy="87" r="4"  fill="white" opacity="0.9" />
      <circle cx="81"  cy="89" r="2"  fill="white" opacity="0.6" />
      {/* Right eye */}
      <circle cx="136" cy="92" r="13" fill={`url(#eye-${id})`} />
      <circle cx="136" cy="92" r="10" fill="#1A0A00" />
      <circle cx="131" cy="87" r="4"  fill="white" opacity="0.9" />
      <circle cx="133" cy="89" r="2"  fill="white" opacity="0.6" />

      {/* ── Nose ─────────────────────────────────────── */}
      <ellipse cx="110" cy="111" rx="12" ry="8.5" fill={`url(#nose-${id})`} />
      <ellipse cx="107" cy="108" rx="4"  ry="2.5" fill="rgba(255,255,255,0.35)" />

      {/* ── Mouth ────────────────────────────────────── */}
      <path d="M 96 122 Q 110 138 124 122" stroke="#3D1F10" strokeWidth="3.5" strokeLinecap="round" fill="none" />
      <path d="M 110 122 L 110 126" stroke="#3D1F10" strokeWidth="3" strokeLinecap="round" />

      {/* ── Cheek blush ──────────────────────────────── */}
      <ellipse cx="70"  cy="112" rx="18" ry="12" fill={`url(#blush-${id})`} />
      <ellipse cx="150" cy="112" rx="18" ry="12" fill={`url(#blush-${id})`} />

      {/* ── Body ─────────────────────────────────────── */}
      <ellipse cx="110" cy="192" rx="58" ry="46" fill={`url(#fur-${id})`} />
      <ellipse cx="110" cy="192" rx="58" ry="46" fill={`url(#body-shadow-${id})`} />
      {/* Tummy */}
      <ellipse cx="110" cy="192" rx="36" ry="30" fill={`url(#tummy-${id})`} />
      <ellipse cx="104" cy="183" rx="16" ry="10" fill="rgba(255,255,255,0.28)" />

      {/* ── Scarf ────────────────────────────────────── */}
      <path d="M 60 160 Q 110 152 160 160 Q 110 172 60 160 Z" fill={`url(#scarf-${id})`} />
      {/* Scarf knot / tail */}
      <path d="M 110 160 Q 104 174 100 184 Q 106 186 114 184 Q 118 174 110 160 Z" fill="#E8507A" />
      {/* Scarf highlight */}
      <path d="M 70 158 Q 110 152 148 158" stroke="rgba(255,255,255,0.3)" strokeWidth="3" fill="none" strokeLinecap="round" />

      {/* ── Left arm ─────────────────────────────────── */}
      <ellipse cx="58"  cy="190" rx="20" ry="34" fill={`url(#fur-${id})`} transform="rotate(-25,58,190)" />
      <ellipse cx="44"  cy="206" rx="15" ry="11" fill="#D4956A" />

      {/* ── Right arm ────────────────────────────────── */}
      <ellipse cx="162" cy="190" rx="20" ry="34" fill={`url(#fur-${id})`} transform="rotate(25,162,190)" />
      <ellipse cx="176" cy="206" rx="15" ry="11" fill="#D4956A" />

      {/* ── Star badge on tummy ──────────────────────── */}
      <text x="110" y="198" textAnchor="middle" fontSize="22" opacity="0.85">⭐</text>
    </svg>
  )
}

/* ─────────────────────────────────────────────────────────────
   SOUND WAVE
───────────────────────────────────────────────────────────── */
function SoundWave({ active, color = '#9333ea' }: { active: boolean; color?: string }) {
  const heights = [14, 22, 32, 40, 48, 52, 48, 40, 32, 22, 14]
  return (
    <div className="flex items-center gap-[3px]" style={{ height: 52 }}>
      {heights.map((h, i) => (
        <div
          key={i}
          className={active ? 'wave-bar' : ''}
          style={{
            width: 6,
            height: active ? h : 6,
            borderRadius: 99,
            background: color,
            opacity: active ? 1 : 0.28,
            transition: 'height 0.3s ease, opacity 0.3s ease',
            transformOrigin: 'center',
          }}
        />
      ))}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   SPEECH BUBBLE
───────────────────────────────────────────────────────────── */
function Bubble({
  children, color = 'white', tail = 'bottom-left', className = '',
}: {
  children: React.ReactNode
  color?: string
  tail?: 'bottom-left' | 'bottom-center' | 'left'
  className?: string
}) {
  const tailStyle: React.CSSProperties =
    tail === 'bottom-left'   ? { bottom: -12, left: 24,
                                  borderLeft: '10px solid transparent',
                                  borderRight: '10px solid transparent',
                                  borderTop: `14px solid ${color}` }
    : tail === 'bottom-center' ? { bottom: -12, left: '50%', transform: 'translateX(-50%)',
                                    borderLeft: '10px solid transparent',
                                    borderRight: '10px solid transparent',
                                    borderTop: `14px solid ${color}` }
    :                           { left: -11, top: 16,
                                   borderTop: '9px solid transparent',
                                   borderBottom: '9px solid transparent',
                                   borderRight: `13px solid ${color}` }

  return (
    <div className={`relative bubble-breathe ${className}`}>
      <div
        className="rounded-3xl px-5 py-3.5"
        style={{
          background: color,
          boxShadow: '0 4px 18px rgba(0,0,0,0.1), inset 0 1.5px 0 rgba(255,255,255,0.8)',
        }}
      >
        {children}
      </div>
      <div className="absolute w-0 h-0" style={tailStyle} />
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   SCREEN 1: HOME
───────────────────────────────────────────────────────────── */
function HomeScreen({ onSelect }: { onSelect: (g: 'sentence' | 'story') => void }) {
  return (
    <div
      className="flex flex-col h-full scroll-hide overflow-y-auto"
      style={{
        background: 'linear-gradient(170deg, #EDE4FF 0%, #F9E4FF 38%, #FFE8DC 72%, #FFF3D4 100%)',
      }}
    >
      {/* ── Header ────────────────────────────────── */}
      <div className="flex items-center justify-between px-6 pt-5 pb-1 flex-shrink-0">
        <div>
          <p className="font-display text-purple-700 leading-none" style={{ fontSize: 32, fontWeight: 700 }}>
            Hi there! 👋
          </p>
          <p className="text-purple-400 font-bold text-sm mt-0.5">Ready to learn?</p>
        </div>
        {/* Stars / progress icon */}
        <div
          className="w-14 h-14 rounded-2xl flex flex-col items-center justify-center card-lift"
          style={{ background: 'linear-gradient(145deg, #FFE168, #FFB347)' }}
        >
          <span style={{ fontSize: 22 }}>⭐</span>
          <span className="text-yellow-800 font-black text-[10px] leading-none mt-0.5">12 pts</span>
        </div>
      </div>

      {/* ── Bear + Bubble ─────────────────────────── */}
      <div className="flex items-end justify-center gap-3 px-6 pt-1 pb-0 flex-shrink-0">
        <div className="bear-float" style={{ marginBottom: -8 }}>
          <TeddyBear size={176} />
        </div>
        <div className="mb-16 flex-shrink-0" style={{ maxWidth: 140 }}>
          <Bubble color="white" tail="left">
            <p className="font-display text-purple-700 leading-snug" style={{ fontSize: 17, fontWeight: 600 }}>
              What shall we<br />play today? 💭
            </p>
          </Bubble>
        </div>
      </div>

      {/* ── Divider label ───────────────────────────── */}
      <div className="px-6 pt-2 pb-3 flex-shrink-0">
        <p
          className="font-display text-center text-purple-700"
          style={{ fontSize: 26, fontWeight: 700 }}
        >
          Choose your adventure! ✨
        </p>
      </div>

      {/* ── Game cards ─────────────────────────────── */}
      <div className="px-5 pb-8 flex flex-col gap-4 flex-shrink-0">

        {/* Card 1 — Finish the Sentence */}
        <button
          onClick={() => onSelect('sentence')}
          className="w-full rounded-[28px] overflow-hidden card-lift text-left shine-btn"
          style={{ background: 'linear-gradient(140deg, #FFD6A0 0%, #FFAA5C 100%)' }}
        >
          {/* Inner top stripe for depth */}
          <div style={{ height: 3, background: 'rgba(255,255,255,0.5)', borderRadius: '28px 28px 0 0' }} />
          <div className="flex items-stretch gap-0">
            {/* Illustration panel */}
            <div
              className="flex items-center justify-center flex-shrink-0"
              style={{
                width: 110,
                background: 'linear-gradient(160deg, #FFE8C0 0%, #FFCA80 100%)',
                borderRadius: '0 0 0 26px',
                padding: '12px 8px 12px 12px',
              }}
            >
              <div style={{ transform: 'scale(0.48)', transformOrigin: 'center', width: 88, height: 96, display: 'flex', alignItems: 'center', justifyContent: 'center', marginLeft: -16, marginRight: -16 }}>
                <TeddyBear size={180} />
              </div>
            </div>
            {/* Text panel */}
            <div className="flex-1 py-5 pl-4 pr-4">
              <p className="font-display text-orange-900 leading-tight mb-1" style={{ fontSize: 22, fontWeight: 700 }}>
                Finish the<br />Sentence
              </p>
              <p className="text-orange-800 font-bold leading-snug mb-4" style={{ fontSize: 13 }}>
                Listen &amp; complete! 🗣️
              </p>
              <div
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-2xl btn-press font-display text-white"
                style={{
                  fontSize: 16,
                  fontWeight: 700,
                  background: 'linear-gradient(145deg, #FF8C42, #E55A12)',
                }}
              >
                <span style={{ fontSize: 18 }}>▶</span> Play!
              </div>
            </div>
          </div>
        </button>

        {/* Card 2 — Story Adventure */}
        <button
          onClick={() => onSelect('story')}
          className="w-full rounded-[28px] overflow-hidden card-lift text-left shine-btn"
          style={{ background: 'linear-gradient(140deg, #C0F0FF 0%, #5DD8F5 100%)' }}
        >
          <div style={{ height: 3, background: 'rgba(255,255,255,0.5)', borderRadius: '28px 28px 0 0' }} />
          <div className="flex items-stretch gap-0">
            {/* Illustration */}
            <div
              className="flex items-center justify-center flex-shrink-0"
              style={{
                width: 110,
                background: 'linear-gradient(160deg, #DEFFFD 0%, #A0EBF8 100%)',
                borderRadius: '0 0 0 26px',
                padding: '12px 8px',
              }}
            >
              <span style={{ fontSize: 64, lineHeight: 1, display: 'block', filter: 'drop-shadow(0 4px 8px rgba(0,100,160,0.22))' }}>📖</span>
            </div>
            {/* Text */}
            <div className="flex-1 py-5 pl-4 pr-4">
              <p className="font-display text-cyan-900 leading-tight mb-1" style={{ fontSize: 22, fontWeight: 700 }}>
                Story<br />Adventure
              </p>
              <p className="text-cyan-800 font-bold leading-snug mb-4" style={{ fontSize: 13 }}>
                Listen &amp; learn! ✨
              </p>
              <div
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-2xl btn-press font-display text-white"
                style={{
                  fontSize: 16,
                  fontWeight: 700,
                  background: 'linear-gradient(145deg, #22C5E8, #0899B5)',
                }}
              >
                <span style={{ fontSize: 18 }}>▶</span> Play!
              </div>
            </div>
          </div>
        </button>
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   SCREEN 2: AUDIO GAME
───────────────────────────────────────────────────────────── */
const SENTENCE_STEPS = [
  {
    state: 'bear-speaking' as AudioState,
    bearLabel: 'Teddy says:',
    bearPrompt: 'Can you finish this sentence? 🎵',
    ttsCard: 'I like to eat ___.',
    sttCard: null,
    feedbackText: null,
  },
  {
    state: 'child-speaking' as AudioState,
    bearLabel: 'I\'m listening… 👂',
    bearPrompt: 'Your turn! Tell me what you eat!',
    ttsCard: 'I like to eat ___.',
    sttCard: 'I like to eat… apples!',
    feedbackText: null,
  },
  {
    state: 'feedback' as AudioState,
    bearLabel: 'Wonderful! 🎉',
    bearPrompt: 'You\'re so smart!',
    ttsCard: 'I like to eat ___.',
    sttCard: 'I like to eat apples.',
    feedbackText: 'Great job! ⭐',
  },
]

const STORY_STEPS = [
  {
    state: 'bear-speaking' as AudioState,
    bearLabel: 'Story time!',
    bearPrompt: 'Let\'s go on a story adventure! 📖',
    ttsCard: 'Once upon a time, a little bear found a magical ___.',
    sttCard: null,
    feedbackText: null,
  },
  {
    state: 'child-speaking' as AudioState,
    bearLabel: 'Tell me! 👂',
    bearPrompt: 'What did the bear find?',
    ttsCard: 'Once upon a time, a little bear found a magical ___.',
    sttCard: 'The bear found a magical apple!',
    feedbackText: null,
  },
  {
    state: 'feedback' as AudioState,
    bearLabel: 'Amazing story! 🌟',
    bearPrompt: 'You\'re a great storyteller!',
    ttsCard: 'Once upon a time…',
    sttCard: 'The bear found a magical apple!',
    feedbackText: 'Brilliant! ⭐⭐',
  },
]

/* ── Mic button ─────────────────────────────────────────────── */
function MicButton({ state, onTap }: { state: AudioState; onTap: () => void }) {
  const isListening = state === 'child-speaking'
  const isFeedback  = state === 'feedback'

  return (
    <button
      onClick={onTap}
      className={`relative rounded-full flex items-center justify-center ${isListening ? 'mic-active' : isFeedback ? '' : 'mic-idle'} btn-press`}
      style={{
        width: 88, height: 88,
        background: isListening
          ? 'linear-gradient(145deg, #FB7185, #E11D48)'
          : isFeedback
          ? 'linear-gradient(145deg, #34D399, #059669)'
          : 'linear-gradient(145deg, #C084FC, #7C3AED)',
        boxShadow: isListening
          ? '0 5px 0 #9B1229, 0 8px 20px rgba(225,29,72,0.4)'
          : isFeedback
          ? '0 5px 0 #047857, 0 8px 20px rgba(5,150,105,0.35)'
          : '0 5px 0 #5B21B6, 0 8px 20px rgba(124,58,237,0.38)',
      }}
    >
      <span style={{ fontSize: 38 }}>
        {isListening ? '🎙️' : isFeedback ? '🎉' : '🎤'}
      </span>
    </button>
  )
}

/* ── Feedback confetti dots ────────────────────────────────────── */
function Confetti() {
  const dots = [
    { color: '#FF6B6B', x: '15%', delay: '0s' },
    { color: '#FFD93D', x: '30%', delay: '0.1s' },
    { color: '#6BCB77', x: '50%', delay: '0.18s' },
    { color: '#4D96FF', x: '68%', delay: '0.08s' },
    { color: '#FF6FC8', x: '83%', delay: '0.14s' },
  ]
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden" style={{ borderRadius: 28 }}>
      {dots.map((d, i) => (
        <div
          key={i}
          className="absolute w-3 h-3 rounded-full"
          style={{
            background: d.color,
            left: d.x,
            top: '10%',
            animationDelay: d.delay,
            animation: 'confetti-fall 0.9s ease-out forwards',
          }}
        />
      ))}
    </div>
  )
}

function AudioGameScreen({ game, onBack }: { game: 'sentence' | 'story'; onBack: () => void }) {
  const steps   = game === 'sentence' ? SENTENCE_STEPS : STORY_STEPS
  const [step, setStep] = useState(0)
  const [showFx, setShowFx] = useState(false)

  const cur     = steps[step]
  const isLast  = step === steps.length - 1
  const isFb    = cur.state === 'feedback'
  const isBear  = cur.state === 'bear-speaking'
  const isChild = cur.state === 'child-speaking'

  useEffect(() => {
    if (isFb) { setTimeout(() => setShowFx(true), 80) }
    else       { setShowFx(false) }
  }, [step])

  function advance() {
    if (!isLast) setStep(s => s + 1)
    else         { setStep(0); setShowFx(false) }
  }

  const accent = game === 'sentence' ? '#F97316' : '#06B6D4'
  const bg     = game === 'sentence'
    ? 'linear-gradient(168deg, #FFF0E8 0%, #FFE4F8 50%, #F0E8FF 100%)'
    : 'linear-gradient(168deg, #E0F9FF 0%, #E8F5FF 50%, #F0E8FF 100%)'

  return (
    <div className="flex flex-col h-full" style={{ background: bg }}>

      {/* ── Top bar ──────────────────────────────── */}
      <div className="flex items-center justify-between px-5 pt-5 pb-2 flex-shrink-0">
        <button
          onClick={onBack}
          className="w-10 h-10 rounded-2xl flex items-center justify-center card-lift font-display text-purple-600"
          style={{ background: 'rgba(255,255,255,0.85)', fontSize: 20 }}
        >
          ←
        </button>
        <p className="font-display text-purple-700" style={{ fontSize: 17, fontWeight: 700 }}>
          {game === 'sentence' ? 'Finish the Sentence' : 'Story Adventure'}
        </p>
        {/* Progress dots */}
        <div className="flex gap-1.5 items-center">
          {steps.map((_, i) => (
            <div
              key={i}
              className="rounded-full transition-all duration-300"
              style={{
                width: i === step ? 22 : 8,
                height: 8,
                background: i === step ? accent : 'rgba(167,139,250,0.35)',
              }}
            />
          ))}
        </div>
      </div>

      {/* ── Bear row ─────────────────────────────── */}
      <div className="flex items-end gap-3 px-5 pb-1 flex-shrink-0">
        <div className={isBear ? 'bear-float-sm' : ''}>
          <TeddyBear size={96} />
        </div>
        <div className="flex-1 pb-4">
          <div className="bubble-breathe">
            <Bubble color="white" tail="left">
              <p className="text-purple-600 font-black leading-tight" style={{ fontSize: 12 }}>
                {cur.bearLabel}
              </p>
              <p className="font-display text-purple-800 leading-snug mt-0.5" style={{ fontSize: 15, fontWeight: 600 }}>
                {cur.bearPrompt}
              </p>
            </Bubble>
          </div>
        </div>
      </div>

      {/* ── Main interaction area ───────────────── */}
      <div className="flex-1 flex flex-col gap-3 px-5 min-h-0">

        {/* TTS card — bear's prompt */}
        <div
          className="w-full rounded-[24px] p-5 card-lift slide-up relative overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, #F3E8FF 0%, #E9D5FF 100%)',
            borderTop: '2px solid rgba(255,255,255,0.75)',
          }}
        >
          <div className="flex items-center gap-2 mb-2">
            <span style={{ fontSize: 18 }}>🐻</span>
            <span
              className="font-black uppercase tracking-widest text-purple-500"
              style={{ fontSize: 10 }}
            >
              Teddy says
            </span>
          </div>
          <p className="font-display text-purple-900 leading-snug" style={{ fontSize: 24, fontWeight: 700 }}>
            {cur.ttsCard}
          </p>
          <div className="mt-3 flex items-center gap-3">
            <SoundWave active={isBear} color="#9333EA" />
            {isBear && (
              <span className="text-purple-400 font-bold text-xs animate-pulse">speaking…</span>
            )}
          </div>
        </div>

        {/* Feedback card */}
        {isFb && showFx && (
          <div
            className="w-full rounded-[24px] p-4 card-lift star-pop relative overflow-hidden"
            style={{ background: 'linear-gradient(135deg, #FEFCE8 0%, #FEF08A 100%)' }}
          >
            <Confetti />
            <p className="font-display text-yellow-800 text-center" style={{ fontSize: 32, fontWeight: 700 }}>
              {cur.feedbackText}
            </p>
            <p className="text-yellow-700 font-black text-center text-sm mt-1">
              You're amazing! 🌟
            </p>
          </div>
        )}

        {/* STT card — child's response */}
        {cur.sttCard && (
          <div
            className="w-full rounded-[24px] p-5 card-lift slide-up relative overflow-hidden"
            style={{
              background: isChild
                ? 'linear-gradient(135deg, #DCFCE7 0%, #A7F3D0 100%)'
                : 'linear-gradient(135deg, #F0FDF4 0%, #D1FAE5 100%)',
              borderTop: '2px solid rgba(255,255,255,0.75)',
            }}
          >
            <div className="flex items-center gap-2 mb-2">
              <span style={{ fontSize: 18 }}>🧒</span>
              <span
                className="font-black uppercase tracking-widest text-green-600"
                style={{ fontSize: 10 }}
              >
                {isChild ? 'Listening…' : 'You said'}
              </span>
            </div>
            <p className="font-display text-green-900 leading-snug" style={{ fontSize: 22, fontWeight: 700 }}>
              {cur.sttCard}
            </p>
            <div className="mt-3 flex items-center gap-3">
              <SoundWave active={isChild} color="#16A34A" />
              {isChild && (
                <span className="text-green-500 font-bold text-xs animate-pulse">recording…</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Mic + action ───────────────────────── */}
      <div className="flex flex-col items-center gap-2 py-5 flex-shrink-0">
        <p className="font-bold text-purple-400" style={{ fontSize: 13 }}>
          {isBear  ? '🔊 Teddy is speaking…'
          : isChild ? 'Tap the mic when you\'re done!'
          :           isLast ? 'Tap to play again! 🔄' : 'Great! Tap next ▶'}
        </p>
        <MicButton state={cur.state} onTap={advance} />
        {!isChild && (
          <button
            onClick={advance}
            className="mt-1 px-8 py-3 rounded-2xl btn-press font-display text-white shine-btn"
            style={{
              fontSize: 18, fontWeight: 700,
              background: `linear-gradient(145deg, ${accent}, #7C3AED)`,
              boxShadow: `0 5px 0 #5B21B6, 0 8px 20px rgba(124,58,237,0.35)`,
            }}
          >
            {isFb && isLast ? 'Play again! 🎵' : 'Next ▶'}
          </button>
        )}
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────
   ROOT
───────────────────────────────────────────────────────────── */
export default function App() {
  const [screen, setScreen] = useState<Screen>('home')

  return (
    <div
      className="min-h-screen flex items-center justify-center"
      style={{
        background: 'radial-gradient(ellipse at 30% 40%, #C4B5FD 0%, #7C3AED 60%, #4C1D95 100%)',
        padding: '24px 16px',
      }}
    >
      {/* Phone frame */}
      <div
        style={{
          width: 390,
          height: 844,
          borderRadius: 52,
          position: 'relative',
          background: '#1A0A2E',
          boxShadow:
            '0 0 0 2px #3D1F60, 0 0 0 10px #1A0A2E, inset 0 2px 0 rgba(255,255,255,0.08), 0 40px 100px rgba(0,0,0,0.6), 0 0 60px rgba(124,58,237,0.25)',
          flexShrink: 0,
        }}
      >
        {/* Screen glass */}
        <div
          style={{
            position: 'absolute',
            inset: 6,
            borderRadius: 46,
            overflow: 'hidden',
            background: '#fff',
          }}
        >
          {/* Status bar */}
          <div
            style={{
              height: 46,
              background: 'rgba(255,255,255,0.7)',
              backdropFilter: 'blur(12px)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 24px',
              flexShrink: 0,
              position: 'relative',
              zIndex: 10,
            }}
          >
            <span style={{ fontFamily: 'Nunito', fontWeight: 900, fontSize: 13, color: '#4C1D95' }}>9:41</span>
            {/* Dynamic island */}
            <div style={{ width: 120, height: 32, borderRadius: 99, background: '#0A0A0A' }} />
            <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
              <div style={{ width: 16, height: 10, borderRadius: 2, border: '1.5px solid #4C1D95', position: 'relative', overflow: 'hidden' }}>
                <div style={{ position: 'absolute', inset: '2px 3px', background: '#4C1D95', borderRadius: 1 }} />
              </div>
            </div>
          </div>

          {/* Content area */}
          <div style={{ position: 'absolute', top: 46, left: 0, right: 0, bottom: 0 }}>
            {screen === 'home' && <HomeScreen onSelect={g => setScreen(g)} />}
            {(screen === 'sentence' || screen === 'story') && (
              <AudioGameScreen game={screen} onBack={() => setScreen('home')} />
            )}
          </div>
        </div>

        {/* Phone side buttons */}
        <div style={{ position: 'absolute', left: -3, top: 120, width: 3, height: 32, borderRadius: '2px 0 0 2px', background: '#3D1F60' }} />
        <div style={{ position: 'absolute', left: -3, top: 164, width: 3, height: 56, borderRadius: '2px 0 0 2px', background: '#3D1F60' }} />
        <div style={{ position: 'absolute', left: -3, top: 230, width: 3, height: 56, borderRadius: '2px 0 0 2px', background: '#3D1F60' }} />
        <div style={{ position: 'absolute', right: -3, top: 180, width: 3, height: 72, borderRadius: '0 2px 2px 0', background: '#3D1F60' }} />
      </div>
    </div>
  )
}

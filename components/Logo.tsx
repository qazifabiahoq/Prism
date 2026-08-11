export function Logo({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <svg width="30" height="30" viewBox="0 0 64 64" fill="none" className="shrink-0">
        <defs>
          <linearGradient id="logoGrad" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#7C7FF2" />
            <stop offset="100%" stopColor="#4143C7" />
          </linearGradient>
        </defs>
        <rect width="64" height="64" rx="16" fill="#0B0F19" />
        <path d="M32 10 L50 24 L32 54 L14 24 Z" fill="url(#logoGrad)" fillOpacity="0.92" />
        <path d="M32 10 L50 24 L32 34 L14 24 Z" fill="#ffffff" fillOpacity="0.28" />
      </svg>
      <span className="font-display text-[1.35rem] font-extrabold tracking-tight text-ink-950">
        Prism
      </span>
    </div>
  );
}

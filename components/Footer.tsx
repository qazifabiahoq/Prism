import { Logo } from "./Logo";

export function Footer() {
  return (
    <footer className="border-t border-white/10 bg-navy-950 py-12">
      <div className="container-shell flex flex-col items-center gap-6 text-center sm:flex-row sm:justify-between sm:text-left">
        <div>
          <Logo light />
          <p className="mt-2 max-w-xs text-sm text-white/40">
            Turning transaction data into financial intelligence through
            machine learning — built with production ML, not prompts.
          </p>
        </div>
        <div className="flex flex-col items-center gap-1 text-xs text-white/35 sm:items-end">
          <p>Secure · Private · Free</p>
          <p>No data retention · No bank linking required</p>
          <p className="mt-2 text-white/25">© {new Date().getFullYear()} Prism</p>
        </div>
      </div>
    </footer>
  );
}

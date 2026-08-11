import { Logo } from "./Logo";

export function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-surface-50 py-12">
      <div className="container-shell flex flex-col items-center gap-6 text-center sm:flex-row sm:justify-between sm:text-left">
        <div>
          <Logo />
          <p className="mt-2 max-w-xs text-sm text-slate-500">
            Turning transaction data into financial intelligence through machine learning. Built with production ML, not prompts.
          </p>
        </div>
        <div className="flex flex-col items-center gap-1 text-xs text-slate-400 sm:items-end">
          <p>Secure · Private · Free</p>
          <p>No data retention · No bank linking required</p>
          <p className="mt-2 text-slate-300">© {new Date().getFullYear()} Prism</p>
        </div>
      </div>
    </footer>
  );
}

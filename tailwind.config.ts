import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#020617",
          900: "#050B18",
          800: "#0A1428",
          700: "#0F1E38",
        },
        navy: {
          950: "#040A17",
          900: "#071433",
          800: "#0B1E4A",
          700: "#0F2A63",
          600: "#14367D",
          500: "#1B4796",
        },
        accent: {
          400: "#5EEAD4",
          500: "#2DD4BF",
          600: "#14B8A6",
        },
        gold: {
          400: "#F5D68A",
          500: "#E8C05F",
        },
        surface: {
          50: "#F7F9FC",
          100: "#EEF2F8",
          200: "#E2E8F3",
        },
      },
      fontFamily: {
        display: ["var(--font-display)", "sans-serif"],
        sans: ["var(--font-sans)", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(10, 20, 40, 0.04), 0 8px 24px -8px rgba(10, 20, 40, 0.10)",
        "card-hover": "0 4px 12px rgba(10, 20, 40, 0.06), 0 16px 40px -12px rgba(10, 20, 40, 0.18)",
        glow: "0 0 0 1px rgba(94, 234, 212, 0.15), 0 8px 32px -8px rgba(45, 212, 191, 0.25)",
      },
      backgroundImage: {
        "grid-pattern":
          "linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px)",
        "radial-fade": "radial-gradient(circle, rgba(94,234,212,0.16) 0%, transparent 70%)",
      },
      animation: {
        "fade-up": "fadeUp 0.6s ease-out both",
        "fade-in": "fadeIn 0.5s ease-out both",
        float: "float 6s ease-in-out infinite",
        shimmer: "shimmer 2.2s linear infinite",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-1000px 0" },
          "100%": { backgroundPosition: "1000px 0" },
        },
      },
    },
  },
  plugins: [],
};
export default config;

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
          950: "#0B0F19",
          900: "#131826",
          800: "#1E2536",
          700: "#333B4F",
          600: "#4B5468",
          500: "#646D82",
          400: "#8891A3",
        },
        accent: {
          50: "#EEF0FF",
          400: "#7C7FF2",
          500: "#5457E5",
          600: "#4143C7",
        },
        surface: {
          50: "#FAFBFC",
          100: "#F4F5F9",
          200: "#ECEEF4",
        },
      },
      fontFamily: {
        display: ["var(--font-display)", "sans-serif"],
        sans: ["var(--font-sans)", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(19, 24, 38, 0.04), 0 8px 24px -8px rgba(19, 24, 38, 0.08)",
        "card-hover": "0 4px 12px rgba(19, 24, 38, 0.05), 0 16px 40px -12px rgba(19, 24, 38, 0.14)",
        glow: "0 0 0 1px rgba(84, 87, 229, 0.12), 0 8px 32px -8px rgba(84, 87, 229, 0.28)",
      },
      backgroundImage: {
        "grid-pattern":
          "linear-gradient(rgba(19,24,38,0.045) 1px, transparent 1px), linear-gradient(90deg, rgba(19,24,38,0.045) 1px, transparent 1px)",
        "radial-fade": "radial-gradient(circle, rgba(84,87,229,0.12) 0%, transparent 70%)",
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

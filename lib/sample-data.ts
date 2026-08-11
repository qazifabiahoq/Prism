// Deterministic demo dataset so visitors (and recruiters) can try Prism instantly, no file required.
function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

const MERCHANTS: Array<{ desc: string; base: number; jitter: number }> = [
  { desc: "Whole Foods Market", base: 62, jitter: 30 },
  { desc: "Starbucks Coffee", base: 6, jitter: 4 },
  { desc: "Uber Trip", base: 18, jitter: 14 },
  { desc: "Shell Gas Station", base: 45, jitter: 15 },
  { desc: "Amazon.com", base: 55, jitter: 60 },
  { desc: "Netflix Subscription", base: 15.49, jitter: 0 },
  { desc: "Spotify Premium", base: 11.99, jitter: 0 },
  { desc: "Chipotle Mexican Grill", base: 13, jitter: 6 },
  { desc: "Con Edison Utility", base: 120, jitter: 35 },
  { desc: "Equinox Gym Membership", base: 180, jitter: 0 },
  { desc: "CVS Pharmacy", base: 22, jitter: 18 },
  { desc: "AMC Theatres", base: 32, jitter: 10 },
  { desc: "Trader Joe's", base: 48, jitter: 22 },
  { desc: "Verizon Wireless", base: 95, jitter: 5 },
  { desc: "Rent Payment - Parkside Apts", base: 1850, jitter: 0 },
];

export function generateSampleTransactions(days = 120) {
  const rng = seededRandom(42);
  const rows: Array<{ Date: string; Amount: number; Description: string }> = [];
  const start = new Date();
  start.setDate(start.getDate() - days);

  for (let i = 0; i < days; i++) {
    const date = new Date(start);
    date.setDate(start.getDate() + i);
    const isWeekend = date.getDay() === 0 || date.getDay() === 6;
    const txCount = isWeekend ? 2 + Math.floor(rng() * 3) : 1 + Math.floor(rng() * 2);

    for (let t = 0; t < txCount; t++) {
      const merchant = MERCHANTS[Math.floor(rng() * MERCHANTS.length)];
      const amount = Math.max(1.5, merchant.base + (rng() - 0.5) * 2 * merchant.jitter);
      rows.push({
        Date: date.toISOString().slice(0, 10),
        Amount: Math.round(amount * 100) / 100,
        Description: merchant.desc,
      });
    }
  }

  // Inject a few anomalies for the fraud-detection story
  const anomalyDates = [20, 55, 88];
  anomalyDates.forEach((idx, i) => {
    const date = new Date(start);
    date.setDate(start.getDate() + idx);
    rows.push({
      Date: date.toISOString().slice(0, 10),
      Amount: [890, 1240, 675][i],
      Description: ["Unrecognized Electronics Purchase", "International Wire Transfer", "Late Night Retail Charge"][i],
    });
  });

  rows.sort((a, b) => a.Date.localeCompare(b.Date));
  return rows;
}

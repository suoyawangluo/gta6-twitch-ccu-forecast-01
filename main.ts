// Hourly Twitch-viewer forecast for a game launch day.
//
// Idea: past launches' release-day curves, rescaled by their own first
// hour, are the best guide to a new launch's day. For every comparable
// launch in the case file we take its log viewer-ratio curve
// ln(viewers_h / viewers_1) for hours 2..24, then fit one small ridge
// regression per hour: the log ratio as a function of the launch's
// first-hour size, how its last pre-midnight hour compared with its
// first hour, and whether it had an early-access head start. The
// forecast's quantiles are the regression's prediction plus normal
// residual quantiles, smoothed lightly across neighbouring hours.
//
// Launches whose first hour was still dead (a single simultaneous
// unlock later in the day) are left out: their ratio curves explode
// and say nothing about a rolling-midnight launch like this one.
// Early-access launches stay in with their own flag, so they inform
// the fit without bending it. Everything is computed from the case
// file alone; the program never touches the network, never reads the
// clock, and is a pure function of its input, so two runs agree byte
// for byte.
//
// Fallbacks keep every case valid: with too few comparables the
// forecast falls back to the median ratio curve with a wide spread,
// and with none at all to persistence at the first hour's print.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const HOUR_MS = 3600_000;

/** Quantile levels the contest scores, in row order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Ridge penalty on the non-intercept coefficients. */
const RIDGE = 0.5;
/** Half-width of the smoothing window over neighbouring hours. */
const SMOOTH = 1;
/** Minimum comparable curves for the regression; fewer falls back. */
const MIN_ROWS = 6;
/** A comparable needs this many of the 24 release-day hours present. */
const MIN_DAY_HOURS = 20;
/** Below this first-hour share of the day's peak, the launch had not started at midnight. */
const LIVE_AT_MIDNIGHT = 0.05;
/** How far past the largest training launch the size feature may extrapolate, in log units. */
const SIZE_EXTRAPOLATION = 1.0;
/** How far past the training range the pre-midnight feature may extrapolate, in log units. */
const SHAPE_EXTRAPOLATION = 0.5;
/** Spread of the persistence fallback, in log units per square-root hour. */
const FALLBACK_SIGMA = 0.15;
/** Floor on the spread of the small-sample fallback, in log units. */
const SMALL_SAMPLE_SD = 0.3;

interface CaseFile {
  id?: string;
  day_start?: string;
  known?: unknown;
  history?: Array<{ at?: string; viewers?: number }>;
  history_hourly?: Array<{ from?: string; viewers?: number }>;
  comparables?: Comparable[];
}

interface Comparable {
  name?: string;
  day_start?: string;
  unlock?: string;
  platforms?: string;
  hourly?: Array<{ from?: string; viewers?: number }>;
}

interface TrainingRow {
  features: number[];
  logRatios: Array<number | null>;
  logSize: number;
}

/** Standard normal quantile (Acklam's approximation, ~1e-9 accurate). */
function normInv(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  const phigh = 1 - plow;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

const Z = LEVELS.map(normInv);

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const round2 = (x: number): number => Math.round(x * 100) / 100;
const safeLog = (x: number): number => Math.log(Math.max(1, x));

/** Solve A x = b by Gaussian elimination with partial pivoting; null when singular. */
function solveLinear(aIn: number[][], bIn: number[]): number[] | null {
  const k = bIn.length;
  const m = aIn.map((row, i) => [...row, bIn[i]]);
  for (let col = 0; col < k; col++) {
    let piv = col;
    for (let r = col + 1; r < k; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    if (Math.abs(m[piv][col]) < 1e-12) return null;
    if (piv !== col) {
      const tmp = m[col];
      m[col] = m[piv];
      m[piv] = tmp;
    }
    const pv = m[col][col];
    for (let r = 0; r < k; r++) {
      if (r === col || m[r][col] === 0) continue;
      const fac = m[r][col] / pv;
      for (let cc = col; cc <= k; cc++) m[r][cc] -= fac * m[col][cc];
    }
  }
  return m.map((row, i) => row[k] / m[i][i]);
}

/** Hour-1 print: the `known` row for hour 1, else the last pre-checkpoint level, else 1. */
function hourOnePrint(c: CaseFile): number {
  const known = c.known;
  if (Array.isArray(known)) {
    const row = (known.find((k) => (k as { hour?: number }).hour === 1) ?? known[0]) as { viewers?: number } | undefined;
    if (row && typeof row.viewers === 'number' && Number.isFinite(row.viewers) && row.viewers > 0) return row.viewers;
  } else if (known && typeof known === 'object' && typeof (known as { viewers?: number }).viewers === 'number') {
    const v = (known as { viewers: number }).viewers;
    if (Number.isFinite(v) && v > 0) return v;
  }
  const hh = c.history_hourly;
  if (Array.isArray(hh) && hh.length > 0) {
    for (let i = hh.length - 1; i >= 0; i--) {
      const v = hh[i].viewers;
      if (typeof v === 'number' && Number.isFinite(v) && v > 0) return v;
    }
  }
  const hist = c.history;
  if (Array.isArray(hist) && hist.length > 0) {
    for (let i = hist.length - 1; i >= 0; i--) {
      const v = hist[i].viewers;
      if (typeof v === 'number' && Number.isFinite(v) && v > 0) return v;
    }
  }
  return 1;
}

/**
 * The pre-midnight shape feature: ln(viewers in the hour before the
 * release day) - ln(hour-1 print). Positive when the world was already
 * watching before midnight (a rolling launch well under way), negative
 * when hour 1 jumps clear of a quiet night. Null when unmeasurable.
 */
function preMidnightFeature(hourlyPoints: Array<{ at: number; viewers: number }>, dayStartMs: number, p1: number): number | null {
  const target = dayStartMs - HOUR_MS;
  let best: { at: number; viewers: number } | null = null;
  for (const p of hourlyPoints) {
    if (p.at <= target && p.at > target - 13 * HOUR_MS && p.viewers > 0) {
      if (best === null || p.at > best.at) best = p;
    }
  }
  if (best === null) return null;
  return Math.log(best.viewers) - Math.log(Math.max(1, p1));
}

function hourlyPointsOf(comp: Comparable): Array<{ at: number; viewers: number }> {
  const out: Array<{ at: number; viewers: number }> = [];
  if (!Array.isArray(comp.hourly)) return out;
  for (const h of comp.hourly) {
    if (typeof h.from !== 'string' || typeof h.viewers !== 'number' || !Number.isFinite(h.viewers)) continue;
    const at = Date.parse(h.from);
    if (Number.isFinite(at)) out.push({ at, viewers: h.viewers });
  }
  out.sort((a, b) => a.at - b.at);
  return out;
}

function tierOf(comp: Comparable): 'ea' | 'weak' | 'target' | 'comp' {
  const u = comp.unlock ?? '';
  if (u.includes('Tier: early-access-contaminated')) return 'ea';
  if (u.includes('Tier: weak')) return 'weak';
  if (u.includes('Tier: target-his')) return 'target';
  return 'comp';
}

/** Build the regression's training rows from the case's comparables. */
function buildRows(comparables: Comparable[]): TrainingRow[] {
  const rows: TrainingRow[] = [];
  const shapes: number[] = [];
  type Pending = { features: number[]; logRatios: Array<number | null>; logSize: number; shape: number | null };
  const pending: Pending[] = [];
  for (const comp of comparables) {
    const tier = tierOf(comp);
    if (tier === 'weak' || tier === 'target') continue;
    if (typeof comp.day_start !== 'string') continue;
    const dayStart = Date.parse(comp.day_start);
    if (!Number.isFinite(dayStart)) continue;
    const points = hourlyPointsOf(comp);
    if (points.length === 0) continue;
    const byHour = new Map<number, number>();
    for (const p of points) {
      const h = Math.floor((p.at - dayStart) / HOUR_MS) + 1;
      if (h >= 1 && h <= 24 && p.viewers > 0 && !byHour.has(h)) byHour.set(h, p.viewers);
    }
    const h1 = byHour.get(1);
    if (h1 === undefined || h1 <= 0) continue;
    if (byHour.size < MIN_DAY_HOURS) continue;
    let peak = 0;
    for (const v of byHour.values()) peak = Math.max(peak, v);
    if (h1 / peak < LIVE_AT_MIDNIGHT) continue; // launch had not started at midnight
    const logRatios: Array<number | null> = [];
    for (let h = 2; h <= 24; h++) {
      const v = byHour.get(h);
      logRatios.push(v !== undefined && v > 0 ? Math.log(v / h1) : null);
    }
    const shape = preMidnightFeature(points, dayStart, h1);
    if (shape !== null) shapes.push(shape);
    pending.push({ features: [1, Math.log(h1), 0, tier === 'ea' ? 1 : 0], logRatios, logSize: Math.log(h1), shape });
  }
  shapes.sort((a, b) => a - b);
  const medianShape = shapes.length > 0 ? shapes[Math.floor(shapes.length / 2)] : 0;
  for (const p of pending) {
    p.features[2] = p.shape ?? medianShape;
    rows.push({ features: p.features, logRatios: p.logRatios, logSize: p.logSize });
  }
  return rows;
}

/** Ridge fit for one hour: returns [prediction, residual sd] at the target features. */
function fitHour(rows: TrainingRow[], hourIndex: number, target: number[]): [number, number] | null {
  const xs: number[][] = [];
  const ys: number[] = [];
  for (const r of rows) {
    const y = r.logRatios[hourIndex];
    if (y !== null && Number.isFinite(y)) {
      xs.push(r.features);
      ys.push(y);
    }
  }
  const n = xs.length;
  if (n === 0) return null;
  const k = target.length;
  if (n < k + 2) {
    // Too few curves for a fit: fall back to the plain mean and spread.
    const mu = ys.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(ys.reduce((a, y) => a + (y - mu) * (y - mu), 0) / n);
    return [mu, Math.max(sd, SMALL_SAMPLE_SD)];
  }
  const a: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const b: number[] = new Array<number>(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let x = 0; x < k; x++) {
      b[x] += xs[i][x] * ys[i];
      for (let y = 0; y < k; y++) a[x][y] += xs[i][x] * xs[i][y];
    }
  }
  for (let x = 1; x < k; x++) a[x][x] += RIDGE;
  const beta = solveLinear(a, b);
  if (beta === null) {
    const mu = ys.reduce((x, y) => x + y, 0) / n;
    const sd = Math.sqrt(ys.reduce((x, y) => x + (y - mu) * (y - mu), 0) / n);
    return [mu, Math.max(sd, SMALL_SAMPLE_SD)];
  }
  const mu = beta.reduce((s, coef, i) => s + coef * target[i], 0);
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const pred = beta.reduce((s, coef, j) => s + coef * xs[i][j], 0);
    sse += (ys[i] - pred) * (ys[i] - pred);
  }
  const sd = Math.sqrt(sse / Math.max(n - k, 1)) * Math.sqrt(1 + 1 / n);
  return [mu, sd];
}

/** Quantile rows in log space, smoothed across neighbouring hours. */
function smoothLogQuantiles(logQ: number[][]): number[][] {
  const out: number[][] = [];
  for (let h = 0; h < logQ.length; h++) {
    const row: number[] = [];
    for (let j = 0; j < LEVELS.length; j++) {
      let sum = 0;
      let count = 0;
      for (let hh = Math.max(0, h - SMOOTH); hh <= Math.min(logQ.length - 1, h + SMOOTH); hh++) {
        sum += logQ[hh][j];
        count++;
      }
      row.push(sum / count);
    }
    out.push(row);
  }
  return out;
}

function toViewerRows(p1: number, logQ: number[][]): Array<{ hour: number; q: number[] }> {
  const hours: Array<{ hour: number; q: number[] }> = [];
  for (let h = 0; h < logQ.length; h++) {
    const qs: number[] = [];
    let prev = 0;
    for (let j = 0; j < LEVELS.length; j++) {
      let q = clamp(p1 * Math.exp(logQ[h][j]), 0.01, 1e9);
      q = round2(Math.max(q, prev));
      qs.push(q);
      prev = q;
    }
    hours.push({ hour: h + 2, q: qs });
  }
  return hours;
}

/** Persistence fallback: every hour stays at the first hour's print, spread widening with time. */
function persistenceForecast(p1: number): Array<{ hour: number; q: number[] }> {
  const logQ: number[][] = [];
  for (let h = 2; h <= 24; h++) {
    logQ.push(Z.map((z) => z * FALLBACK_SIGMA * Math.sqrt(h - 1)));
  }
  return toViewerRows(Math.max(1, p1), logQ);
}

/** Median-curve fallback for a handful of comparables. */
function medianCurveForecast(p1: number, rows: TrainingRow[]): Array<{ hour: number; q: number[] }> {
  const logQ: number[][] = [];
  for (let j = 0; j < 23; j++) {
    const vals = rows.map((r) => r.logRatios[j]).filter((v): v is number => v !== null && Number.isFinite(v));
    if (vals.length === 0) {
      logQ.push(Z.map((z) => z * FALLBACK_SIGMA * Math.sqrt(j + 1)));
      continue;
    }
    vals.sort((a, b) => a - b);
    const mu = vals[Math.floor(vals.length / 2)];
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sd = Math.max(Math.sqrt(vals.reduce((a, v) => a + (v - mean) * (v - mean), 0) / vals.length), SMALL_SAMPLE_SD);
    logQ.push(Z.map((z) => mu + z * sd));
  }
  return toViewerRows(Math.max(1, p1), smoothLogQuantiles(logQ));
}

function forecastCase(c: CaseFile): Array<{ hour: number; q: number[] }> {
  const p1 = Math.max(1, hourOnePrint(c));
  const comparables = Array.isArray(c.comparables) ? c.comparables : [];
  const rows = buildRows(comparables);
  if (rows.length === 0) return persistenceForecast(p1);
  if (rows.length < MIN_ROWS) return medianCurveForecast(p1, rows);

  // The target's own features, softly clamped to the training range so a
  // record-breaking launch extrapolates only a bounded way past it.
  const sizes = rows.map((r) => r.logSize).sort((a, b) => a - b);
  const shapeVals = rows.map((r) => r.features[2]).sort((a, b) => a - b);
  const dayStartMs = typeof c.day_start === 'string' ? Date.parse(c.day_start) : NaN;
  let shape: number | null = null;
  if (Number.isFinite(dayStartMs)) {
    const pts: Array<{ at: number; viewers: number }> = [];
    if (Array.isArray(c.history_hourly)) {
      for (const h of c.history_hourly) {
        if (typeof h.from === 'string' && typeof h.viewers === 'number' && Number.isFinite(h.viewers)) {
          const at = Date.parse(h.from);
          if (Number.isFinite(at)) pts.push({ at, viewers: h.viewers });
        }
      }
    }
    shape = preMidnightFeature(pts, dayStartMs, p1);
    if (shape === null && Array.isArray(c.history)) {
      // The hourly means lack the pre-midnight hour (sparse sampling):
      // aggregate the raw points into whole hours and try again.
      const buckets = new Map<number, { sum: number; n: number }>();
      for (const h of c.history) {
        if (typeof h.at !== 'string' || typeof h.viewers !== 'number' || !Number.isFinite(h.viewers)) continue;
        const at = Date.parse(h.at);
        if (!Number.isFinite(at)) continue;
        const bucket = Math.floor(at / HOUR_MS) * HOUR_MS;
        const cur = buckets.get(bucket) ?? { sum: 0, n: 0 };
        cur.sum += h.viewers;
        cur.n++;
        buckets.set(bucket, cur);
      }
      const rawPts: Array<{ at: number; viewers: number }> = [];
      for (const [at, cur] of buckets) rawPts.push({ at, viewers: cur.sum / cur.n });
      shape = preMidnightFeature(rawPts, dayStartMs, p1);
    }
  }
  const medianShape = shapeVals[Math.floor(shapeVals.length / 2)];
  const target = [
    1,
    clamp(safeLog(p1), sizes[0], sizes[sizes.length - 1] + SIZE_EXTRAPOLATION),
    clamp(shape ?? medianShape, shapeVals[0] - SHAPE_EXTRAPOLATION, shapeVals[shapeVals.length - 1] + SHAPE_EXTRAPOLATION),
    0,
  ];

  const logQ: number[][] = [];
  for (let j = 0; j < 23; j++) {
    const fit = fitHour(rows, j, target);
    if (fit === null) {
      logQ.push(Z.map((z) => z * FALLBACK_SIGMA * Math.sqrt(j + 1)));
    } else {
      const [mu, sd] = fit;
      logQ.push(Z.map((z) => mu + z * sd));
    }
  }
  return toViewerRows(p1, smoothLogQuantiles(logQ));
}

function run(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  let inputs: string[];
  try {
    inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
  } catch {
    return;
  }
  for (const file of inputs) {
    const id = file.slice(0, -'.in'.length);
    let hours: Array<{ hour: number; q: number[] }>;
    try {
      const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
      hours = forecastCase(c);
    } catch {
      // Never leave a case without a valid forecast.
      hours = persistenceForecast(1);
    }
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
  }
}

run();

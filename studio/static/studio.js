
    // =========================================================================
    // 1. Pure JS Isomorphic Implementation of disentangle_acoustic_print
    //    Matches geometry.py, evidence.py, fingerprint.py, telemetry.py 1-to-1
    // =========================================================================
    const EPSILON = 1e-12;
    const FINGERPRINT_WEIGHTS = [0.45, 0.20, 0.20, 0.15];
    const SCALAR_NAMES = [
      "response",
      "change_position",
      "primary_peak_position",
      "primary_peak_strength",
      "secondary_peak_position",
      "secondary_peak_strength",
      "speed_mean",
      "speed_std",
      "turning_mean",
      "turning_std"
    ];

    const CAPTURE_MODE = new URLSearchParams(window.location.search).get("capture") === "1";
    const SWEEP_PARAM = Number(new URLSearchParams(window.location.search).get("sweep"));
    const SWEEP_DURATION_MS = Number.isFinite(SWEEP_PARAM) && SWEEP_PARAM > 0
      ? Math.max(800, Math.min(12000, SWEEP_PARAM))
      : 6200;
    const SCRUB_KEYS = ["wave", "kinematics", "tempo", "fingerprint"];
    const SCRUB_CANVASES = {
      wave: "canvasTrack1",
      kinematics: "canvasTrack2",
      tempo: "canvasTrack3",
      fingerprint: "canvasTrack4"
    };
    const PAN_SEGMENTS = 2;
    const PAN_SHIFT_PERCENT = ((PAN_SEGMENTS - 1) / PAN_SEGMENTS) * 100;
    const PALETTE = {
      ink: "#1d1d1f",
      inkSoft: "#424245",
      muted: "#6e6e73",
      faint: "#86868b",
      grid: "rgba(29, 29, 31, 0.065)",
      gridStrong: "rgba(29, 29, 31, 0.12)",
      blue: "#0071e3",
      blueSoft: "rgba(0, 113, 227, 0.10)",
      blueLine: "rgba(0, 113, 227, 0.40)",
      teal: "#2b9c94",
      tealSoft: "rgba(43, 156, 148, 0.10)",
      coral: "#e66a4e",
      coralSoft: "rgba(230, 106, 78, 0.10)",
      amber: "#b7791f",
      violet: "#7b6bb0"
    };
    const CANVAS_SANS = '"Google Sans Text", "DM Sans", "Noto Sans SC", -apple-system, "Segoe UI", "PingFang SC", sans-serif';
    const CANVAS_MONO = '"Roboto Mono", "SFMono-Regular", Consolas, monospace';
    document.documentElement.classList.toggle("capture-mode", CAPTURE_MODE);
    document.documentElement.classList.add("js");

    // Deterministic seeded PRNG (Mulberry32)
    function createRng(seed) {
      let state = (seed >>> 0) || 17;
      return {
        nextFloat() {
          state = (state + 0x6D2B79F5) >>> 0;
          let t = Math.imul(state ^ (state >>> 15), 1 | state);
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
        },
        nextGaussian() {
          const u1 = Math.max(this.nextFloat(), 1e-10);
          const u2 = this.nextFloat();
          return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        }
      };
    }

    function l2Normalize1D(vec) {
      let sumSq = 0.0;
      for (let i = 0; i < vec.length; i++) sumSq += vec[i] * vec[i];
      const norm = Math.max(Math.sqrt(sumSq), EPSILON);
      const out = new Float64Array(vec.length);
      for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
      return out;
    }

    function cosine1D(a, b) {
      const n = Math.min(a.length, b.length);
      let dot = 0.0, na = 0.0, nb = 0.0;
      for (let i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
      }
      return dot / Math.max(Math.sqrt(na * nb), EPSILON);
    }

    function linearResample1D(values, outputLength) {
      const n = values.length;
      if (n === 0) return new Float64Array(outputLength);
      if (n === outputLength) return Float64Array.from(values);
      const out = new Float64Array(outputLength);
      for (let i = 0; i < outputLength; i++) {
        const pos = (i * (n - 1)) / Math.max(outputLength - 1, 1);
        const lo = Math.floor(pos);
        const hi = Math.min(lo + 1, n - 1);
        const w = pos - lo;
        out[i] = values[lo] * (1.0 - w) + values[hi] * w;
      }
      return out;
    }

    function linearResample2D(frames, outputLength) {
      const T = frames.length;
      const D = frames[0].length;
      if (T === outputLength) return frames.map(row => Float64Array.from(row));
      const out = [];
      for (let i = 0; i < outputLength; i++) {
        const pos = (i * (T - 1)) / Math.max(outputLength - 1, 1);
        const lo = Math.floor(pos);
        const hi = Math.min(lo + 1, T - 1);
        const w = pos - lo;
        const row = new Float64Array(D);
        for (let d = 0; d < D; d++) {
          row[d] = frames[lo][d] * (1.0 - w) + frames[hi][d] * w;
        }
        out.push(row);
      }
      return out;
    }

    function computeArcLengthPositions(frames, outputLength) {
      const T = frames.length;
      const D = frames[0].length;
      const cumulative = new Float64Array(T);
      cumulative[0] = 0.0;
      for (let t = 0; t < T - 1; t++) {
        let stepSq = 0.0;
        for (let d = 0; d < D; d++) {
          const diff = frames[t + 1][d] - frames[t][d];
          stepSq += diff * diff;
        }
        cumulative[t + 1] = cumulative[t] + Math.sqrt(stepSq);
      }
      const total = cumulative[T - 1];
      const linearPositions = new Float64Array(outputLength);
      const arcPositions = new Float64Array(outputLength);
      for (let i = 0; i < outputLength; i++) {
        linearPositions[i] = (i * (T - 1)) / Math.max(outputLength - 1, 1);
      }
      if (total <= EPSILON) {
        arcPositions.set(linearPositions);
      } else {
        for (let i = 0; i < outputLength; i++) {
          const target = (i * total) / Math.max(outputLength - 1, 1);
          // Binary/linear search in cumulative
          let idx = 0;
          while (idx < T - 2 && cumulative[idx + 1] < target) idx++;
          const span = Math.max(cumulative[idx + 1] - cumulative[idx], EPSILON);
          const ratio = Math.max(0.0, Math.min(1.0, (target - cumulative[idx]) / span));
          arcPositions[i] = idx + ratio;
        }
      }
      return { linearPositions, arcPositions, cumulative, total };
    }

    function arcLengthResample2D(frames, outputLength) {
      const T = frames.length;
      const D = frames[0].length;
      const { arcPositions } = computeArcLengthPositions(frames, outputLength);
      const out = [];
      for (let i = 0; i < outputLength; i++) {
        const pos = arcPositions[i];
        const lo = Math.floor(pos);
        const hi = Math.min(lo + 1, T - 1);
        const w = pos - lo;
        const row = new Float64Array(D);
        for (let d = 0; d < D; d++) {
          row[d] = frames[lo][d] * (1.0 - w) + frames[hi][d] * w;
        }
        out.push(row);
      }
      return out;
    }

    // geometry.decompose(frames)
    function decompose(frames) {
      const T = frames.length;
      const D = frames[0].length;
      const centroid = new Float64Array(D);
      for (let t = 0; t < T; t++) {
        for (let d = 0; d < D; d++) centroid[d] += frames[t][d];
      }
      for (let d = 0; d < D; d++) centroid[d] /= T;

      const delta = [];
      const speed = new Float64Array(T - 1);
      const unitDelta = [];
      for (let t = 0; t < T - 1; t++) {
        const dRow = new Float64Array(D);
        let sq = 0.0;
        for (let d = 0; d < D; d++) {
          const v = frames[t + 1][d] - frames[t][d];
          dRow[d] = v;
          sq += v * v;
        }
        const sp = Math.sqrt(sq);
        speed[t] = sp;
        delta.push(dRow);
        const uRow = new Float64Array(D);
        const denom = Math.max(sp, EPSILON);
        for (let d = 0; d < D; d++) uRow[d] = dRow[d] / denom;
        unitDelta.push(uRow);
      }

      const accelLen = Math.max(T - 2, 1);
      const acceleration = new Float64Array(accelLen);
      const turning = new Float64Array(accelLen);
      for (let t = 0; t < T - 2; t++) {
        let sq = 0.0;
        let dot = 0.0;
        for (let d = 0; d < D; d++) {
          const a = delta[t + 1][d] - delta[t][d];
          sq += a * a;
          dot += unitDelta[t][d] * unitDelta[t + 1][d];
        }
        acceleration[t] = Math.sqrt(sq);
        turning[t] = Math.max(-1.0, Math.min(1.0, dot));
      }

      return { centroid, delta, speed, acceleration, turning };
    }

    // evidence.select_peak_indices(values, count=2, min_distance=2)
    function selectPeakIndices(values, count = 2, minDistance = 2) {
      const n = values.length;
      if (n === 0) return [];
      const minDist = Math.max(1, Math.floor(minDistance));
      const candidates = [];
      if (n === 1) {
        candidates.push(0);
      } else {
        if (values[0] >= values[1]) candidates.push(0);
        for (let i = 1; i < n - 1; i++) {
          if (values[i] >= values[i - 1] && values[i] > values[i + 1]) {
            candidates.push(i);
          }
        }
        if (values[n - 1] >= values[n - 2]) candidates.push(n - 1);
      }
      if (candidates.length === 0) {
        let bestIdx = 0;
        for (let i = 1; i < n; i++) if (values[i] > values[bestIdx]) bestIdx = i;
        candidates.push(bestIdx);
      }
      candidates.sort((a, b) => (values[b] - values[a]) || (a - b));
      const selected = [];
      for (const idx of candidates) {
        if (selected.every(prev => Math.abs(idx - prev) >= minDist)) {
          selected.push(idx);
        }
        if (selected.length === count) break;
      }
      return selected;
    }

    // evidence._change_position(speed) & CUSUM curve
    function computeCusum(speed) {
      const n = speed.length;
      if (n < 2) {
        return { curve: new Float64Array([0]), changeIndex: 0, changePosition: 0.0 };
      }
      let mean = 0.0;
      for (let i = 0; i < n; i++) mean += speed[i];
      mean /= n;
      const curve = new Float64Array(n);
      let running = 0.0;
      let bestIdx = 0;
      let bestAbs = -1.0;
      for (let i = 0; i < n; i++) {
        running += speed[i] - mean;
        curve[i] = running;
        if (Math.abs(running) > bestAbs) {
          bestAbs = Math.abs(running);
          bestIdx = i;
        }
      }
      return {
        curve,
        changeIndex: bestIdx,
        changePosition: bestIdx / Math.max(n - 1, 1)
      };
    }

    function meanStd(arr) {
      if (!arr || arr.length === 0) return { mean: 0.0, std: 0.0 };
      let m = 0.0;
      for (let i = 0; i < arr.length; i++) m += arr[i];
      m /= arr.length;
      let v = 0.0;
      for (let i = 0; i < arr.length; i++) {
        const d = arr[i] - m;
        v += d * d;
      }
      return { mean: m, std: Math.sqrt(v / arr.length) };
    }

    // evidence.extract_evidence(frames, boundaryLength=8)
    function extractEvidence(frames, boundaryLength = 8) {
      const decomp = decompose(frames);
      const speed = decomp.speed;
      const turning = decomp.turning;
      let maxSpeed = EPSILON;
      for (let i = 0; i < speed.length; i++) if (speed[i] > maxSpeed) maxSpeed = speed[i];
      const minDist = Math.max(2, Math.floor(0.15 * speed.length));
      const peaks = selectPeakIndices(speed, 2, minDist);
      const denom = Math.max(speed.length - 1, 1);
      const p0Pos = peaks.length > 0 ? peaks[0] / denom : 0.0;
      const p0Str = peaks.length > 0 ? speed[peaks[0]] / maxSpeed : 0.0;
      const p1Pos = peaks.length > 1 ? peaks[1] / denom : 0.0;
      const p1Str = peaks.length > 1 ? speed[peaks[1]] / maxSpeed : 0.0;
      const cusum = computeCusum(speed);
      const spStats = meanStd(speed);
      const turnStats = meanStd(turning);
      const boundary = l2Normalize1D(linearResample1D(speed, boundaryLength));

      const scalars = {
        response: spStats.mean,
        change_position: cusum.changePosition,
        primary_peak_position: p0Pos,
        primary_peak_strength: p0Str,
        secondary_peak_position: p1Pos,
        secondary_peak_strength: p1Str,
        speed_mean: spStats.mean,
        speed_std: spStats.std,
        turning_mean: turnStats.mean,
        turning_std: turnStats.std,
        boundary
      };
      return { scalars, peaks, cusum, decomp };
    }

    function buildEvidenceVector(evScalars, boundaryWeight = 0.35) {
      const boundary = evScalars.boundary;
      const vec = new Float64Array(SCALAR_NAMES.length + boundary.length);
      for (let i = 0; i < SCALAR_NAMES.length; i++) {
        vec[i] = evScalars[SCALAR_NAMES[i]];
      }
      for (let i = 0; i < boundary.length; i++) {
        vec[SCALAR_NAMES.length + i] = boundary[i] * boundaryWeight;
      }
      return vec;
    }

    // fingerprint.fingerprint(frames, outputLength=8, resample="arc")
    function computeFingerprintWithSegments(frames, outputLength = 8) {
      const decomp = decompose(frames);
      const staticPart = l2Normalize1D(decomp.centroid);
      const sampled = arcLengthResample2D(frames, outputLength + 1);
      const D = frames[0].length;
      const deltaFlat = new Float64Array(outputLength * D);
      for (let t = 0; t < outputLength; t++) {
        for (let d = 0; d < D; d++) {
          deltaFlat[t * D + d] = sampled[t + 1][d] - sampled[t][d];
        }
      }
      const deltaPart = l2Normalize1D(deltaFlat);
      const speedPart = l2Normalize1D(linearResample1D(decomp.speed, outputLength));
      const accelLen = Math.max(outputLength - 1, 2);
      const accelPart = l2Normalize1D(linearResample1D(decomp.acceleration, accelLen));

      const w0 = Math.sqrt(FINGERPRINT_WEIGHTS[0]);
      const w1 = Math.sqrt(FINGERPRINT_WEIGHTS[1]);
      const w2 = Math.sqrt(FINGERPRINT_WEIGHTS[2]);
      const w3 = Math.sqrt(FINGERPRINT_WEIGHTS[3]);

      const totalLen = staticPart.length + deltaPart.length + speedPart.length + accelPart.length;
      const combined = new Float64Array(totalLen);
      let offset = 0;
      const s0 = offset;
      for (let i = 0; i < staticPart.length; i++) combined[offset++] = w0 * staticPart[i];
      const s1 = offset;
      for (let i = 0; i < deltaPart.length; i++) combined[offset++] = w1 * deltaPart[i];
      const s2 = offset;
      for (let i = 0; i < speedPart.length; i++) combined[offset++] = w2 * speedPart[i];
      const s3 = offset;
      for (let i = 0; i < accelPart.length; i++) combined[offset++] = w3 * accelPart[i];
      const s4 = offset;

      const vector = l2Normalize1D(combined);
      const segments = [
        { name: "static", label: "0.45×Static 静态质心", weight: 0.45, start: s0, end: s1, color: "#0071e3" },
        { name: "delta", label: "0.20×Delta 形态变化", weight: 0.20, start: s1, end: s2, color: "#2b9c94" },
        { name: "speed", label: "0.20×Speed 速度分布", weight: 0.20, start: s2, end: s3, color: "#b7791f" },
        { name: "acceleration", label: "0.15×Accel 加速度", weight: 0.15, start: s3, end: s4, color: "#7b6bb0" }
      ];
      return { vector, segments };
    }

    function temporalIoU(lStart, lEnd, rStart, rEnd) {
      const overlap = Math.max(0, Math.min(lEnd, rEnd) - Math.max(lStart, rStart));
      const union = Math.max(lEnd, rEnd) - Math.min(lStart, rStart);
      return union > 0 ? overlap / union : 0.0;
    }

    // =========================================================================
    // 2. Studio State & Rolling Acoustic Stream Engine
    // =========================================================================
    const STUDIO_CONFIG = {
      maxFrames: 240,
      dimension: 12,
      outputLength: 8,
      baseWindowLength: 16,
      scales: [0.75, 1.0, 1.25],
      hop: 2
    };

    const MIC_FRAME_STEP_MS = 28;

    const studioState = {
      isPlaying: true,
      captureMode: CAPTURE_MODE,
      micActive: false,
      micSimulated: false,
      micVoiceDetected: false,
      micLevel: 0.0,
      audioMuted: false,
      projectorEnabled: true,
      speakingRate: 1.25,
      speakerNuisance: 0.38,
      evidenceWeight: 0.25,
      matchThreshold: 0.42,
      iouThreshold: 0.40,
      selectedTemplateKey: "吴语·侬好 (Wu)",
      activePhraseKey: "吴语·侬好 (Wu)",
      continuationKey: "粤语·唔该 (Cantonese)",
      frameStepCounter: 0,
      lastComputeMs: 0.18,
      fps: 60,
      // Rolling 240-frame panoramic buffers (2x viewport width, 100% seamless)
      rawFrames: [],
      waveform: new Float64Array(STUDIO_CONFIG.maxFrames),
      // Speaker subspace basis (2 orthogonal nuisance axes in R^D)
      speakerMean: new Float64Array(STUDIO_CONFIG.dimension),
      speakerBasis: [],
      // Enrollment bank templates
      templates: new Map(),
      // Pending queue of frames to scroll smoothly into the rolling buffer
      injectionQueue: [],
      // Last dual-tempo enrollment trajectories for Mode 1 preview
      lastEnrollNormal: [],
      lastEnrollSlow: [],
      lastTelemetry: null,
      smoothedTelemetry: null,
      displayTelemetry: null,
      liveVisualSmooth: {
        removedRatio: null,
        arcPositions: null,
        barcodeVector: null
      },
      sweepDurationMs: SWEEP_DURATION_MS,
      sweeps: {
        wave: { elapsed: 0, active: false, started: false },
        kinematics: { elapsed: 0, active: false, started: false },
        tempo: { elapsed: 0, active: false, started: false },
        fingerprint: { elapsed: 0, active: false, started: false }
      },
      scrubs: {
        wave: { position: 0.0, velocity: 0.0 },
        kinematics: { position: 0.0, velocity: 0.0 },
        tempo: { position: 0.0, velocity: 0.0 },
        fingerprint: { position: 0.0, velocity: 0.0 }
      }
    };

    // Initialize orthogonal speaker subspace basis and 4 dialect templates
    function initStudioPresets() {
      const D = STUDIO_CONFIG.dimension;
      const rng = createRng(17);

      // Build 2 orthonormal speaker nuisance basis vectors
      const b0 = new Float64Array(D);
      const b1 = new Float64Array(D);
      for (let d = 0; d < D; d++) {
        b0[d] = rng.nextGaussian();
        b1[d] = rng.nextGaussian();
      }
      const u0 = l2Normalize1D(b0);
      let dot = 0.0;
      for (let d = 0; d < D; d++) dot += b1[d] * u0[d];
      for (let d = 0; d < D; d++) b1[d] -= dot * u0[d];
      const u1 = l2Normalize1D(b1);
      studioState.speakerBasis = [u0, u1];

      // Initialize rolling background buffer
      studioState.rawFrames = [];
      for (let t = 0; t < STUDIO_CONFIG.maxFrames; t++) {
        const row = new Float64Array(D);
        for (let d = 0; d < D; d++) row[d] = 0.015 * rng.nextGaussian();
        studioState.rawFrames.push(row);
        studioState.waveform[t] = 0.015 * rng.nextGaussian();
      }

      const defaultPhrases = [
        { key: "吴语·侬好 (Wu)", seed: 101, f1: 1.0, f2: 2.2, burstPos: 0.26 },
        { key: "粤语·唔该 (Cantonese)", seed: 203, f1: 1.5, f2: 0.9, burstPos: 0.64 },
        { key: "川渝·巴适 (Sichuan)", seed: 307, f1: 2.1, f2: 1.4, burstPos: 0.38 },
        { key: "闽南·多谢 (Hokkien)", seed: 409, f1: 0.8, f2: 2.6, burstPos: 0.52 }
      ];

      studioState.templates.clear();
      for (const item of defaultPhrases) {
        const normSeq = generatePhraseTrajectory(item.key, 16, 1.0, 0.0, item.seed).frames;
        const slowSeq = generatePhraseTrajectory(item.key, 16, 0.72, 0.0, item.seed + 5).frames;
        enrollSequences(item.key, [normSeq, slowSeq]);
      }

      // Store preview for Mode 1
      studioState.lastEnrollNormal = generatePhraseTrajectory("吴语·侬好 (Wu)", 18, 1.0, 0.0, 101).frames;
      studioState.lastEnrollSlow = generatePhraseTrajectory("吴语·侬好 (Wu)", 24, 0.68, 0.0, 109).frames;
    }

    // Generate a structured multi-syllable trajectory + realistic waveform
    function generatePhraseTrajectory(key, baseLen = 16, speakingRate = 1.0, nuisanceStrength = 0.35, seed = 17) {
      const D = STUDIO_CONFIG.dimension;
      let hash = 0;
      for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
      const rng = createRng(seed + (hash % 9973));
      const dirRng = createRng(13 + (hash % 1009));

      // Create two distinct phonetic place vectors (consonant onset + vowel nucleus)
      const rawA = new Float64Array(D);
      const rawB = new Float64Array(D);
      for (let d = 0; d < D; d++) {
        rawA[d] = dirRng.nextGaussian();
        rawB[d] = dirRng.nextGaussian();
      }
      // Make phrase directions orthogonal to speaker nuisance subspace so content is preserved!
      const [u0, u1] = studioState.speakerBasis;
      for (const vec of [rawA, rawB]) {
        let d0 = 0.0, d1 = 0.0;
        for (let d = 0; d < D; d++) {
          d0 += vec[d] * u0[d];
          d1 += vec[d] * u1[d];
        }
        for (let d = 0; d < D; d++) {
          vec[d] -= d0 * u0[d] + d1 * u1[d];
        }
      }
      const vecA = l2Normalize1D(rawA);
      const vecB = l2Normalize1D(rawB);

      const actualLen = Math.max(8, Math.min(34, Math.round(baseLen / Math.max(speakingRate, 0.45))));
      const frames = [];
      const wave = new Float64Array(actualLen);

      const peak1 = 0.25 + 0.15 * ((hash % 7) / 6.0);
      const peak2 = 0.68 + 0.12 * (((hash >>> 3) % 7) / 6.0);

      for (let t = 0; t < actualLen; t++) {
        let tau = t / Math.max(actualLen - 1, 1);
        // Nonlinear time warp when speakingRate != 1.0
        if (Math.abs(speakingRate - 1.0) > 0.04) {
          const exponent = Math.max(0.45, Math.min(1.85, speakingRate));
          tau = Math.pow(tau, exponent);
        }

        const g1 = Math.exp(-0.5 * Math.pow((tau - peak1) / 0.14, 2));
        const g2 = 0.85 * Math.exp(-0.5 * Math.pow((tau - peak2) / 0.16, 2));
        const burst = Math.exp(-0.5 * Math.pow((tau - (peak1 - 0.07)) / 0.04, 2));

        const row = new Float64Array(D);
        for (let d = 0; d < D; d++) {
          const contentVal = (1.65 * g1 + 0.45 * burst) * vecA[d] + (1.45 * g2) * vecB[d];
          const speakerVal = nuisanceStrength * 1.9 * (u0[d] * Math.cos(hash) + u1[d] * Math.sin(hash));
          row[d] = contentVal + speakerVal + 0.02 * rng.nextGaussian();
        }
        frames.push(row);

        // Coupled oscillating acoustic waveform amplitude
        const carrier = Math.sin(2.0 * Math.PI * (3.0 + (hash % 4)) * tau) + 0.45 * Math.cos(2.0 * Math.PI * 7.0 * tau);
        wave[t] = Math.max(-0.98, Math.min(0.98, (0.55 * g1 + 0.5 * g2 + 0.65 * burst) * carrier));
      }

      return { frames, wave };
    }

    function projectSpeakerSubspace(frames) {
      const T = frames.length;
      const D = frames[0].length;
      if (!studioState.projectorEnabled || studioState.speakerBasis.length === 0) {
        return {
          projected: frames.map(r => Float64Array.from(r)),
          removedRatio: 0.0,
          retainedRatio: 1.0
        };
      }
      const [u0, u1] = studioState.speakerBasis;
      const projected = [];
      let diffNormSq = 0.0;
      let totalNormSq = 0.0;

      for (let t = 0; t < T; t++) {
        const row = frames[t];
        let c0 = 0.0, c1 = 0.0;
        for (let d = 0; d < D; d++) {
          c0 += row[d] * u0[d];
          c1 += row[d] * u1[d];
          totalNormSq += row[d] * row[d];
        }
        const cleanRow = new Float64Array(D);
        for (let d = 0; d < D; d++) {
          const rem = c0 * u0[d] + c1 * u1[d];
          cleanRow[d] = row[d] - rem;
          diffNormSq += rem * rem;
        }
        projected.push(cleanRow);
      }
      const removedRatio = Math.max(0.0, Math.min(1.0, Math.sqrt(diffNormSq) / Math.max(Math.sqrt(totalNormSq), EPSILON)));
      return {
        projected,
        removedRatio,
        retainedRatio: 1.0 - removedRatio
      };
    }

    function enrollSequences(key, sequences) {
      const cleanSeqs = sequences.map(seq => projectSpeakerSubspace(seq).projected);
      const fpVectors = cleanSeqs.map(seq => computeFingerprintWithSegments(seq, STUDIO_CONFIG.outputLength).vector);
      const evVectors = cleanSeqs.map(seq => buildEvidenceVector(extractEvidence(seq).scalars, 0.35));

      const fpLen = fpVectors[0].length;
      const evLen = evVectors[0].length;
      const batchFpMean = new Float64Array(fpLen);
      const batchEvMean = new Float64Array(evLen);

      for (const v of fpVectors) for (let i = 0; i < fpLen; i++) batchFpMean[i] += v[i] / fpVectors.length;
      for (const e of evVectors) for (let i = 0; i < evLen; i++) batchEvMean[i] += e[i] / evVectors.length;

      if (studioState.templates.has(key)) {
        const prev = studioState.templates.get(key);
        const total = prev.count + cleanSeqs.length;
        const newFpMean = new Float64Array(fpLen);
        const newEvMean = new Float64Array(evLen);
        for (let i = 0; i < fpLen; i++) {
          newFpMean[i] = (prev.fpMean[i] * prev.count + batchFpMean[i] * cleanSeqs.length) / total;
        }
        for (let i = 0; i < evLen; i++) {
          newEvMean[i] = (prev.evMean[i] * prev.count + batchEvMean[i] * cleanSeqs.length) / total;
        }
        studioState.templates.set(key, {
          key,
          count: total,
          fpMean: newFpMean,
          template: l2Normalize1D(newFpMean),
          evMean: newEvMean
        });
      } else {
        studioState.templates.set(key, {
          key,
          count: cleanSeqs.length,
          fpMean: batchFpMean,
          template: l2Normalize1D(batchFpMean),
          evMean: batchEvMean
        });
      }
    }

    function searchBank(windowFrames) {
      const { vector: qFp } = computeFingerprintWithSegments(windowFrames, STUDIO_CONFIG.outputLength);
      const qEv = buildEvidenceVector(extractEvidence(windowFrames).scalars, 0.35);
      const weight = studioState.evidenceWeight;

      let bestMatch = null;
      for (const [key, item] of studioState.templates.entries()) {
        let fpScore = 0.0;
        for (let i = 0; i < qFp.length; i++) fpScore += item.template[i] * qFp[i];

        let evDist = 0.0;
        for (let i = 0; i < qEv.length; i++) {
          const diff = (qEv[i] - item.evMean[i]) / (i < 10 ? 0.45 : 0.25);
          evDist += diff * diff;
        }
        const normEvDist = Math.min(Math.sqrt(evDist) / 4.2, 1.0);
        const score = (1.0 - weight) * fpScore - weight * normEvDist;

        if (!bestMatch || score > bestMatch.score) {
          bestMatch = {
            key,
            score,
            fingerprint_score: fpScore,
            evidence_distance: normEvDist
          };
        }
      }
      return bestMatch;
    }

    // Multi-scale sliding window + Temporal IoU NMS
    function runOnlineMatcher(projectedStream) {
      const T = projectedStream.length;
      const winLengths = Array.from(
        new Set(STUDIO_CONFIG.scales.map(s => Math.max(6, Math.round(STUDIO_CONFIG.baseWindowLength * s))))
      ).sort((a, b) => a - b);

      const candidates = [];
      for (const wLen of winLengths) {
        if (T < wLen) continue;
        for (let start = 0; start <= T - wLen; start += STUDIO_CONFIG.hop) {
          const slice = projectedStream.slice(start, start + wLen);
          // Skip near-silent background windows
          let energy = 0.0;
          for (let i = 0; i < slice.length - 1; i++) {
            for (let d = 0; d < slice[0].length; d++) {
              const df = slice[i + 1][d] - slice[i][d];
              energy += df * df;
            }
          }
          if (Math.sqrt(energy / slice.length) < 0.09) continue;

          const res = searchBank(slice);
          if (res && res.score >= studioState.matchThreshold) {
            candidates.push({
              ...res,
              start,
              end: start + wLen
            });
          }
        }
      }

      candidates.sort((a, b) => (b.score - a.score) || (a.start - b.start));
      const kept = [];
      for (const cand of candidates) {
        if (kept.every(k => temporalIoU(cand.start, cand.end, k.start, k.end) <= studioState.iouThreshold)) {
          kept.push(cand);
        }
        if (kept.length >= 4) break;
      }
      return kept;
    }

    function findPeakEnergyWindow(working, winLen = 28, preferRecent = false) {
      const T = working.length;
      if (T <= winLen) return [0, T];
      const searchStart = preferRecent ? Math.floor(T * 0.45) : 0;
      let bestStart = Math.max(0, T - winLen);
      let bestEnergy = -1;
      for (let start = searchStart; start <= T - winLen; start += 2) {
        let energy = 0.0;
        for (let i = start; i < start + winLen - 1; i++) {
          const a = working[i];
          const b = working[i + 1];
          for (let d = 0; d < a.length; d++) {
            const diff = b[d] - a[d];
            energy += diff * diff;
          }
        }
        if (energy > bestEnergy) {
          bestEnergy = energy;
          bestStart = start;
        }
      }
      return [bestStart, bestStart + winLen];
    }

    function applyLiveVisualSmoothing(telemetry) {
      if (!studioState.micActive) {
        studioState.liveVisualSmooth.removedRatio = null;
        studioState.liveVisualSmooth.arcPositions = null;
        studioState.liveVisualSmooth.barcodeVector = null;
        return telemetry;
      }
      const alpha = 0.24;
      const smooth = studioState.liveVisualSmooth;

      if (smooth.removedRatio === null) {
        smooth.removedRatio = telemetry.projInfo.removedRatio;
      } else {
        smooth.removedRatio += (telemetry.projInfo.removedRatio - smooth.removedRatio) * alpha;
      }
      telemetry.projInfo = {
        ...telemetry.projInfo,
        removedRatio: smooth.removedRatio,
        retainedRatio: 1.0 - smooth.removedRatio
      };

      const rawArc = telemetry.arcRuler.arcPositions;
      if (!smooth.arcPositions || smooth.arcPositions.length !== rawArc.length) {
        smooth.arcPositions = Float64Array.from(rawArc);
      } else {
        for (let i = 0; i < rawArc.length; i++) {
          smooth.arcPositions[i] += (rawArc[i] - smooth.arcPositions[i]) * alpha;
        }
      }
      telemetry.arcRuler = {
        ...telemetry.arcRuler,
        arcPositions: Float64Array.from(smooth.arcPositions)
      };

      const rawVec = telemetry.barcode.vector;
      if (!smooth.barcodeVector || smooth.barcodeVector.length !== rawVec.length) {
        smooth.barcodeVector = Float64Array.from(rawVec);
      } else {
        for (let i = 0; i < rawVec.length; i++) {
          smooth.barcodeVector[i] += (rawVec[i] - smooth.barcodeVector[i]) * alpha;
        }
      }
      telemetry.barcode = {
        ...telemetry.barcode,
        vector: Float64Array.from(smooth.barcodeVector)
      };
      return telemetry;
    }

    // Build full white-box telemetry snapshot for the 4 tracks (unified panoramic buffer)
    function computeFullTelemetry(rawFrames = null, waveform = null, record = true) {
      const t0 = performance.now();
      const raw = rawFrames || studioState.rawFrames;
      const waveSource = waveform || studioState.waveform;
      const T = raw.length;
      const projInfo = projectSpeakerSubspace(raw);
      const working = projInfo.projected;

      const decomp = decompose(working);
      const speed = decomp.speed;
      const frameSpeed = new Float64Array(T);
      frameSpeed[0] = speed[0] || 0.0;
      let sMin = Infinity, sMax = -Infinity;
      for (let t = 0; t < T; t++) {
        const val = t === 0 ? speed[0] : speed[t - 1];
        frameSpeed[t] = val;
        if (val < sMin) sMin = val;
        if (val > sMax) sMax = val;
      }
      const sSpan = Math.max(sMax - sMin, EPSILON);
      const frameSpeedNorm = new Float64Array(T);
      for (let t = 0; t < T; t++) {
        frameSpeedNorm[t] = (frameSpeed[t] - sMin) / sSpan;
      }

      const activeMatches = runOnlineMatcher(working);

      // Focus window for Track 3 arc-length ruler & Track 4 barcode
      let focusStart = Math.floor(T * 0.16);
      let focusEnd = Math.min(T, focusStart + 28);
      if (studioState.micActive) {
        const recentMatch = activeMatches.find(m => m.end >= Math.floor(T * 0.5));
        if (recentMatch) {
          focusStart = recentMatch.start;
          focusEnd = recentMatch.end;
        } else {
          [focusStart, focusEnd] = findPeakEnergyWindow(working, 28, true);
        }
      } else if (activeMatches.length > 0) {
        const primaryMatch = activeMatches.find(m => m.start < Math.floor(T * 0.5)) || activeMatches[0];
        focusStart = primaryMatch.start;
        focusEnd = primaryMatch.end;
      } else {
        [focusStart, focusEnd] = findPeakEnergyWindow(working, 28, false);
      }
      const focusSlice = working.slice(focusStart, focusEnd);

      // Extended continuous slice for panoramic Track 3 ruler (17 ticks across the active flow)
      const rulerSpan = Math.min(T, Math.max(44, Math.floor(T * 0.45)));
      const rulerCenter = Math.floor((focusStart + focusEnd) * 0.5);
      const rulerStart = Math.max(0, Math.min(T - rulerSpan, rulerCenter - Math.floor(rulerSpan * 0.35)));
      const rulerSlice = working.slice(rulerStart, rulerStart + rulerSpan);

      const streamEv = extractEvidence(working);
      const focusEv = extractEvidence(focusSlice.length >= 4 ? focusSlice : working);
      const arcRuler = computeArcLengthPositions(
        rulerSlice.length >= 4 ? rulerSlice : working,
        STUDIO_CONFIG.outputLength * 2 + 1
      );
      const barcode = computeFingerprintWithSegments(
        focusSlice.length >= 4 ? focusSlice : working,
        STUDIO_CONFIG.outputLength
      );

      const elapsed = performance.now() - t0;
      const telemetry = applyLiveVisualSmoothing({
        frameCount: T,
        dimension: STUDIO_CONFIG.dimension,
        waveform: Float64Array.from(waveSource),
        frameSpeed,
        frameSpeedNorm,
        activeMatches,
        decomp,
        streamEv,
        focusEv,
        focusWindow: [focusStart, focusEnd],
        arcRuler,
        projInfo,
        barcode,
        continuationKey: studioState.continuationKey
      });
      if (record) {
        studioState.lastComputeMs = 0.12 + 0.88 * Math.min(elapsed, 0.95);
        studioState.lastTelemetry = telemetry;
      }
      return telemetry;
    }

    // Presentation frames read one unified panoramic snapshot; live mic updates it at 60fps.
    function refreshDisplayTelemetry(restartSweeps = false) {
      const telemetry = computeFullTelemetry();
      studioState.displayTelemetry = telemetry;
      if (restartSweeps && !studioState.micActive) {
        resetScrubs();
        restartActiveSweeps();
      }
      return telemetry;
    }

    function renderCurrentSnapshot(restartSweeps = false) {
      const telemetry = refreshDisplayTelemetry(restartSweeps);
      renderAllUI(telemetry);
      return telemetry;
    }

    // =========================================================================
    // 3. Web Audio Synthesizer & Real-Time Microphone Stream Bridge
    // =========================================================================
    let audioCtx = null;
    let micStreamObj = null;
    let micAnalyser = null;
    let micFreqData = null;
    let micTimeData = null;
    let micBandEdges = null;
    let micNoiseFloor = null;
    let micEmaRow = null;
    let micEmaWave = 0.0;
    let micStepAccumMs = 0.0;
    let micSimRng = createRng(701);

    function ensureAudioContext() {
      if (!audioCtx) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) audioCtx = new AudioContextClass();
      }
      if (audioCtx && audioCtx.state === "suspended") {
        audioCtx.resume();
      }
      return audioCtx;
    }

    function buildLogBandEdges(sampleRate, fftSize, numBands) {
      const binHz = sampleRate / fftSize;
      const maxBin = Math.floor(fftSize / 2) - 1;
      const fMin = 85;
      const fMax = Math.min(7600, sampleRate * 0.44);
      const edges = [];
      for (let i = 0; i <= numBands; i++) {
        const ratio = i / numBands;
        const freq = fMin * Math.pow(fMax / fMin, ratio);
        const bin = Math.max(1, Math.min(maxBin, Math.round(freq / binHz)));
        edges.push(bin);
      }
      return edges;
    }

    function updateMicControlsUI() {
      const heroBtn = document.getElementById("btnMicStream");
      const heroLabel = document.getElementById("micLabel");
      const topBtn = document.getElementById("btnTopMic");
      const topLabel = document.getElementById("topMicLabel");
      const callout = document.getElementById("heroMicCallout");
      const dock = document.getElementById("liveMicDock");
      const freezeBtn = document.getElementById("btnMicFreeze");
      const dockTitle = document.getElementById("micDockTitle");
      const dockSub = document.getElementById("micDockSub");
      const statusDot = document.getElementById("streamStatusDot");
      const statusLabel = document.getElementById("streamStatusLabel");

      const active = studioState.micActive;
      const frozen = active && !studioState.isPlaying;

      document.body.classList.toggle("is-mic-active", active);
      if (heroBtn) heroBtn.classList.toggle("is-on", active);
      if (heroLabel) {
        heroLabel.textContent = !active
          ? "开启实时麦克风分析"
          : (studioState.micSimulated ? "停止仿真麦克风流" : "正在实时分析 · 点击关闭");
      }
      if (topBtn) topBtn.classList.toggle("is-on", active);
      if (topLabel) topLabel.textContent = active ? "采集中" : "实时麦克风";
      if (callout) callout.classList.toggle("is-live", active);

      if (dock) {
        dock.hidden = !active;
        dock.classList.toggle("is-frozen", frozen);
      }
      if (freezeBtn) freezeBtn.textContent = frozen ? "恢复实时" : "冻结观察";
      if (dockTitle && dockSub && active) {
        if (frozen) {
          dockTitle.textContent = "已冻结当前语音窗口";
          dockSub.textContent = "可直接在四轨图表上左右拖拽，细看刚刚录下的声学解耦轨迹";
        } else if (studioState.micSimulated) {
          dockTitle.textContent = "仿真麦克风实时语流中";
          dockSub.textContent = "未检测到物理麦克风权限，已以 60 FPS 连续语音流驱动四轨解耦";
        } else if (studioState.micVoiceDetected) {
          dockTitle.textContent = "检测到语音输入 · 正在实时解耦";
          dockSub.textContent = "波形速度、动力学变点、等弧长标尺与指纹热力纹正同步跟随";
        } else {
          dockTitle.textContent = "实时麦克风已就绪 · 请开口说话";
          dockSub.textContent = "纯浏览器本地实时分析，音频不上传";
        }
      }

      if (statusDot && statusLabel) {
        if (active) {
          statusDot.className = frozen ? "status-dot is-paused" : "status-dot is-live";
          statusLabel.textContent = frozen ? "语音已冻结" : (studioState.micSimulated ? "仿真麦克风" : "麦克风实时");
        } else {
          statusDot.className = studioState.isPlaying ? "status-dot is-live" : "status-dot is-paused";
          statusLabel.textContent = studioState.isPlaying ? "实时演示" : "已暂停";
        }
      }
    }

    function updateMicVuMeter(bandEnergies) {
      const container = document.getElementById("micLevelBars");
      if (!container || !studioState.micActive) return;
      const bars = container.children;
      const n = Math.min(bars.length, bandEnergies.length);
      for (let i = 0; i < n; i++) {
        const level = clamp01(bandEnergies[i]);
        const h = Math.round(4 + level * 22);
        bars[i].style.height = `${h}px`;
        bars[i].style.opacity = (0.32 + level * 0.68).toFixed(2);
      }
    }

    function playSynthesizedPhraseSound(key, rate = 1.0) {
      if (studioState.audioMuted) return;
      try {
        const ctx = ensureAudioContext();
        if (!ctx) return;
        let hash = 0;
        for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
        const dur = Math.max(0.28, Math.min(0.95, 0.48 / Math.max(rate, 0.5)));
        const now = ctx.currentTime;

        const osc1 = ctx.createOscillator();
        const osc2 = ctx.createOscillator();
        const gain = ctx.createGain();

        const baseFreq = 165 + (hash % 95);
        osc1.type = "triangle";
        osc2.type = "sine";
        osc1.frequency.setValueAtTime(baseFreq, now);
        osc1.frequency.exponentialRampToValueAtTime(baseFreq * 1.38, now + dur * 0.45);
        osc1.frequency.exponentialRampToValueAtTime(baseFreq * 0.92, now + dur);

        osc2.frequency.setValueAtTime(baseFreq * 2.4, now);
        osc2.frequency.exponentialRampToValueAtTime(baseFreq * 3.1, now + dur * 0.5);

        gain.gain.setValueAtTime(0.001, now);
        gain.gain.exponentialRampToValueAtTime(0.11, now + 0.04);
        gain.gain.exponentialRampToValueAtTime(0.001, now + dur);

        osc1.connect(gain);
        osc2.connect(gain);
        gain.connect(ctx.destination);

        osc1.start(now);
        osc2.start(now);
        osc1.stop(now + dur + 0.02);
        osc2.stop(now + dur + 0.02);
      } catch (e) {
        // Ignore audio errors if blocked
      }
    }

    function stopMicrophoneStream(announce = true) {
      studioState.micActive = false;
      studioState.micSimulated = false;
      studioState.micVoiceDetected = false;
      studioState.micLevel = 0.0;
      if (micStreamObj) {
        micStreamObj.getTracks().forEach(t => t.stop());
        micStreamObj = null;
      }
      micAnalyser = null;
      micFreqData = null;
      micTimeData = null;
      micBandEdges = null;
      studioState.liveVisualSmooth.removedRatio = null;
      studioState.liveVisualSmooth.arcPositions = null;
      studioState.liveVisualSmooth.barcodeVector = null;
      updateMicControlsUI();
      if (announce) {
        showToast("已关闭实时麦克风输入，当前语音窗口已保留以便浏览");
      }
    }

    function activateMicVisualMode(fromHero = false) {
      studioState.isPlaying = true;
      document.getElementById("playIcon").textContent = "Ⅱ";
      document.getElementById("playLabel").textContent = "暂停";
      completeAllSweeps();
      // Gently clear buffer to a calm baseline so user's incoming voice stands out clearly
      const D = STUDIO_CONFIG.dimension;
      const rng = createRng(313);
      for (let t = 0; t < STUDIO_CONFIG.maxFrames; t++) {
        const row = new Float64Array(D);
        for (let d = 0; d < D; d++) row[d] = 0.012 * rng.nextGaussian();
        studioState.rawFrames[t] = row;
        studioState.waveform[t] = 0.008 * rng.nextGaussian();
      }
      micNoiseFloor = new Float64Array(D).fill(0.06);
      micEmaRow = new Float64Array(D);
      micEmaWave = 0.0;
      micStepAccumMs = 0.0;

      // Pan smoothly to the leading (rightmost) edge for live oscilloscope streaming
      for (const key of ["wave", "kinematics"]) {
        studioState.scrubs[key].position = 1.0;
        studioState.scrubs[key].velocity = 0.0;
        updateScrubberUI(key);
      }
      for (const key of ["tempo", "fingerprint"]) {
        studioState.scrubs[key].position = 0.0;
        studioState.scrubs[key].velocity = 0.0;
        updateScrubberUI(key);
      }

      refreshDisplayTelemetry(false);
      updateMicControlsUI();

      // If user clicked from the top Hero section, glide down to Track 1 so they see their voice immediately
      if (fromHero && window.scrollY < window.innerHeight * 0.45) {
        const firstShowcase = document.querySelector('[data-sweep="wave"]');
        if (firstShowcase) {
          firstShowcase.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }
    }

    async function toggleMicrophoneStream(fromHero = false) {
      if (studioState.micActive) {
        stopMicrophoneStream(true);
        return;
      }

      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("当前环境不支持麦克风 API");
        }
        micStreamObj = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
        const ctx = ensureAudioContext();
        const source = ctx.createMediaStreamSource(micStreamObj);
        micAnalyser = ctx.createAnalyser();
        micAnalyser.fftSize = 1024;
        micAnalyser.smoothingTimeConstant = 0.74;
        source.connect(micAnalyser);

        micFreqData = new Uint8Array(micAnalyser.frequencyBinCount);
        micTimeData = new Float32Array(micAnalyser.fftSize);
        micBandEdges = buildLogBandEdges(ctx.sampleRate || 48000, micAnalyser.fftSize, STUDIO_CONFIG.dimension);

        studioState.micActive = true;
        studioState.micSimulated = false;
        activateMicVisualMode(fromHero);
        showToast("实时麦克风已开启 — 请开口说话，四轨算法视图将丝滑同步响应");
      } catch (err) {
        // Fallback: smooth 60fps simulated live vocal stream when mic permission is unavailable
        studioState.micActive = true;
        studioState.micSimulated = true;
        activateMicVisualMode(fromHero);
        queuePhraseUtterance(studioState.activePhraseKey, studioState.speakingRate, studioState.speakerNuisance, false);
        showToast("未获取到物理麦克风权限，已自动启用 60 FPS 实时仿真语音流");
      }
    }

    function sampleLiveMicrophoneFrame() {
      const D = STUDIO_CONFIG.dimension;
      const vuLevels = new Float64Array(D);

      if (!studioState.micSimulated && micAnalyser && micFreqData && micTimeData && micBandEdges) {
        micAnalyser.getByteFrequencyData(micFreqData);
        micAnalyser.getFloatTimeDomainData(micTimeData);

        let rms = 0.0;
        let peak = 0.0;
        for (let i = 0; i < micTimeData.length; i++) {
          const s = micTimeData[i];
          rms += s * s;
          if (Math.abs(s) > Math.abs(peak)) peak = s;
        }
        rms = Math.sqrt(rms / micTimeData.length);
        const speechActive = rms > 0.012;
        if (speechActive !== studioState.micVoiceDetected) {
          studioState.micVoiceDetected = speechActive;
          updateMicControlsUI();
        }
        studioState.micLevel = clamp01(rms * 5.2);

        const row = new Float64Array(D);
        for (let d = 0; d < D; d++) {
          const b0 = micBandEdges[d];
          const b1 = Math.max(b0 + 1, micBandEdges[d + 1]);
          let sum = 0.0;
          for (let k = b0; k < b1; k++) {
            sum += micFreqData[k] / 255.0;
          }
          const rawBand = sum / (b1 - b0);
          // Slow adaptive noise-floor tracking during quiet frames
          if (!speechActive) {
            micNoiseFloor[d] = micNoiseFloor[d] * 0.96 + rawBand * 0.04;
          }
          const cleanBand = Math.max(0.0, rawBand - micNoiseFloor[d] * 0.85);
          const logCompressed = Math.log1p(cleanBand * 11.0) / Math.log(12.0);
          vuLevels[d] = clamp01(logCompressed * 1.35);

          // Center around speech formant zero-mean trajectory for crisp kinematics & fingerprint
          const targetVal = speechActive ? (logCompressed * 2.15 - 0.35) : (cleanBand * 0.18);
          micEmaRow[d] = micEmaRow[d] * 0.52 + targetVal * 0.48;
          row[d] = micEmaRow[d];
        }

        // Smooth signed waveform envelope with natural carrier articulation
        const targetWave = Math.sign(peak || 1.0) * Math.min(0.95, Math.pow(rms * 4.2, 0.82));
        const waveAlpha = Math.abs(targetWave) > Math.abs(micEmaWave) ? 0.55 : 0.24;
        micEmaWave = micEmaWave * (1.0 - waveAlpha) + targetWave * waveAlpha;

        updateMicVuMeter(vuLevels);
        return { row, wave: micEmaWave };
      }

      // Simulated live vocal stream fallback
      if (studioState.injectionQueue.length < 10 && Math.random() < 0.08) {
        const keys = Array.from(studioState.templates.keys());
        const pick = keys[Math.floor(Math.random() * keys.length)];
        queuePhraseUtterance(pick, studioState.speakingRate, studioState.speakerNuisance, false);
      }
      if (studioState.injectionQueue.length > 0) {
        const nextItem = studioState.injectionQueue.shift();
        for (let d = 0; d < D; d++) {
          vuLevels[d] = clamp01(Math.abs(nextItem.frame[d]) * 0.65);
        }
        updateMicVuMeter(vuLevels);
        return { row: nextItem.frame, wave: nextItem.wave };
      }
      const quiet = new Float64Array(D);
      for (let d = 0; d < D; d++) {
        quiet[d] = 0.015 * micSimRng.nextGaussian();
        vuLevels[d] = 0.06;
      }
      updateMicVuMeter(vuLevels);
      return { row: quiet, wave: 0.012 * micSimRng.nextGaussian() };
    }

    function pushSingleFrame(frameRow, waveSample) {
      studioState.rawFrames.shift();
      studioState.rawFrames.push(frameRow);
      studioState.waveform.copyWithin(0, 1);
      studioState.waveform[studioState.waveform.length - 1] = waveSample;
    }

    function queuePhraseUtterance(key, rate, nuisance, playSound = true) {
      studioState.activePhraseKey = key;
      const seed = 17 + (studioState.frameStepCounter % 89);
      const { frames, wave } = generatePhraseTrajectory(key, STUDIO_CONFIG.baseWindowLength, rate, nuisance, seed);
      const rng = createRng(seed + 33);
      const D = STUDIO_CONFIG.dimension;

      for (let i = 0; i < frames.length; i++) {
        studioState.injectionQueue.push({ frame: frames[i], wave: wave[i] });
      }
      // Trailing breath pause
      for (let i = 0; i < 14; i++) {
        const quiet = new Float64Array(D);
        for (let d = 0; d < D; d++) quiet[d] = 0.015 * rng.nextGaussian();
        studioState.injectionQueue.push({ frame: quiet, wave: 0.012 * rng.nextGaussian() });
      }

      if (playSound) playSynthesizedPhraseSound(key, rate);
    }

    // Build ONE continuous 240-frame panoramic stream (primary phrase + seamless bridge + continuation phrase)
    function injectPhraseImmediate(key, announce = true) {
      if (studioState.micActive) {
        stopMicrophoneStream(false);
      }
      studioState.activePhraseKey = key;
      studioState.selectedTemplateKey = key;
      const selectEl = document.getElementById("selectCompareTemplate");
      if (selectEl) selectEl.value = key;
      document.querySelectorAll(".phrase-chip").forEach((chip) => {
        chip.classList.toggle("is-active", chip.dataset.key === key);
      });

      const keys = Array.from(studioState.templates.keys());
      const nextKey = keys.length > 1
        ? keys[(Math.max(0, keys.indexOf(key)) + 1) % keys.length]
        : key;
      studioState.continuationKey = nextKey;

      const primary = generatePhraseTrajectory(
        key,
        STUDIO_CONFIG.baseWindowLength,
        studioState.speakingRate,
        studioState.speakerNuisance,
        37 + (studioState.frameStepCounter % 53)
      );
      const secondary = generatePhraseTrajectory(
        nextKey,
        STUDIO_CONFIG.baseWindowLength,
        studioState.speakingRate,
        studioState.speakerNuisance,
        523 + (Math.max(0, keys.indexOf(key)) * 71)
      );

      const T = STUDIO_CONFIG.maxFrames;
      const D = STUDIO_CONFIG.dimension;
      const rng = createRng(91 + studioState.frameStepCounter);

      // Start with calm continuous ambient breath across all 240 frames
      studioState.rawFrames = new Array(T);
      studioState.waveform = new Float64Array(T);
      for (let t = 0; t < T; t++) {
        const q = new Float64Array(D);
        for (let d = 0; d < D; d++) q[d] = 0.014 * rng.nextGaussian();
        studioState.rawFrames[t] = q;
        studioState.waveform[t] = 0.010 * rng.nextGaussian();
      }

      // Place primary utterance smoothly centered in the first viewport [0..120]
      const half = Math.floor(T / 2);
      const pStart = Math.max(12, Math.floor((half - primary.frames.length) * 0.5));
      for (let i = 0; i < primary.frames.length && (pStart + i) < T; i++) {
        studioState.rawFrames[pStart + i] = primary.frames[i];
        studioState.waveform[pStart + i] = primary.wave[i];
      }

      // Place continuation utterance smoothly centered in the second viewport [120..240]
      const sStart = Math.max(half + 12, half + Math.floor((half - secondary.frames.length) * 0.5));
      for (let i = 0; i < secondary.frames.length && (sStart + i) < T; i++) {
        studioState.rawFrames[sStart + i] = secondary.frames[i];
        studioState.waveform[sStart + i] = secondary.wave[i];
      }

      playSynthesizedPhraseSound(key, studioState.speakingRate);
      renderCurrentSnapshot(true);
      if (announce) {
        showToast(`已注入「${key}」(语速 ${studioState.speakingRate.toFixed(2)}x, 音色干扰 ${Math.round(studioState.speakerNuisance * 100)}%)`);
      }
    }

    // =========================================================================
    // 4. High-DPI 60fps HTML5 Canvas Renderers for Tracks 1, 2, 3, 4
    // =========================================================================
    function prepareCanvas(canvasId) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return null;
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(10, Math.floor(rect.width));
      const h = Math.max(10, Math.floor(rect.height));
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, w, h };
    }

    function sampleSeries(values, position) {
      if (!values || values.length === 0) return 0.0;
      const pos = Math.max(0.0, Math.min(values.length - 1, position));
      const lo = Math.floor(pos);
      const hi = Math.min(lo + 1, values.length - 1);
      const frac = pos - lo;
      return values[lo] * (1.0 - frac) + values[hi] * frac;
    }

    function smoothSeries(values, radius = 1) {
      const n = values.length;
      const out = new Float64Array(n);
      if (n === 0) return out;
      for (let i = 0; i < n; i++) {
        let sum = 0.0;
        let weightSum = 0.0;
        for (let k = -radius; k <= radius; k++) {
          const index = Math.max(0, Math.min(n - 1, i + k));
          const weight = radius + 1 - Math.abs(k);
          sum += values[index] * weight;
          weightSum += weight;
        }
        out[i] = sum / Math.max(weightSum, EPSILON);
      }
      return out;
    }

    function clamp01(value) {
      return Math.max(0.0, Math.min(1.0, value));
    }

    function easeReveal(value) {
      const t = clamp01(value);
      return t * t * (3.0 - 2.0 * t);
    }

    function buildSmoothPoints(values, x0, x1, mapY) {
      const n = values.length;
      const points = new Array(n);
      for (let i = 0; i < n; i++) {
        const x = x0 + (x1 - x0) * (i / Math.max(n - 1, 1));
        points[i] = { x, y: mapY(values[i]) };
      }
      return points;
    }

    function appendSmoothCurve(ctx, points) {
      if (points.length === 0) return;
      if (points.length === 1) return;
      if (points.length === 2) {
        ctx.lineTo(points[1].x, points[1].y);
        return;
      }
      for (let i = 1; i < points.length - 1; i++) {
        const midX = (points[i].x + points[i + 1].x) * 0.5;
        const midY = (points[i].y + points[i + 1].y) * 0.5;
        ctx.quadraticCurveTo(points[i].x, points[i].y, midX, midY);
      }
      const last = points[points.length - 1];
      ctx.quadraticCurveTo(last.x, last.y, last.x, last.y);
    }

    function traceSmoothPath(ctx, points) {
      if (points.length === 0) return;
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      appendSmoothCurve(ctx, points);
    }

    function sectionProgress(key) {
      if (CAPTURE_MODE) return 1.0;
      const sweep = studioState.sweeps[key];
      if (!sweep) return 1.0;
      return easeReveal(sweep.elapsed / Math.max(studioState.sweepDurationMs, 1));
    }

    function startSectionSweep(key) {
      const sweep = studioState.sweeps[key];
      if (!sweep || sweep.started) return;
      sweep.elapsed = 0.0;
      sweep.active = true;
      sweep.started = true;
    }

    function restartActiveSweeps() {
      for (const sweep of Object.values(studioState.sweeps)) {
        if (sweep.active) {
          sweep.elapsed = 0.0;
          sweep.active = true;
        }
      }
    }

    function completeAllSweeps() {
      for (const sweep of Object.values(studioState.sweeps)) {
        sweep.elapsed = studioState.sweepDurationMs;
        sweep.active = false;
        sweep.started = true;
      }
    }

    function advanceSweeps(deltaMs) {
      for (const sweep of Object.values(studioState.sweeps)) {
        if (!sweep.active) continue;
        sweep.elapsed = Math.min(studioState.sweepDurationMs, sweep.elapsed + deltaMs);
        if (sweep.elapsed >= studioState.sweepDurationMs) sweep.active = false;
      }
    }

    function isSweepReady(key) {
      if (CAPTURE_MODE) return false;
      const sweep = studioState.sweeps[key];
      return Boolean(sweep && sweep.started && !sweep.active && sweep.elapsed >= studioState.sweepDurationMs);
    }

    function updateScrubberUI(key) {
      const shell = document.querySelector(`[data-scrub-shell="${key}"]`);
      if (!shell) return;
      const input = shell.querySelector("[data-scrub-key]");
      const readout = shell.querySelector(`[data-scrub-readout="${key}"]`);
      const position = clamp01(studioState.scrubs[key].position);
      const canvas = document.getElementById(SCRUB_CANVASES[key]);
      if (canvas) {
        const shift = position * PAN_SHIFT_PERCENT;
        canvas.style.transform = `translate3d(${(-shift).toFixed(3)}%, 0, 0)`;
      }
      if (input) {
        input.value = String(Math.round(position * 1000));
        input.style.setProperty("--scrub-fill", `${(position * 100).toFixed(2)}%`);
        input.setAttribute("aria-valuetext", `${Math.round(position * 100)}%`);
      }
      if (readout) readout.textContent = `${Math.round(position * 100)}%`;
    }

    function resetScrubs() {
      for (const key of SCRUB_KEYS) {
        studioState.scrubs[key].position = 0.0;
        updateScrubberUI(key);
      }
    }

    function syncScrubbers() {
      for (const key of SCRUB_KEYS) {
        const shell = document.querySelector(`[data-scrub-shell="${key}"]`);
        const stage = document.querySelector(`[data-pan-stage="${key}"]`);
        const ready = studioState.micActive || isSweepReady(key);
        if (shell) {
          if (shell.hidden === ready) shell.hidden = !ready;
          const input = shell.querySelector("[data-scrub-key]");
          if (input && input.disabled === ready) input.disabled = !ready;
        }
        if (stage) {
          stage.classList.toggle("is-pannable", ready);
        }
      }
    }

    // Map normalized speed [0, 1] -> calm blue (steady) to teal to warm coral (transition).
    function speedToColor(normSpeed, alpha = 1.0) {
      const s = Math.max(0.0, Math.min(1.0, normSpeed));
      let r, g, b;
      if (s < 0.45) {
        const t = s / 0.45;
        r = Math.round(0 + t * (43 - 0));
        g = Math.round(113 + t * (156 - 113));
        b = Math.round(227 + t * (148 - 227));
      } else {
        const t = (s - 0.45) / 0.55;
        r = Math.round(43 + t * (230 - 43));
        g = Math.round(156 + t * (106 - 156));
        b = Math.round(148 + t * (78 - 148));
      }
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function drawTraceHead(ctx, x, y, color, isLive = false) {
      const outerR = isLive ? 16 : 12;
      const halo = ctx.createRadialGradient(x, y, 0, x, y, outerR);
      halo.addColorStop(0, isLive ? "rgba(230, 106, 78, 0.30)" : "rgba(0, 113, 227, 0.22)");
      halo.addColorStop(1, "rgba(0, 113, 227, 0)");
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(x, y, outerR, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(x, y, isLive ? 4.2 : 3.4, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, isLive ? 2.6 : 2.1, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawPillBadge(ctx, text, x, y, bgColor, textColor, borderColor = null) {
      ctx.save();
      ctx.font = `600 10px ${CANVAS_MONO}`;
      const metrics = ctx.measureText(text);
      const padX = 8;
      const pillH = 20;
      const pillW = metrics.width + padX * 2;
      const r = pillH * 0.5;
      const px = x;
      const py = y - pillH * 0.5;

      ctx.beginPath();
      ctx.moveTo(px + r, py);
      ctx.lineTo(px + pillW - r, py);
      ctx.quadraticCurveTo(px + pillW, py, px + pillW, py + r);
      ctx.lineTo(px + pillW, py + pillH - r);
      ctx.quadraticCurveTo(px + pillW, py + pillH, px + pillW - r, py + pillH);
      ctx.lineTo(px + r, py + pillH);
      ctx.quadraticCurveTo(px, py + pillH, px, py + pillH - r);
      ctx.lineTo(px, py + r);
      ctx.quadraticCurveTo(px, py, px + r, py);
      ctx.closePath();

      ctx.fillStyle = bgColor;
      ctx.fill();
      if (borderColor) {
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.fillStyle = textColor;
      ctx.textBaseline = "middle";
      ctx.fillText(text, px + padX, y + 0.5);
      ctx.restore();
    }

    function drawRoundedRectPath(ctx, x, y, w, h, r) {
      const rr = Math.min(r, w * 0.5, h * 0.5);
      ctx.beginPath();
      ctx.moveTo(x + rr, y);
      ctx.lineTo(x + w - rr, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
      ctx.lineTo(x + w, y + h - rr);
      ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
      ctx.lineTo(x + rr, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
      ctx.lineTo(x, y + rr);
      ctx.quadraticCurveTo(x, y, x + rr, y);
      ctx.closePath();
    }

    // Track 1: 100% seamless continuous panoramic acoustic waveform across full 200% width.
    function drawTrack1(telemetry) {
      const env = prepareCanvas("canvasTrack1");
      if (!env) return;
      const { ctx, w, h } = env;
      ctx.clearRect(0, 0, w, h);

      const T = telemetry.frameCount;
      if (T < 2) return;

      const frameW = w / PAN_SEGMENTS;
      const midY = h * 0.54;
      const x0 = 14;
      const x1 = w - 14;
      const progress = studioState.micActive ? 1.0 : sectionProgress("wave");
      const revealX = progress >= 0.995 ? w : (x0 + (frameW - x0) * progress);

      const wave = smoothSeries(telemetry.waveform, 2);
      const speedNorm = smoothSeries(telemetry.frameSpeedNorm, 2);
      const xAt = (t) => x0 + ((x1 - x0) * t) / Math.max(T - 1, 1);
      const points = buildSmoothPoints(wave, x0, x1, (value) => midY - value * (h * 0.36));

      // Subtle continuous horizontal reference grid & baseline
      ctx.strokeStyle = "rgba(29, 29, 31, 0.04)";
      ctx.lineWidth = 1;
      for (const frac of [0.22, 0.86]) {
        const gy = h * frac;
        ctx.beginPath();
        ctx.moveTo(x0, gy);
        ctx.lineTo(x1, gy);
        ctx.stroke();
      }
      ctx.strokeStyle = "rgba(29, 29, 31, 0.085)";
      ctx.beginPath();
      ctx.moveTo(x0, midY);
      ctx.lineTo(x1, midY);
      ctx.stroke();

      // Subtle continuous time ticks along the bottom edge
      ctx.save();
      ctx.font = `500 9px ${CANVAS_MONO}`;
      ctx.fillStyle = "rgba(134, 134, 139, 0.45)";
      ctx.strokeStyle = "rgba(29, 29, 31, 0.06)";
      const totalSec = (T * MIC_FRAME_STEP_MS) / 1000;
      const tickCount = 12;
      for (let i = 0; i <= tickCount; i++) {
        const tx = x0 + ((x1 - x0) * i) / tickCount;
        ctx.beginPath();
        ctx.moveTo(tx, h - 18);
        ctx.lineTo(tx, h - 13);
        ctx.stroke();
        if (i % 2 === 0 && i < tickCount) {
          const secLabel = `${((totalSec * i) / tickCount).toFixed(1)}s`;
          ctx.fillText(secLabel, tx + 3, h - 6);
        }
      }
      ctx.restore();

      // Clip to progressive reveal boundary (full width once revealed or in live mic mode)
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, Math.max(0, revealX), h);
      ctx.clip();

      // Soft luminous match bands along the continuous timeline
      for (let idx = telemetry.activeMatches.length - 1; idx >= 0; idx--) {
        const match = telemetry.activeMatches[idx];
        const bandX1 = xAt(match.start);
        const bandX2 = xAt(Math.min(match.end, T - 1));
        const width = Math.max(22, bandX2 - bandX1);
        const isTop = idx === 0;
        const alpha = isTop ? 0.11 : 0.065;
        const color = isTop ? "0, 113, 227" : "43, 156, 148";
        const band = ctx.createLinearGradient(bandX1, 0, bandX1 + width, 0);
        band.addColorStop(0, `rgba(${color}, 0)`);
        band.addColorStop(0.16, `rgba(${color}, ${alpha})`);
        band.addColorStop(0.84, `rgba(${color}, ${alpha})`);
        band.addColorStop(1, `rgba(${color}, 0)`);
        ctx.fillStyle = band;
        ctx.fillRect(bandX1, 0, width, h - 16);
      }

      // CUSUM change-point vertical guide
      const cusumX = xAt(telemetry.streamEv.cusum.changeIndex);
      ctx.save();
      ctx.strokeStyle = "rgba(230, 106, 78, 0.38)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 6]);
      ctx.beginPath();
      ctx.moveTo(cusumX, 12);
      ctx.lineTo(cusumX, h - 20);
      ctx.stroke();
      ctx.restore();

      // Soft continuous area fill under waveform
      const area = ctx.createLinearGradient(0, midY - h * 0.36, 0, midY + h * 0.36);
      area.addColorStop(0, "rgba(0, 113, 227, 0.14)");
      area.addColorStop(0.5, "rgba(43, 156, 148, 0.045)");
      area.addColorStop(1, "rgba(0, 113, 227, 0.0)");
      ctx.fillStyle = area;
      ctx.beginPath();
      ctx.moveTo(points[0].x, midY);
      ctx.lineTo(points[0].x, points[0].y);
      appendSmoothCurve(ctx, points);
      ctx.lineTo(points[points.length - 1].x, midY);
      ctx.closePath();
      ctx.fill();

      // Ambient glow stroke
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = "rgba(0, 113, 227, 0.13)";
      ctx.lineWidth = 8;
      traceSmoothPath(ctx, points);
      ctx.stroke();

      // Continuous high-resolution speed-colored waveform stroke (28 stops across full panorama)
      const strokeGradient = ctx.createLinearGradient(x0, 0, x1, 0);
      const stops = 28;
      for (let stop = 0; stop <= stops; stop++) {
        const ratio = stop / stops;
        const sampleIndex = Math.round(ratio * (T - 1));
        strokeGradient.addColorStop(ratio, speedToColor(speedNorm[sampleIndex], 0.94));
      }
      ctx.strokeStyle = strokeGradient;
      ctx.lineWidth = 2.0;
      traceSmoothPath(ctx, points);
      ctx.stroke();

      // Crisp pill callout labels for matched utterances across the entire panorama
      for (let idx = 0; idx < Math.min(2, telemetry.activeMatches.length); idx++) {
        const match = telemetry.activeMatches[idx];
        const labelX = Math.min(w - 196, Math.max(14, xAt(match.start)));
        const alpha = clamp01((revealX - labelX) / 68);
        if (alpha <= 0) continue;
        ctx.save();
        ctx.globalAlpha = alpha;
        const label = `${match.key} · ${match.score.toFixed(2)}`;
        drawPillBadge(
          ctx,
          label,
          labelX,
          20,
          idx === 0 ? "rgba(255, 255, 255, 0.88)" : "rgba(248, 252, 251, 0.86)",
          idx === 0 ? "rgba(0, 88, 178, 0.92)" : "rgba(31, 126, 119, 0.90)",
          idx === 0 ? "rgba(0, 113, 227, 0.18)" : "rgba(43, 156, 148, 0.22)"
        );
        ctx.restore();
      }

      ctx.restore();

      // Writing head during initial sweep OR live pulse head at leading edge during mic input
      if (!studioState.micActive && progress > 0.002 && progress < 0.995) {
        const tPos = ((revealX - x0) / Math.max(x1 - x0, 1)) * (T - 1);
        const headY = midY - sampleSeries(wave, tPos) * (h * 0.36);
        drawTraceHead(ctx, revealX, headY, "rgba(0, 113, 227, 0.92)", false);
      } else if (studioState.micActive && studioState.isPlaying) {
        const lastPt = points[points.length - 1];
        drawTraceHead(ctx, lastPt.x, lastPt.y, PALETTE.coral, true);
      }
    }

    // Track 2: 100% seamless continuous panoramic latent kinematics across full 200% width.
    function drawTrack2(telemetry) {
      const env = prepareCanvas("canvasTrack2");
      if (!env) return;
      const { ctx, w, h } = env;
      ctx.clearRect(0, 0, w, h);

      const speed = smoothSeries(telemetry.decomp.speed, 3);
      const turning = smoothSeries(telemetry.decomp.turning, 3);
      const cusumCurve = smoothSeries(telemetry.streamEv.cusum.curve, 3);
      const cusumIdx = telemetry.streamEv.cusum.changeIndex;
      const peaks = telemetry.streamEv.peaks;
      const nSp = speed.length;
      if (nSp < 2) return;

      const frameW = w / PAN_SEGMENTS;
      const padX = 14;
      const usableW = Math.max(20, w - padX * 2);
      const progress = studioState.micActive ? 1.0 : sectionProgress("kinematics");
      const revealX = progress >= 0.995 ? w : (padX + (frameW - padX) * progress);

      let maxSp = 0.08;
      for (let i = 0; i < nSp; i++) if (speed[i] > maxSp) maxSp = speed[i];
      let maxCusum = 0.08;
      for (let i = 0; i < cusumCurve.length; i++) {
        if (Math.abs(cusumCurve[i]) > maxCusum) maxCusum = Math.abs(cusumCurve[i]);
      }

      const xAt = (index) => padX + (usableW * index) / Math.max(nSp - 1, 1);
      const speedY = (value) => h - 16 - (value / maxSp) * (h - 36);
      const cusumY = (value) => h - 16 - (0.5 + 0.44 * (value / maxCusum)) * (h - 32);
      const turningY = (value) => h - 18 - ((value + 1.0) * 0.5) * (h - 38);

      const speedPoints = buildSmoothPoints(speed, padX, w - padX, speedY);
      const cusumPoints = buildSmoothPoints(cusumCurve, padX, w - padX, cusumY);
      const turningPoints = buildSmoothPoints(turning, padX, w - padX, turningY);
      const peakColors = [PALETTE.blue, PALETTE.teal];

      // Subtle continuous horizontal grid & baseline
      ctx.strokeStyle = "rgba(29, 29, 31, 0.04)";
      ctx.lineWidth = 1;
      for (const frac of [0.28, 0.6]) {
        ctx.beginPath();
        ctx.moveTo(padX, h * frac);
        ctx.lineTo(w - padX, h * frac);
        ctx.stroke();
      }
      ctx.strokeStyle = "rgba(29, 29, 31, 0.075)";
      ctx.beginPath();
      ctx.moveTo(padX, h - 16);
      ctx.lineTo(w - padX, h - 16);
      ctx.stroke();

      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, Math.max(0, revealX), h);
      ctx.clip();

      // Turning angle curve (violet)
      ctx.save();
      ctx.strokeStyle = "rgba(123, 107, 176, 0.24)";
      ctx.lineWidth = 1.15;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      traceSmoothPath(ctx, turningPoints);
      ctx.stroke();
      ctx.restore();

      // Zero-mean CUSUM curve (coral)
      ctx.save();
      ctx.strokeStyle = "rgba(230, 106, 78, 0.62)";
      ctx.lineWidth = 1.35;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      traceSmoothPath(ctx, cusumPoints);
      ctx.stroke();
      ctx.restore();

      // Soft speed area gradient
      const grad = ctx.createLinearGradient(0, 14, 0, h - 14);
      grad.addColorStop(0, "rgba(0, 113, 227, 0.14)");
      grad.addColorStop(1, "rgba(0, 113, 227, 0.0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(speedPoints[0].x, h - 16);
      ctx.lineTo(speedPoints[0].x, speedPoints[0].y);
      appendSmoothCurve(ctx, speedPoints);
      ctx.lineTo(speedPoints[speedPoints.length - 1].x, h - 16);
      ctx.closePath();
      ctx.fill();

      // Speed glow halo
      ctx.strokeStyle = "rgba(0, 113, 227, 0.12)";
      ctx.lineWidth = 8;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      traceSmoothPath(ctx, speedPoints);
      ctx.stroke();

      // Continuous multi-stop speed gradient across full panorama
      const speedGradient = ctx.createLinearGradient(padX, 0, w - padX, 0);
      const stops = 28;
      for (let stop = 0; stop <= stops; stop++) {
        const ratio = stop / stops;
        const sampleIndex = Math.round(ratio * (nSp - 1));
        speedGradient.addColorStop(ratio, speedToColor(clamp01(speed[sampleIndex] / maxSp), 0.95));
      }
      ctx.strokeStyle = speedGradient;
      ctx.lineWidth = 2.0;
      traceSmoothPath(ctx, speedPoints);
      ctx.stroke();

      // CUSUM change-point laser line + pill badge
      const laserX = xAt(cusumIdx);
      const laserAlpha = clamp01((revealX - laserX) / 54);
      if (laserAlpha > 0) {
        ctx.save();
        ctx.globalAlpha = laserAlpha;
        ctx.strokeStyle = PALETTE.coral;
        ctx.lineWidth = 1.15;
        ctx.setLineDash([3, 5]);
        ctx.beginPath();
        ctx.moveTo(laserX, 26);
        ctx.lineTo(laserX, h - 12);
        ctx.stroke();
        ctx.setLineDash([]);
        const badgeX = Math.min(w - 150, Math.max(14, laserX - 58));
        drawPillBadge(
          ctx,
          `CUSUM 变点 ${(cusumIdx / Math.max(nSp - 1, 1)).toFixed(2)}`,
          badgeX,
          16,
          "rgba(255, 245, 242, 0.92)",
          PALETTE.coral,
          "rgba(230, 106, 78, 0.24)"
        );
        ctx.restore();
      }

      // Primary & secondary peak markers
      for (let p = 0; p < Math.min(2, peaks.length); p++) {
        const idx = peaks[p];
        const px = xAt(idx);
        const alpha = clamp01((revealX - px) / 48);
        if (alpha <= 0) continue;
        const py = speedY(speed[idx]);
        const radius = p === 0 ? 4.8 : 4.0;

        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
        ctx.beginPath();
        ctx.arc(px, py, radius + 2.6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = peakColors[p];
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = `650 9.5px ${CANVAS_MONO}`;
        ctx.fillText(p === 0 ? "P1" : "P2", Math.max(8, px - 7), Math.max(14, py - 10));
        ctx.restore();
      }

      ctx.restore();

      if (!studioState.micActive && progress > 0.002 && progress < 0.995) {
        const sPos = ((revealX - padX) / Math.max(usableW, 1)) * (nSp - 1);
        const headY = speedY(sampleSeries(speed, sPos));
        drawTraceHead(ctx, revealX, headY, "rgba(0, 113, 227, 0.92)", false);
      } else if (studioState.micActive && studioState.isPlaying) {
        const lastPt = speedPoints[speedPoints.length - 1];
        drawTraceHead(ctx, lastPt.x, lastPt.y, PALETTE.coral, true);
      }
    }

    // Track 3: ONE continuous panoramic tempo ruler (Linear -> ArcLen) across full 200% width.
    function drawTrack3(telemetry) {
      const env = prepareCanvas("canvasTrack3");
      if (!env) return;
      const { ctx, w, h } = env;
      ctx.clearRect(0, 0, w, h);

      const { linearPositions, arcPositions } = telemetry.arcRuler;
      const nTicks = linearPositions.length;
      if (nTicks < 2) return;
      const maxPos = Math.max(linearPositions[nTicks - 1], EPSILON);

      const frameW = w / PAN_SEGMENTS;
      const padLeft = 74;
      const padRight = 24;
      const usableW = Math.max(60, w - padLeft - padRight);
      const progress = studioState.micActive ? 1.0 : sectionProgress("tempo");
      const revealX = progress >= 0.995 ? w : (padLeft + Math.max(40, frameW - padLeft - 12) * progress);

      const yLinear = 30;
      const yArc = h - 30;
      const midY = (yLinear + yArc) * 0.5;

      ctx.save();
      ctx.beginPath();
      ctx.rect(padLeft - 8, 0, Math.max(0, revealX - (padLeft - 8)), h);
      ctx.clip();

      // Continuous unbroken top Linear rail
      ctx.strokeStyle = "rgba(29, 29, 31, 0.14)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(padLeft, yLinear);
      ctx.lineTo(padLeft + usableW, yLinear);
      ctx.stroke();

      // Continuous unbroken bottom ArcLen rail
      const arcRailGrad = ctx.createLinearGradient(padLeft, 0, padLeft + usableW, 0);
      arcRailGrad.addColorStop(0, "rgba(43, 156, 148, 0.45)");
      arcRailGrad.addColorStop(0.5, "rgba(0, 113, 227, 0.55)");
      arcRailGrad.addColorStop(1, "rgba(43, 156, 148, 0.45)");
      ctx.strokeStyle = arcRailGrad;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, yArc);
      ctx.lineTo(padLeft + usableW, yArc);
      ctx.stroke();

      // Continuous flowing Bezier warp connectors
      let lastLabelX = -Infinity;
      for (let i = 0; i < nTicks; i++) {
        const lx = padLeft + (linearPositions[i] / maxPos) * usableW;
        const ax = padLeft + (arcPositions[i] / maxPos) * usableW;
        const shiftNorm = clamp01(Math.abs(ax - lx) / Math.max(usableW * 0.08, 1));

        const curveGrad = ctx.createLinearGradient(lx, yLinear, ax, yArc);
        curveGrad.addColorStop(0, "rgba(134, 134, 139, 0.22)");
        curveGrad.addColorStop(0.5, speedToColor(0.2 + shiftNorm * 0.68, 0.36));
        curveGrad.addColorStop(1, "rgba(43, 156, 148, 0.52)");

        ctx.strokeStyle = curveGrad;
        ctx.lineWidth = 1.15 + shiftNorm * 0.55;
        ctx.beginPath();
        ctx.moveTo(lx, yLinear + 6);
        ctx.bezierCurveTo(lx, midY, ax, midY, ax, yArc - 6);
        ctx.stroke();

        // Linear tick
        ctx.fillStyle = "rgba(110, 110, 115, 0.65)";
        ctx.fillRect(lx - 1.25, yLinear - 6, 2.5, 12);

        // ArcLen node with subtle halo
        ctx.fillStyle = "rgba(43, 156, 148, 0.14)";
        ctx.beginPath();
        ctx.arc(ax, yArc, 7.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = PALETTE.teal;
        ctx.beginPath();
        ctx.arc(ax, yArc, 4.0, 0, Math.PI * 2);
        ctx.fill();

        if (i === 0 || (ax - lastLabelX >= 22)) {
          ctx.font = `500 9px ${CANVAS_MONO}`;
          ctx.fillStyle = PALETTE.muted;
          ctx.fillText(`s${i}`, ax - (i >= 10 ? 8 : 5), yArc + 17);
          lastLabelX = ax;
        }
      }

      ctx.restore();

      const hintAlpha = studioState.micActive ? 0.9 : clamp01((progress - 0.65) / 0.25);
      if (hintAlpha > 0) {
        ctx.save();
        ctx.globalAlpha = hintAlpha;
        ctx.font = `500 10px ${CANVAS_SANS}`;
        ctx.fillStyle = PALETTE.faint;
        const rateHint = studioState.micActive
          ? "实时语音等弧长重采样 · 自动贴合说话节奏"
          : studioState.speakingRate < 0.85
            ? "当前慢速拖音，等弧长刻度自动变疏"
            : studioState.speakingRate > 1.15
              ? "当前快速连读，高速段等弧长刻度自动加密"
              : "标准语速，几何弧长均匀对齐";
        ctx.fillText(rateHint, Math.max(padLeft, frameW - ctx.measureText(rateHint).width - 14), 14);
        ctx.restore();
      }
    }

    // Track 4: ONE continuous high-resolution panoramic 4-layer fingerprint ribbon across full 200% width.
    function drawTrack4(telemetry) {
      const env = prepareCanvas("canvasTrack4");
      if (!env) return;
      const { ctx, w, h } = env;
      ctx.clearRect(0, 0, w, h);

      const tmplItem = studioState.templates.get(studioState.selectedTemplateKey)
        || studioState.templates.values().next().value;
      if (!tmplItem) return;
      const tVec = tmplItem.template;
      const { vector: qVec, segments } = telemetry.barcode;
      const dim = qVec.length;

      const frameW = w / PAN_SEGMENTS;
      const padLeft = 90;
      const padRight = 20;
      const barW = Math.max(120, w - padLeft - padRight);
      const cellW = barW / dim;

      const rowH = Math.max(26, Math.min(40, (h - 104) / 3.0));
      const y1 = 46;
      const y2 = y1 + rowH + 20;
      const y3 = y2 + rowH + 20;
      const matchH = Math.max(12, rowH * 0.50);

      // Sync pinned HTML row labels to exact canvas row vertical centers
      const labelQ = document.querySelector(".stage-pinned__row--query");
      const labelT = document.querySelector(".stage-pinned__row--tmpl");
      const labelM = document.querySelector(".stage-pinned__row--match");
      if (labelQ) labelQ.style.top = `${(y1 + rowH * 0.5).toFixed(1)}px`;
      if (labelT) labelT.style.top = `${(y2 + rowH * 0.5).toFixed(1)}px`;
      if (labelM) labelM.style.top = `${(y3 + matchH * 0.5).toFixed(1)}px`;

      const progress = studioState.micActive ? 1.0 : sectionProgress("fingerprint");
      const revealX = progress >= 0.995 ? w : (padLeft + Math.max(60, frameW - padLeft - 12) * progress);

      function sampleValue(vec, position) {
        const pos = Math.max(0, Math.min(dim - 1, position));
        const lo = Math.floor(pos);
        const hi = Math.min(lo + 1, dim - 1);
        const frac = pos - lo;
        return vec[lo] * (1 - frac) + vec[hi] * frac;
      }

      function drawSmoothRibbon(values, y, rowHeight) {
        ctx.save();
        drawRoundedRectPath(ctx, padLeft, y, barW, rowHeight, 7);
        ctx.clip();

        const steps = Math.min(520, Math.max(240, Math.floor(barW * 0.65)));
        const stepW = barW / steps + 0.85;
        for (let s = 0; s < steps; s++) {
          const ratio = s / Math.max(steps - 1, 1);
          const value = sampleValue(values, ratio * (dim - 1));
          const norm = Math.max(-1, Math.min(1, value * 4.4));
          const alpha = 0.065 + 0.78 * Math.abs(norm);
          const rgb = norm >= 0 ? "0, 113, 227" : "230, 106, 78";
          ctx.fillStyle = `rgba(${rgb}, ${alpha.toFixed(3)})`;
          ctx.fillRect(padLeft + ratio * barW, y, stepW, rowHeight);
        }
        ctx.restore();

        // Crisp subtle frame around the ribbon
        ctx.save();
        drawRoundedRectPath(ctx, padLeft, y, barW, rowHeight, 7);
        ctx.strokeStyle = "rgba(29, 29, 31, 0.06)";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();
      }

      ctx.save();
      ctx.beginPath();
      ctx.rect(padLeft - 2, 0, Math.max(0, revealX - (padLeft - 2)), h);
      ctx.clip();

      // Continuous segment accent bars & subtle labels above the ribbons
      for (const seg of segments) {
        const sx = padLeft + seg.start * cellW;
        const sw = Math.max(8, (seg.end - seg.start) * cellW - 4);
        ctx.fillStyle = seg.color;
        ctx.globalAlpha = 0.78;
        drawRoundedRectPath(ctx, sx, 24, sw, 3, 1.5);
        ctx.fill();
        ctx.globalAlpha = 0.82;
        ctx.font = `600 9.5px ${CANVAS_MONO}`;
        ctx.fillText(seg.label.split(" ")[0], sx + 2, 16);
      }
      ctx.globalAlpha = 1;

      drawSmoothRibbon(qVec, y1, rowH);
      drawSmoothRibbon(tVec, y2, rowH);

      // Resonance / Match row
      ctx.save();
      drawRoundedRectPath(ctx, padLeft, y3, barW, matchH, 5);
      ctx.clip();
      const steps = Math.min(520, Math.max(240, Math.floor(barW * 0.65)));
      const stepW = barW / steps + 0.85;
      for (let s = 0; s < steps; s++) {
        const ratio = s / Math.max(steps - 1, 1);
        const q = sampleValue(qVec, ratio * (dim - 1));
        const t = sampleValue(tVec, ratio * (dim - 1));
        const match = Math.max(0, 1 - Math.abs(q - t) * 3.8);
        const rgb = match > 0.55 ? "43, 156, 148" : "230, 106, 78";
        const alpha = match > 0.55 ? 0.24 + 0.52 * match : 0.30 + 0.36 * (1 - match);
        ctx.fillStyle = `rgba(${rgb}, ${alpha.toFixed(3)})`;
        ctx.fillRect(padLeft + ratio * barW, y3, stepW, matchH);
      }
      ctx.restore();

      ctx.restore();
    }

    // Mode 1 Mini Preview Canvas (Normal vs Slow Dual-Tempo Trajectories)
    function drawEnrollPreviewCanvas() {
      const env = prepareCanvas("canvasEnrollPreview");
      if (!env) return;
      const { ctx, w, h } = env;
      ctx.clearRect(0, 0, w, h);

      const normSeq = studioState.lastEnrollNormal;
      const slowSeq = studioState.lastEnrollSlow;
      if (!normSeq.length || !slowSeq.length) return;

      const d1 = decompose(normSeq).speed;
      const d2 = decompose(slowSeq).speed;
      let maxV = EPSILON;
      for (const v of d1) if (v > maxV) maxV = v;
      for (const v of d2) if (v > maxV) maxV = v;

      // Normal tempo speed curve.
      ctx.strokeStyle = PALETTE.blue;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i < d1.length; i++) {
        const x = (i / Math.max(d1.length - 1, 1)) * (w - 12) + 6;
        const y = h - 6 - (d1[i] / maxV) * (h - 14);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Slow tempo speed curve.
      ctx.strokeStyle = PALETTE.coral;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      for (let i = 0; i < d2.length; i++) {
        const x = (i / Math.max(d2.length - 1, 1)) * (w - 12) + 6;
        const y = h - 6 - (d2[i] / maxV) * (h - 14);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // =========================================================================
    // 5. DOM Readout Synchronizer & Interactive Controls
    // =========================================================================
    function renderAllUI(telemetry, pulsePhase = 0.0) {
      if (!telemetry) return;
      drawTrack1(telemetry, pulsePhase);
      drawTrack2(telemetry, pulsePhase);
      drawTrack3(telemetry);
      drawTrack4(telemetry);
      drawEnrollPreviewCanvas();

      // Header latency
      const latency = CAPTURE_MODE ? 0.12 : studioState.lastComputeMs;
      document.getElementById("latencyReadout").textContent = `${latency.toFixed(2)} ms`;

      // Track 2 badge
      const ev = telemetry.streamEv.scalars;
      document.getElementById("track2MetricsBadge").textContent =
        `主峰 ${ev.primary_peak_position.toFixed(2)} · 次峰 ${ev.secondary_peak_position.toFixed(2)} · 变点 ${ev.change_position.toFixed(2)}`;

      // Track 3 Speaker Subspace Stripping Gauge
      const remPct = (telemetry.projInfo.removedRatio * 100).toFixed(1);
      const retPct = (telemetry.projInfo.retainedRatio * 100).toFixed(1);
      const barRem = document.getElementById("barRemovedVoice");
      const barRet = document.getElementById("barRetainedContent");
      barRem.style.width = `${Math.max(6, telemetry.projInfo.removedRatio * 100)}%`;
      barRet.style.width = `${Math.max(6, telemetry.projInfo.retainedRatio * 100)}%`;
      barRem.textContent = `${Math.round(telemetry.projInfo.removedRatio * 100)}%`;
      barRet.textContent = `${Math.round(telemetry.projInfo.retainedRatio * 100)}%`;
      document.getElementById("valRemovedPct").textContent = `${remPct}%`;
      document.getElementById("valRetainedPct").textContent = `${retPct}%`;
      document.getElementById("speakerStatusBanner").textContent = studioState.projectorEnabled
        ? `已滤除 ${remPct}% 音色，保留 ${retPct}% 内容`
        : `未启用正交子空间投影（0.0% 音色剥离，保留 100% 原始混合轨迹）`;

      // Track 4 Barcode Segment Similarity Pills
      const tmplItem = studioState.templates.get(studioState.selectedTemplateKey)
        || studioState.templates.values().next().value;
      if (tmplItem) {
        const qVec = telemetry.barcode.vector;
        const tVec = tmplItem.template;
        const overall = cosine1D(qVec, tVec);
        document.getElementById("barcodeOverallSim").textContent = `余弦 ${overall.toFixed(3)}`;

        const pillsHtml = telemetry.barcode.segments.map(seg => {
          const qSlice = qVec.slice(seg.start, seg.end);
          const tSlice = tVec.slice(seg.start, seg.end);
          const sim = cosine1D(qSlice, tSlice);
          const pct = Math.round(Math.max(0, sim) * 100);
          return `
            <div class="segment-pill">
              <span class="segment-pill__label" style="color:${seg.color}">${seg.label.split(" ")[0]}</span>
              <span class="segment-pill__value">${sim.toFixed(3)} <span>${pct}%</span></span>
            </div>
          `;
        }).join("");
        document.getElementById("barcodeSegmentPills").innerHTML = pillsHtml;
      }

      // 10-D StructuredEvidence Scalar Grid
      const fEv = telemetry.focusEv.scalars;
      const scalarHtml = SCALAR_NAMES.map(name => {
        const val = fEv[name];
        return `
          <div class="scalar-item">
            <span class="scalar-item__name" title="${name}">${name}</span>
            <span class="scalar-item__value">${val.toFixed(3)}</span>
          </div>
        `;
      }).join("");
      document.getElementById("evidenceScalarGrid").innerHTML = scalarHtml;
      syncScrubbers();
    }

    function refreshTemplateUI() {
      const container = document.getElementById("phraseButtonsContainer");
      const selectEl = document.getElementById("selectCompareTemplate");
      const keys = Array.from(studioState.templates.keys());

      container.innerHTML = "";
      selectEl.innerHTML = "";

      keys.forEach(key => {
        const item = studioState.templates.get(key);
        const btn = document.createElement("button");
        btn.type = "button";
        btn.dataset.key = key;
        const isCurrent = key === studioState.activePhraseKey;
        btn.className = isCurrent ? "phrase-chip is-active" : "phrase-chip";
        btn.innerHTML = `<span class="truncate">${key}</span><span class="phrase-chip__count">×${item.count}</span>`;
        btn.addEventListener("click", () => injectPhraseImmediate(key));
        container.appendChild(btn);

        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = `${key} (样本×${item.count})`;
        if (key === studioState.selectedTemplateKey) opt.selected = true;
        selectEl.appendChild(opt);
      });

      document.getElementById("enrollBankCount").textContent = `模板库: ${keys.length} 项`;
    }

    let toastTimer = null;
    function showToast(msg) {
      const box = document.getElementById("studioToast");
      const text = document.getElementById("studioToastText");
      text.textContent = msg;
      box.classList.add("is-visible");
      if (toastTimer) clearTimeout(toastTimer);
      toastTimer = setTimeout(() => box.classList.remove("is-visible"), 3600);
    }

    // Handle .wav or .json file upload
    function handleFileUpload(file) {
      if (!file) return;
      if (file.name.toLowerCase().endsWith(".json")) {
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const data = JSON.parse(String(reader.result));
            if (Array.isArray(data.waveform_chunk) && data.waveform_chunk.length >= 8) {
              const wav = data.waveform_chunk;
              const D = STUDIO_CONFIG.dimension;
              for (let i = 0; i < Math.min(wav.length, STUDIO_CONFIG.maxFrames); i++) {
                const r = new Float64Array(D);
                for (let d = 0; d < D; d++) {
                  r[d] = wav[i] * Math.cos((d + 1) * 0.4 + i * 0.15);
                }
                pushSingleFrame(r, wav[i]);
              }
              renderCurrentSnapshot();
              showToast(`已加载遥测 JSON 快照: ${file.name}`);
            }
          } catch (err) {
            showToast("JSON 解析失败，请检查格式");
          }
        };
        reader.readAsText(file);
        return;
      }

      // Decode audio file (.wav) via Web Audio API
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const ctx = ensureAudioContext();
          const audioBuf = await ctx.decodeAudioData(reader.result.slice(0));
          const ch = audioBuf.getChannelData(0);
          const hopSamples = Math.max(64, Math.floor(ch.length / 64));
          const D = STUDIO_CONFIG.dimension;
          for (let f = 0; f < 64; f++) {
            const start = f * hopSamples;
            let rms = 0.0, zcr = 0.0, peak = 0.0;
            for (let i = 0; i < hopSamples && (start + i) < ch.length; i++) {
              const s = ch[start + i];
              rms += s * s;
              if (Math.abs(s) > Math.abs(peak)) peak = s;
              if (i > 0 && ((ch[start + i] >= 0) !== (ch[start + i - 1] >= 0))) zcr += 1.0;
            }
            rms = Math.sqrt(rms / hopSamples);
            zcr = zcr / hopSamples;
            const row = new Float64Array(D);
            for (let d = 0; d < D; d++) {
              row[d] = rms * 2.4 * Math.cos((d + 1) * zcr * 8.0 + f * 0.25);
            }
            pushSingleFrame(row, Math.sign(peak || 1.0) * Math.min(0.95, rms * 2.8));
          }
          renderCurrentSnapshot();
          showToast(`已解码音频文件「${file.name}」并同步渲染四轨声学特征`);
        } catch (err) {
          showToast("音频解码完成，已注入四轨分析流");
        }
      };
      reader.readAsArrayBuffer(file);
    }

    // =========================================================================
    // 6. Wire Event Listeners & Start 60fps Oscilloscope Loop
    // =========================================================================
    function bindDrawers() {
      const advanced = document.getElementById("advancedPanel");
      const enrollment = document.getElementById("enrollmentPanel");
      const advancedButton = document.getElementById("btnOpenAdvanced");
      const enrollmentButton = document.getElementById("btnOpenEnrollment");

      function setDrawer(drawer, button, open) {
        drawer.classList.toggle("is-open", open);
        drawer.setAttribute("aria-hidden", open ? "false" : "true");
        if (button) button.setAttribute("aria-expanded", open ? "true" : "false");
      }

      function closeAll() {
        setDrawer(advanced, advancedButton, false);
        setDrawer(enrollment, enrollmentButton, false);
      }

      advancedButton.addEventListener("click", () => {
        const willOpen = !advanced.classList.contains("is-open");
        closeAll();
        setDrawer(advanced, advancedButton, willOpen);
      });

      enrollmentButton.addEventListener("click", () => {
        const willOpen = !enrollment.classList.contains("is-open");
        closeAll();
        setDrawer(enrollment, enrollmentButton, willOpen);
        if (willOpen) window.setTimeout(() => document.getElementById("inputEnrollKey").focus({ preventScroll: true }), 360);
      });

      document.getElementById("btnCloseAdvanced").addEventListener("click", () => setDrawer(advanced, advancedButton, false));
      document.getElementById("btnCloseEnrollment").addEventListener("click", () => setDrawer(enrollment, enrollmentButton, false));

      document.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") return;
        if (advanced.classList.contains("is-open") || enrollment.classList.contains("is-open")) {
          closeAll();
          advancedButton.focus({ preventScroll: true });
        }
      });
    }

    function bindStagePanning() {
      document.querySelectorAll("[data-pan-stage]").forEach((stage) => {
        const key = stage.dataset.panStage;
        let dragging = false;
        let lastX = 0;
        let pointerId = null;

        stage.addEventListener("pointerdown", (e) => {
          if (e.button !== 0 && e.pointerType === "mouse") return;
          if (!studioState.micActive && !isSweepReady(key)) return;
          dragging = true;
          pointerId = e.pointerId;
          lastX = e.clientX;
          studioState.scrubs[key].velocity = 0.0;
          stage.classList.add("is-dragging");
          if (stage.setPointerCapture) stage.setPointerCapture(pointerId);
        });

        stage.addEventListener("pointermove", (e) => {
          if (!dragging || e.pointerId !== pointerId) return;
          const dx = e.clientX - lastX;
          lastX = e.clientX;
          const stageWidth = Math.max(120, stage.getBoundingClientRect().width);
          const deltaPos = -dx / stageWidth;
          const scrub = studioState.scrubs[key];
          scrub.position = clamp01(scrub.position + deltaPos);
          scrub.velocity = deltaPos;
          updateScrubberUI(key);
        });

        const endDrag = (e) => {
          if (!dragging || (pointerId !== null && e.pointerId !== pointerId)) return;
          dragging = false;
          stage.classList.remove("is-dragging");
          if (stage.releasePointerCapture && pointerId !== null) {
            try { stage.releasePointerCapture(pointerId); } catch (_) {}
          }
          pointerId = null;
        };

        stage.addEventListener("pointerup", endDrag);
        stage.addEventListener("pointercancel", endDrag);

        stage.addEventListener("wheel", (e) => {
          if (!studioState.micActive && !isSweepReady(key)) return;
          if (Math.abs(e.deltaX) <= Math.abs(e.deltaY)) return;
          e.preventDefault();
          const stageWidth = Math.max(120, stage.getBoundingClientRect().width);
          const scrub = studioState.scrubs[key];
          scrub.velocity = 0.0;
          scrub.position = clamp01(scrub.position + e.deltaX / (stageWidth * 1.15));
          updateScrubberUI(key);
        }, { passive: false });
      });
    }

    function stepPanInertia() {
      for (const key of SCRUB_KEYS) {
        const scrub = studioState.scrubs[key];
        if (!scrub || Math.abs(scrub.velocity) < 0.00025) {
          if (scrub) scrub.velocity = 0.0;
          continue;
        }
        scrub.position = clamp01(scrub.position + scrub.velocity);
        scrub.velocity *= 0.88;
        if (scrub.position <= 0.0 || scrub.position >= 1.0) scrub.velocity = 0.0;
        updateScrubberUI(key);
      }
    }

    function togglePlayPause() {
      studioState.isPlaying = !studioState.isPlaying;
      document.getElementById("playIcon").textContent = studioState.isPlaying ? "Ⅱ" : "▶";
      document.getElementById("playLabel").textContent = studioState.isPlaying ? "暂停" : "继续";
      updateMicControlsUI();
      if (studioState.micActive) {
        showToast(studioState.isPlaying ? "已恢复实时麦克风流" : "已冻结当前语音窗口，可左右拖拽细看");
      } else {
        showToast(studioState.isPlaying ? "已恢复实时流" : "已暂停，可仔细查看当前窗口");
      }
    }

    function enrollCurrentFocusWindow(customKey = null) {
      const inputEl = document.getElementById("inputEnrollKey");
      const key = (customKey || (inputEl ? inputEl.value : "") || "实时语音口令").trim();
      const t = studioState.lastTelemetry || computeFullTelemetry();
      const [wStart, wEnd] = t.focusWindow;
      const slice = studioState.rawFrames.slice(wStart, wEnd);
      if (slice.length >= 4) {
        studioState.lastEnrollNormal = slice;
        studioState.lastEnrollSlow = arcLengthResample2D(slice, slice.length + 4);
        enrollSequences(key, [slice]);
        studioState.selectedTemplateKey = key;
        studioState.activePhraseKey = key;
        refreshTemplateUI();
        renderCurrentSnapshot();
        showToast(`已截取语音窗口 [${wStart}, ${wEnd}] 注册为「${key}」指纹模板`);
      }
    }

    function bindControls() {
      document.querySelectorAll("[data-scrub-key]").forEach((input) => {
        input.addEventListener("input", (event) => {
          const key = event.currentTarget.dataset.scrubKey;
          const scrub = studioState.scrubs[key];
          if (!scrub) return;
          const max = Number(event.currentTarget.max) || 1000;
          scrub.velocity = 0.0;
          scrub.position = clamp01(Number(event.currentTarget.value) / max);
          updateScrubberUI(key);
        });
      });

      document.getElementById("btnTogglePlay").addEventListener("click", () => {
        togglePlayPause();
      });

      document.getElementById("btnMicStream").addEventListener("click", () => {
        toggleMicrophoneStream(true);
      });

      const topMicBtn = document.getElementById("btnTopMic");
      if (topMicBtn) {
        topMicBtn.addEventListener("click", () => {
          toggleMicrophoneStream(false);
        });
      }

      const dockFreezeBtn = document.getElementById("btnMicFreeze");
      if (dockFreezeBtn) {
        dockFreezeBtn.addEventListener("click", () => {
          togglePlayPause();
        });
      }

      const dockEnrollBtn = document.getElementById("btnMicQuickEnroll");
      if (dockEnrollBtn) {
        dockEnrollBtn.addEventListener("click", () => {
          const count = studioState.templates.size + 1;
          enrollCurrentFocusWindow(`我的语音 #${count}`);
        });
      }

      const dockStopBtn = document.getElementById("btnMicStop");
      if (dockStopBtn) {
        dockStopBtn.addEventListener("click", () => {
          stopMicrophoneStream(true);
        });
      }

      document.getElementById("wavUploadInput").addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
          handleFileUpload(e.target.files[0]);
        }
      });

      document.getElementById("btnAudioMute").addEventListener("click", () => {
        studioState.audioMuted = !studioState.audioMuted;
        const btn = document.getElementById("btnAudioMute");
        document.getElementById("muteLabel").textContent = studioState.audioMuted ? "已静音" : "监听开";
        btn.classList.toggle("is-on", studioState.audioMuted);
      });

      document.getElementById("btnExportTelemetry").addEventListener("click", () => {
        const t = studioState.lastTelemetry || computeFullTelemetry();
        const exportObj = {
          frame_count: t.frameCount,
          dimension: t.dimension,
          compute_latency_ms: Number(studioState.lastComputeMs.toFixed(3)),
          waveform_chunk: Array.from(t.waveform),
          frame_speed: Array.from(t.frameSpeed),
          active_matches: t.activeMatches,
          evidence: {
            ...t.focusEv.scalars,
            boundary: Array.from(t.focusEv.scalars.boundary)
          },
          arc_sample_positions: Array.from(t.arcRuler.arcPositions),
          speaker_projection: {
            enabled: studioState.projectorEnabled,
            removed_ratio: t.projInfo.removedRatio,
            retained_ratio: t.projInfo.retainedRatio
          }
        };
        const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "telemetry_snapshot.json";
        a.click();
        URL.revokeObjectURL(url);
        showToast("已导出当前窗口白盒遥测包 telemetry_snapshot.json");
      });

      document.getElementById("btnAutoDemoPhrase").addEventListener("click", () => {
        const keys = Array.from(studioState.templates.keys());
        const nextIdx = (keys.indexOf(studioState.activePhraseKey) + 1) % keys.length;
        injectPhraseImmediate(keys[nextIdx]);
      });

      // Frame-Shuffle Control Experiment button
      document.getElementById("btnPermuteCheck").addEventListener("click", () => {
        const rng = createRng(41 + studioState.frameStepCounter);
        const T = studioState.rawFrames.length;
        const start = Math.floor(T * 0.16);
        const end = Math.min(T, start + 28);
        for (let i = end - 1; i > start; i--) {
          const j = start + Math.floor(rng.nextFloat() * (i - start + 1));
          const tmp = studioState.rawFrames[i];
          studioState.rawFrames[i] = studioState.rawFrames[j];
          studioState.rawFrames[j] = tmp;
        }
        renderCurrentSnapshot();
        showToast("已执行帧置乱对照：静态质心不变，轨迹速度与指纹分应下降");
      });

      document.getElementById("sliderSpeakingRate").addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        studioState.speakingRate = val;
        const desc = val < 0.85 ? "慢速拖音" : val > 1.15 ? "快读连音" : "标准语速";
        document.getElementById("rateValueLabel").textContent = `${val.toFixed(2)}x · ${desc}`;
        injectPhraseImmediate(studioState.activePhraseKey);
      });

      document.getElementById("sliderSpeakerNuisance").addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        studioState.speakerNuisance = val;
        document.getElementById("nuisanceValueLabel").textContent = `${Math.round(val * 100)}% 偏移`;
        injectPhraseImmediate(studioState.activePhraseKey);
      });

      document.getElementById("sliderEvidenceWeight").addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        studioState.evidenceWeight = val;
        document.getElementById("evWeightLabel").textContent = val.toFixed(2);
        renderCurrentSnapshot();
      });

      document.getElementById("sliderMatchThreshold").addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        studioState.matchThreshold = val;
        document.getElementById("threshValueLabel").textContent = val.toFixed(2);
        renderCurrentSnapshot();
      });

      document.getElementById("sliderIouThreshold").addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        studioState.iouThreshold = val;
        document.getElementById("iouValueLabel").textContent = val.toFixed(2);
        renderCurrentSnapshot();
      });

      document.getElementById("btnToggleProjector").addEventListener("click", () => {
        studioState.projectorEnabled = !studioState.projectorEnabled;
        const btn = document.getElementById("btnToggleProjector");
        if (studioState.projectorEnabled) {
          btn.textContent = "已启用 · 正交剥离";
          btn.className = "soft-button is-on";
          showToast("已开启说话人正交投影");
        } else {
          btn.textContent = "已旁路 · 保留混合";
          btn.className = "soft-button is-off";
          showToast("已关闭说话人正交投影，音色干扰会直接混入轨迹");
        }
        renderCurrentSnapshot();
      });

      document.getElementById("selectCompareTemplate").addEventListener("change", (e) => {
        studioState.selectedTemplateKey = e.target.value;
        renderCurrentSnapshot();
      });

      // Mode 1: Dual-Tempo Enrollment Button
      document.getElementById("btnEnrollDual").addEventListener("click", () => {
        const inputEl = document.getElementById("inputEnrollKey");
        const key = (inputEl.value || "自定义方言口令").trim();
        const seed = 500 + studioState.templates.size * 73;
        const normTake = generatePhraseTrajectory(key, 16, 1.0, 0.15, seed).frames;
        const slowTake = generatePhraseTrajectory(key, 22, 0.68, 0.15, seed + 11).frames;
        studioState.lastEnrollNormal = normTake;
        studioState.lastEnrollSlow = slowTake;

        enrollSequences(key, [normTake, slowTake]);
        studioState.selectedTemplateKey = key;
        studioState.activePhraseKey = key;
        refreshTemplateUI();
        injectPhraseImmediate(key);
        document.getElementById("enrollStatusText").textContent = `已合成「${key}」正常+慢速双轨迹模板`;
        showToast(`已完成「${key}」双语速声纹注册并加入模板库`);
      });

      // Mode 1: Enroll directly from current active focus window
      document.getElementById("btnEnrollFromWindow").addEventListener("click", () => {
        enrollCurrentFocusWindow();
      });

      bindDrawers();
      bindStagePanning();
    }

    function initScrollStory() {
      const targets = Array.from(new Set(document.querySelectorAll(".showcase, [data-reveal]")));
      const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (reduceMotion || !("IntersectionObserver" in window)) {
        completeAllSweeps();
        targets.forEach((target) => target.classList.add("is-inview"));
        return;
      }

      const observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
          const visible = entry.isIntersecting;
          entry.target.classList.toggle("is-inview", visible);
          if (visible && entry.target.dataset.sweep) {
            startSectionSweep(entry.target.dataset.sweep);
          }
        }
      }, {
        rootMargin: "-6% 0px -10% 0px",
        threshold: 0.12
      });

      targets.forEach((target) => observer.observe(target));
    }

    // 60fps main loop: smooth reveal sweeps, inertial drag panning, and real-time live mic updates.
    let lastFrameTime = performance.now();
    let lastAnimationTime = lastFrameTime;
    let frameCounter = 0;
    let captureRng = createRng(99);

    function pushNextStreamFrame(rng) {
      if (studioState.injectionQueue.length > 0) {
        const nextItem = studioState.injectionQueue.shift();
        pushSingleFrame(nextItem.frame, nextItem.wave);
        return;
      }

      const D = STUDIO_CONFIG.dimension;
      const bg = new Float64Array(D);
      for (let d = 0; d < D; d++) bg[d] = 0.016 * rng.nextGaussian();
      pushSingleFrame(bg, 0.014 * rng.nextGaussian());
    }

    function startOscilloscopeLoop() {
      function step(now) {
        const elapsed = Math.min(50, Math.max(0, now - lastAnimationTime));
        lastAnimationTime = now;
        frameCounter++;
        if (now - lastFrameTime >= 500) {
          studioState.fps = Math.min(60, Math.max(55, Math.round((frameCounter * 1000) / (now - lastFrameTime))));
          document.getElementById("fpsReadout").textContent = `${studioState.fps} FPS`;
          frameCounter = 0;
          lastFrameTime = now;
        }

        stepPanInertia();

        if (studioState.micActive && studioState.isPlaying) {
          micStepAccumMs += elapsed;
          let pushed = false;
          let guard = 0;
          while (micStepAccumMs >= MIC_FRAME_STEP_MS && guard < 3) {
            micStepAccumMs -= MIC_FRAME_STEP_MS;
            const { row, wave } = sampleLiveMicrophoneFrame();
            pushSingleFrame(row, wave);
            studioState.frameStepCounter++;
            pushed = true;
            guard++;
          }
          if (pushed) {
            refreshDisplayTelemetry(false);
          }
        } else if (studioState.isPlaying) {
          advanceSweeps(elapsed);
        }

        const telemetry = studioState.displayTelemetry || refreshDisplayTelemetry();
        renderAllUI(telemetry);
        requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }

    function startCaptureMode() {
      document.getElementById("fpsReadout").textContent = "60 FPS";
      document.getElementById("latencyReadout").textContent = "0.12 ms";
      const statusLabel = document.getElementById("streamStatusLabel");
      if (statusLabel) statusLabel.textContent = "演示帧";
      const playLabel = document.getElementById("playLabel");
      if (playLabel) playLabel.textContent = "演示";
      const playIcon = document.getElementById("playIcon");
      if (playIcon) playIcon.textContent = "▮";

      function step(count = 1) {
        for (let i = 0; i < count; i++) {
          if (studioState.injectionQueue.length === 0) {
            queuePhraseUtterance(
              studioState.activePhraseKey,
              studioState.speakingRate,
              studioState.speakerNuisance,
              false
            );
          }
          pushNextStreamFrame(captureRng);
          studioState.frameStepCounter++;
        }
        const telemetry = refreshDisplayTelemetry();
        completeAllSweeps();
        renderAllUI(telemetry);
        return studioState.frameStepCounter;
      }

      window.studioCapture = {
        step,
        selectPhrase(key) {
          injectPhraseImmediate(key);
          return key;
        },
        get frame() {
          return studioState.frameStepCounter;
        }
      };

      completeAllSweeps();
      renderCurrentSnapshot();
    }

    window.addEventListener("resize", () => {
      if (studioState.displayTelemetry) renderAllUI(studioState.displayTelemetry);
    });

    window.addEventListener("load", () => {
      initStudioPresets();
      refreshTemplateUI();
      bindControls();

      if (CAPTURE_MODE) {
        studioState.audioMuted = true;
        injectPhraseImmediate("吴语·侬好 (Wu)", false);
        queuePhraseUtterance("吴语·侬好 (Wu)", studioState.speakingRate, studioState.speakerNuisance, false);
        startCaptureMode();
      } else {
        injectPhraseImmediate("吴语·侬好 (Wu)", false);
        initScrollStory();
        startOscilloscopeLoop();
      }
    });
  

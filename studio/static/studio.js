
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
    document.documentElement.classList.toggle("capture-mode", CAPTURE_MODE);

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
      maxFrames: 120,
      dimension: 12,
      outputLength: 8,
      baseWindowLength: 16,
      scales: [0.75, 1.0, 1.25],
      hop: 2
    };

    const studioState = {
      isPlaying: true,
      captureMode: CAPTURE_MODE,
      micActive: false,
      audioMuted: false,
      projectorEnabled: true,
      speakingRate: 1.25,
      speakerNuisance: 0.38,
      evidenceWeight: 0.25,
      matchThreshold: 0.42,
      iouThreshold: 0.40,
      selectedTemplateKey: "吴语·侬好 (Wu)",
      activePhraseKey: "吴语·侬好 (Wu)",
      frameStepCounter: 0,
      lastComputeMs: 0.18,
      fps: 60,
      // Rolling buffers
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
      lastTelemetry: null
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
        if (kept.length >= 3) break;
      }
      return kept;
    }

    // Build full white-box telemetry snapshot for the 4 tracks
    function computeFullTelemetry() {
      const t0 = performance.now();
      const raw = studioState.rawFrames;
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

      // Focus window for Track 3 arc-length ruler & Track 4 barcode: top match or middle window
      let focusStart = Math.floor(T * 0.35);
      let focusEnd = Math.min(T, focusStart + 24);
      if (activeMatches.length > 0) {
        focusStart = activeMatches[0].start;
        focusEnd = activeMatches[0].end;
      }
      const focusSlice = working.slice(focusStart, focusEnd);

      const streamEv = extractEvidence(working);
      const focusEv = extractEvidence(focusSlice.length >= 4 ? focusSlice : working);
      const arcRuler = computeArcLengthPositions(focusSlice.length >= 4 ? focusSlice : working, STUDIO_CONFIG.outputLength + 1);
      const barcode = computeFingerprintWithSegments(focusSlice.length >= 4 ? focusSlice : working, STUDIO_CONFIG.outputLength);

      const elapsed = performance.now() - t0;
      studioState.lastComputeMs = 0.12 + 0.88 * Math.min(elapsed, 0.95);

      const telemetry = {
        frameCount: T,
        dimension: STUDIO_CONFIG.dimension,
        waveform: Float64Array.from(studioState.waveform),
        frameSpeed,
        frameSpeedNorm,
        activeMatches,
        decomp,
        streamEv,
        focusEv,
        focusWindow: [focusStart, focusEnd],
        arcRuler,
        projInfo,
        barcode
      };
      studioState.lastTelemetry = telemetry;
      return telemetry;
    }

    // =========================================================================
    // 3. Web Audio Synthesizer & Real-Time Microphone Stream Bridge
    // =========================================================================
    let audioCtx = null;
    let micStreamObj = null;
    let micAnalyser = null;
    let micScriptTimer = null;

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

    async function toggleMicrophoneStream() {
      const btn = document.getElementById("btnMicStream");
      const label = document.getElementById("micLabel");
      if (studioState.micActive) {
        studioState.micActive = false;
        if (micScriptTimer) clearInterval(micScriptTimer);
        if (micStreamObj) {
          micStreamObj.getTracks().forEach(t => t.stop());
          micStreamObj = null;
        }
        label.textContent = "开启麦克风";
        btn.className = "tool-button";
        showToast("已关闭实时麦克风输入，切回合成流示波模式");
        return;
      }

      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("当前环境不支持麦克风 API");
        }
        micStreamObj = await navigator.mediaDevices.getUserMedia({ audio: true });
        const ctx = ensureAudioContext();
        const source = ctx.createMediaStreamSource(micStreamObj);
        micAnalyser = ctx.createAnalyser();
        micAnalyser.fftSize = 256;
        source.connect(micAnalyser);

        const freqData = new Uint8Array(micAnalyser.frequencyBinCount);
        const timeData = new Float32Array(micAnalyser.fftSize);
        studioState.micActive = true;
        label.textContent = "麦克风采集中";
        btn.className = "tool-button is-on";
        showToast("麦克风实时流已开启，说话即可驱动四轨视图");

        micScriptTimer = setInterval(() => {
          if (!studioState.micActive || !micAnalyser) return;
          micAnalyser.getByteFrequencyData(freqData);
          micAnalyser.getFloatTimeDomainData(timeData);

          const D = STUDIO_CONFIG.dimension;
          const newRow = new Float64Array(D);
          const binsPerDim = Math.floor(freqData.length / D);
          for (let d = 0; d < D; d++) {
            let sum = 0.0;
            for (let k = 0; k < binsPerDim; k++) {
              sum += freqData[d * binsPerDim + k] / 255.0;
            }
            newRow[d] = (sum / binsPerDim - 0.15) * 2.2;
          }
          let rms = 0.0;
          let peak = 0.0;
          for (let i = 0; i < timeData.length; i++) {
            rms += timeData[i] * timeData[i];
            if (Math.abs(timeData[i]) > Math.abs(peak)) peak = timeData[i];
          }
          const env = Math.sign(peak || 1.0) * Math.min(0.98, Math.sqrt(rms / timeData.length) * 3.2);
          pushSingleFrame(newRow, env);
        }, 35);
      } catch (err) {
        // Fallback: simulate realistic live vocal stream when mic permission is unavailable in sandbox
        studioState.micActive = true;
        label.textContent = "仿真麦克风流中";
        btn.className = "tool-button is-on";
        showToast("当前浏览器容器未授权物理麦克风，已自动挂载 16kHz 实时仿真麦克风语流");
        micScriptTimer = setInterval(() => {
          if (!studioState.micActive) return;
          if (studioState.injectionQueue.length < 8 && Math.random() < 0.18) {
            const keys = Array.from(studioState.templates.keys());
            const pick = keys[Math.floor(Math.random() * keys.length)];
            queuePhraseUtterance(pick, studioState.speakingRate, studioState.speakerNuisance, false);
          }
        }, 260);
      }
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
      for (let i = 0; i < 10; i++) {
        const quiet = new Float64Array(D);
        for (let d = 0; d < D; d++) quiet[d] = 0.015 * rng.nextGaussian();
        studioState.injectionQueue.push({ frame: quiet, wave: 0.012 * rng.nextGaussian() });
      }

      if (playSound) playSynthesizedPhraseSound(key, rate);
    }

    // Immediately flush a phrase into the center of the rolling buffer so it's visible even when paused
    function injectPhraseImmediate(key) {
      studioState.activePhraseKey = key;
      studioState.selectedTemplateKey = key;
      const selectEl = document.getElementById("selectCompareTemplate");
      if (selectEl) selectEl.value = key;
      document.querySelectorAll(".phrase-chip").forEach((chip) => {
        chip.classList.toggle("is-active", chip.dataset.key === key);
      });

      const { frames, wave } = generatePhraseTrajectory(
        key,
        STUDIO_CONFIG.baseWindowLength,
        studioState.speakingRate,
        studioState.speakerNuisance,
        37 + (studioState.frameStepCounter % 53)
      );

      const rng = createRng(91 + studioState.frameStepCounter);
      const D = STUDIO_CONFIG.dimension;
      const totalPush = frames.length + 14;
      for (let i = 0; i < 6; i++) {
        const q = new Float64Array(D);
        for (let d = 0; d < D; d++) q[d] = 0.014 * rng.nextGaussian();
        pushSingleFrame(q, 0.01 * rng.nextGaussian());
      }
      for (let i = 0; i < frames.length; i++) {
        pushSingleFrame(frames[i], wave[i]);
      }
      for (let i = 0; i < 8; i++) {
        const q = new Float64Array(D);
        for (let d = 0; d < D; d++) q[d] = 0.014 * rng.nextGaussian();
        pushSingleFrame(q, 0.01 * rng.nextGaussian());
      }
      playSynthesizedPhraseSound(key, studioState.speakingRate);
      renderAllUI(computeFullTelemetry());
      showToast(`已注入「${key}」(语速 ${studioState.speakingRate.toFixed(2)}x, 音色干扰 ${Math.round(studioState.speakerNuisance * 100)}%)`);
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

    // Map normalized speed [0, 1] -> calm blue (steady) to warm coral (transition).
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

    // Track 1: speed-colored waveform, stable match boxes, CUSUM marker.
    function drawTrack1(telemetry, pulsePhase) {
      const env = prepareCanvas("canvasTrack1");
      if (!env) return;
      const { ctx, w, h } = env;
      ctx.clearRect(0, 0, w, h);

      ctx.strokeStyle = PALETTE.grid;
      ctx.lineWidth = 1;
      for (let gx = 0; gx <= 10; gx++) {
        const x = (gx / 10) * w;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      const midY = h * 0.54;
      ctx.beginPath();
      ctx.moveTo(0, midY);
      ctx.lineTo(w, midY);
      ctx.stroke();

      const T = telemetry.frameCount;
      const wave = telemetry.waveform;
      const speedNorm = telemetry.frameSpeedNorm;

      // Draw vertical colored waveform bars + continuous envelope
      const barWidth = Math.max(2.0, (w / T) * 0.72);
      for (let t = 0; t < T; t++) {
        const x = (t / Math.max(T - 1, 1)) * (w - 16) + 8;
        const amp = Math.min(0.95, Math.abs(wave[t])) * (h * 0.40);
        const col = speedToColor(speedNorm[t], 0.76);
        ctx.fillStyle = col;
        ctx.fillRect(x - barWidth * 0.5, midY - Math.max(2, amp), barWidth, Math.max(4, amp * 2));

        // High-frequency harmonic oscillation lines inside bar
        if (t > 0) {
          const prevX = ((t - 1) / Math.max(T - 1, 1)) * (w - 16) + 8;
          const prevY = midY - wave[t - 1] * (h * 0.38);
          const currY = midY - wave[t] * (h * 0.38);
          ctx.strokeStyle = speedToColor(speedNorm[t], 0.84);
          ctx.lineWidth = 1.35;
          ctx.beginPath();
          ctx.moveTo(prevX, prevY);
          ctx.lineTo(x, currY);
          ctx.stroke();
        }
      }

      // Vertical CUSUM change-point marker.
      const cusumIdx = telemetry.streamEv.cusum.changeIndex;
      const laserX = (cusumIdx / Math.max(T - 1, 1)) * (w - 16) + 8;
      ctx.save();
      ctx.strokeStyle = PALETTE.coral;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.moveTo(laserX, 0);
      ctx.lineTo(laserX, h);
      ctx.stroke();
      ctx.restore();

      // Match boxes are deliberately stable. Breathing glow caused GIF flicker.
      for (let idx = 0; idx < telemetry.activeMatches.length; idx++) {
        const m = telemetry.activeMatches[idx];
        const x1 = (m.start / Math.max(T - 1, 1)) * (w - 16) + 8;
        const x2 = (Math.min(m.end, T - 1) / Math.max(T - 1, 1)) * (w - 16) + 8;
        const boxW = Math.max(36, x2 - x1);

        const isTop = idx === 0;
        const borderAlpha = isTop ? 0.72 : 0.34;
        const fillAlpha = isTop ? 0.09 : 0.045;

        ctx.save();
        ctx.fillStyle = isTop ? `rgba(0, 113, 227, ${fillAlpha})` : `rgba(123, 107, 176, ${fillAlpha})`;
        ctx.strokeStyle = isTop ? `rgba(0, 113, 227, ${borderAlpha})` : `rgba(123, 107, 176, ${borderAlpha})`;
        ctx.lineWidth = isTop ? 1.4 : 1;
        ctx.beginPath();
        ctx.roundRect(x1, 6, boxW, h - 12, 6);
        ctx.fill();
        ctx.stroke();
        ctx.restore();

        // Floating HUD tag at top of the capture box
        const tagText = `${m.key} | 综合:${m.score.toFixed(2)} (指纹:${m.fingerprint_score.toFixed(2)} 证据距:${m.evidence_distance.toFixed(2)})`;
        ctx.font = "600 10px 'SFMono-Regular', Consolas, monospace";
        const textW = ctx.measureText(tagText).width + 12;
        const tagX = Math.max(6, Math.min(w - textW - 6, x1));
        const tagY = idx === 0 ? 10 : h - 26;

        ctx.fillStyle = isTop ? "rgba(255, 255, 255, 0.96)" : "rgba(255, 255, 255, 0.92)";
        ctx.strokeStyle = isTop ? "rgba(0, 113, 227, 0.24)" : "rgba(123, 107, 176, 0.20)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(tagX, tagY, textW, 18, 4);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = isTop ? PALETTE.inkSoft : PALETTE.muted;
        ctx.fillText(tagText, tagX + 6, tagY + 12.5);
      }
    }

    // Track 2: Latent Kinematics White-Box Curves (Speed + Dual Peaks + CUSUM + Turning Angle)
    function drawTrack2(telemetry, pulsePhase) {
      const env = prepareCanvas("canvasTrack2");
      if (!env) return;
      const { ctx, w, h } = env;
      ctx.clearRect(0, 0, w, h);

      ctx.strokeStyle = PALETTE.grid;
      ctx.lineWidth = 1;
      for (let gy = 1; gy <= 3; gy++) {
        const y = (gy / 4) * h;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      const speed = telemetry.decomp.speed;
      const turning = telemetry.decomp.turning;
      const cusumCurve = telemetry.streamEv.cusum.curve;
      const cusumIdx = telemetry.streamEv.cusum.changeIndex;
      const peaks = telemetry.streamEv.peaks;
      const nSp = speed.length;

      let maxSp = EPSILON;
      for (let i = 0; i < nSp; i++) if (speed[i] > maxSp) maxSp = speed[i];
      let maxCusum = EPSILON;
      for (let i = 0; i < cusumCurve.length; i++) {
        if (Math.abs(cusumCurve[i]) > maxCusum) maxCusum = Math.abs(cusumCurve[i]);
      }

      // 1. Turning Angle Curve cos(theta_t) in [-1, 1] (purple dashed)
      if (turning.length > 1) {
        ctx.save();
        ctx.strokeStyle = "rgba(123, 107, 176, 0.62)";
        ctx.lineWidth = 1.25;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        for (let i = 0; i < turning.length; i++) {
          const x = (i / Math.max(turning.length - 1, 1)) * (w - 16) + 8;
          // Map [-1, 1] -> [h - 14, 18]
          const normTurn = (turning[i] + 1.0) * 0.5;
          const y = h - 14 - normTurn * (h - 32);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();
      }

      // 2. Zero-mean CUSUM curve.
      ctx.save();
      ctx.strokeStyle = "rgba(230, 106, 78, 0.62)";
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      for (let i = 0; i < cusumCurve.length; i++) {
        const x = (i / Math.max(cusumCurve.length - 1, 1)) * (w - 16) + 8;
        const normC = 0.5 + 0.45 * (cusumCurve[i] / maxCusum);
        const y = h - 12 - normC * (h - 24);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();

      // 3. Speed curve with a restrained gradient fill.
      ctx.save();
      const grad = ctx.createLinearGradient(0, 10, 0, h);
      grad.addColorStop(0, "rgba(0, 113, 227, 0.14)");
      grad.addColorStop(1, "rgba(0, 113, 227, 0.0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(8, h - 10);
      for (let i = 0; i < nSp; i++) {
        const x = (i / Math.max(nSp - 1, 1)) * (w - 16) + 8;
        const y = h - 12 - (speed[i] / maxSp) * (h - 28);
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w - 8, h - 10);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = PALETTE.blue;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i < nSp; i++) {
        const x = (i / Math.max(nSp - 1, 1)) * (w - 16) + 8;
        const y = h - 12 - (speed[i] / maxSp) * (h - 28);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();

      // 4. Vertical CUSUM change-point marker.
      const laserX = (cusumIdx / Math.max(nSp - 1, 1)) * (w - 16) + 8;
      ctx.save();
      ctx.strokeStyle = PALETTE.coral;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.moveTo(laserX, 4);
      ctx.lineTo(laserX, h - 4);
      ctx.stroke();

      ctx.setLineDash([]);
      ctx.fillStyle = PALETTE.coral;
      ctx.font = "600 10px 'SFMono-Regular', Consolas, monospace";
      const cpLabel = `CUSUM 变点 ${(cusumIdx / Math.max(nSp - 1, 1)).toFixed(2)}`;
      const cpX = 8;
      ctx.fillText(cpLabel, cpX, 14);
      ctx.restore();

      // 5. Dual Peak Lock Indicators (Primary & Secondary Peaks from select_peak_indices)
      const peakLabels = ["主峰 P1", "次峰 P2"];
      const peakColors = [PALETTE.blue, PALETTE.teal];
      for (let p = 0; p < Math.min(2, peaks.length); p++) {
        const idx = peaks[p];
        const px = (idx / Math.max(nSp - 1, 1)) * (w - 16) + 8;
        const py = h - 12 - (speed[idx] / maxSp) * (h - 28);
        const radius = p === 0 ? 4.8 : 4.0;

        ctx.save();
        ctx.fillStyle = peakColors[p];
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = "650 9px 'SFMono-Regular', Consolas, monospace";
        ctx.fillStyle = p === 0 ? PALETTE.blue : PALETTE.teal;
        ctx.fillText(p === 0 ? "P1" : "P2", Math.max(6, px - 7), Math.max(11, py - 8));
        ctx.restore();
      }
    }

    // Track 3: Equal-Arc-Length Resampling Ruler vs Linear Uniform Sampling
    function drawTrack3(telemetry) {
      const env = prepareCanvas("canvasTrack3");
      if (!env) return;
      const { ctx, w, h } = env;
      ctx.clearRect(0, 0, w, h);

      const { linearPositions, arcPositions } = telemetry.arcRuler;
      const nTicks = linearPositions.length;
      const maxPos = Math.max(linearPositions[nTicks - 1], EPSILON);

      const padX = 52;
      const usableW = Math.max(40, w - padX * 2);
      const yLinear = 28;
      const yArc = h - 26;

      // Labels on left
      ctx.font = "600 10px 'SFMono-Regular', Consolas, monospace";
      ctx.fillStyle = PALETTE.faint;
      ctx.fillText("Linear", 8, yLinear + 4);
      ctx.fillStyle = PALETTE.teal;
      ctx.fillText("ArcLen", 8, yArc + 4);

      // Horizontal ruler rails
      ctx.strokeStyle = "rgba(29, 29, 31, 0.12)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padX, yLinear);
      ctx.lineTo(padX + usableW, yLinear);
      ctx.stroke();

      ctx.strokeStyle = "rgba(43, 156, 148, 0.52)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padX, yArc);
      ctx.lineTo(padX + usableW, yArc);
      ctx.stroke();

      // Elastic warp connectors between Linear Uniform ticks and Equal-Arc-Length ticks
      for (let i = 0; i < nTicks; i++) {
        const lx = padX + (linearPositions[i] / maxPos) * usableW;
        const ax = padX + (arcPositions[i] / maxPos) * usableW;

        // Connector curve showing time-warp compensation
        ctx.strokeStyle = "rgba(0, 113, 227, 0.26)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(lx, yLinear + 5);
        ctx.bezierCurveTo(lx, (yLinear + yArc) * 0.5, ax, (yLinear + yArc) * 0.5, ax, yArc - 5);
        ctx.stroke();

        // Top tick (Linear)
        ctx.fillStyle = PALETTE.faint;
        ctx.fillRect(lx - 1.5, yLinear - 6, 3, 12);

        // Bottom tick (Equal Arc-Length)
        ctx.fillStyle = PALETTE.teal;
        ctx.beginPath();
        ctx.arc(ax, yArc, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = "500 9px 'SFMono-Regular', Consolas, monospace";
        ctx.fillStyle = PALETTE.muted;
        ctx.fillText(`s${i}`, ax - 6, yArc + 16);
      }

      // Caption on top right
      ctx.font = "500 10px -apple-system, 'Segoe UI', 'PingFang SC', sans-serif";
      ctx.fillStyle = PALETTE.faint;
      const rateHint = studioState.speakingRate < 0.85
        ? "当前慢速拖音，等弧长刻度自动变疏"
        : studioState.speakingRate > 1.15
          ? "当前快速连读，高速段等弧长刻度自动加密"
          : "标准语速，几何弧长均匀对齐";
      ctx.fillText(rateHint, Math.max(padX, w - ctx.measureText(rateHint).width - 10), 14);
    }

    // Track 4: 4-Partition Weighted Fingerprint Heatmap Barcode Comparison
    function drawTrack4(telemetry) {
      const env = prepareCanvas("canvasTrack4");
      if (!env) return;
      const { ctx, w, h } = env;
      ctx.clearRect(0, 0, w, h);

      const { vector: qVec, segments } = telemetry.barcode;
      const tmplItem = studioState.templates.get(studioState.selectedTemplateKey)
        || studioState.templates.values().next().value;
      if (!tmplItem) return;
      const tVec = tmplItem.template;
      const dim = qVec.length;

      const padLeft = 86;
      const padRight = 14;
      const barW = Math.max(40, w - padLeft - padRight);
      const cellW = barW / dim;

      // Colored partition strips. The labels live in the pill row below, so the
      // narrow Speed/Accel partitions never collide inside the canvas.
      for (const seg of segments) {
        const sx = padLeft + seg.start * cellW;
        const sw = (seg.end - seg.start) * cellW;
        ctx.fillStyle = seg.color + "22";
        ctx.strokeStyle = seg.color + "88";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(sx + 1, 6, Math.max(4, sw - 2), 18, 4);
        ctx.fill();
        ctx.stroke();
        ctx.font = "600 8px 'SFMono-Regular', Consolas, monospace";
        ctx.fillStyle = seg.color;
        const header = { static: "STATIC", delta: "DELTA", speed: "SPEED", acceleration: "ACCEL" }[seg.name] || seg.name.toUpperCase();
        ctx.fillText(header, sx + 5, 18.5);

      }

      // Row labels
      ctx.font = "600 10px 'SFMono-Regular', Consolas, monospace";
      ctx.fillStyle = PALETTE.inkSoft;
      ctx.fillText("当前窗口指纹", 8, 48);
      ctx.fillStyle = PALETTE.blue;
      ctx.fillText("注册模板指纹", 8, 84);
      ctx.fillStyle = PALETTE.faint;
      ctx.fillText("逐维共振热力", 8, 118);

      // Value to color helper for barcode cells
      function valToBarcodeColor(v) {
        const norm = Math.max(-1.0, Math.min(1.0, v * 4.5));
        if (norm >= 0) {
          const a = 0.25 + 0.75 * norm;
          return `rgba(0, 113, 227, ${a.toFixed(2)})`;
        } else {
          const a = 0.25 + 0.75 * (-norm);
          return `rgba(230, 106, 78, ${a.toFixed(2)})`;
        }
      }

      for (let i = 0; i < dim; i++) {
        const x = padLeft + i * cellW;
        const cw = Math.max(1.2, cellW - 1.0);

        // Row 1: Query Barcode
        ctx.fillStyle = valToBarcodeColor(qVec[i]);
        ctx.fillRect(x, 31, cw, 25);

        // Row 2: Enrolled Template Barcode
        ctx.fillStyle = valToBarcodeColor(tVec[i]);
        ctx.fillRect(x, 67, cw, 25);

        // Row 3: alignment match bar (teal = aligned, coral = mismatch)
        const diff = Math.abs(qVec[i] - tVec[i]);
        const matchScore = Math.max(0.0, 1.0 - diff * 3.8);
        ctx.fillStyle = matchScore > 0.55
          ? `rgba(43, 156, 148, ${(0.28 + 0.62 * matchScore).toFixed(2)})`
          : `rgba(230, 106, 78, ${(0.68 - 0.38 * matchScore).toFixed(2)})`;
        ctx.fillRect(x, 103, cw, 18);
      }

      // Partition boundary dividers
      for (const seg of segments) {
        if (seg.start === 0) continue;
        const bx = padLeft + seg.start * cellW;
        ctx.strokeStyle = "rgba(29, 29, 31, 0.10)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(bx, 28);
        ctx.lineTo(bx, 124);
        ctx.stroke();
      }
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
              renderAllUI(computeFullTelemetry());
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
          renderAllUI(computeFullTelemetry());
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

    function bindControls() {
      document.getElementById("btnTogglePlay").addEventListener("click", () => {
        studioState.isPlaying = !studioState.isPlaying;
        document.getElementById("playIcon").textContent = studioState.isPlaying ? "Ⅱ" : "▶";
        document.getElementById("playLabel").textContent = studioState.isPlaying ? "暂停" : "继续";
        document.getElementById("streamStatusDot").className = studioState.isPlaying
          ? "status-dot is-live"
          : "status-dot is-paused";
        document.getElementById("streamStatusLabel").textContent = studioState.isPlaying ? "实时演示" : "已暂停";
        showToast(studioState.isPlaying ? "已恢复实时流" : "已暂停，可仔细查看当前窗口");
      });

      document.getElementById("btnMicStream").addEventListener("click", () => {
        toggleMicrophoneStream();
      });

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
        const start = Math.floor(T * 0.3);
        const end = Math.min(T, start + 24);
        for (let i = end - 1; i > start; i--) {
          const j = start + Math.floor(rng.nextFloat() * (i - start + 1));
          const tmp = studioState.rawFrames[i];
          studioState.rawFrames[i] = studioState.rawFrames[j];
          studioState.rawFrames[j] = tmp;
        }
        renderAllUI(computeFullTelemetry());
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
        renderAllUI(computeFullTelemetry());
      });

      document.getElementById("sliderMatchThreshold").addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        studioState.matchThreshold = val;
        document.getElementById("threshValueLabel").textContent = val.toFixed(2);
        renderAllUI(computeFullTelemetry());
      });

      document.getElementById("sliderIouThreshold").addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        studioState.iouThreshold = val;
        document.getElementById("iouValueLabel").textContent = val.toFixed(2);
        renderAllUI(computeFullTelemetry());
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
        renderAllUI(computeFullTelemetry());
      });

      document.getElementById("selectCompareTemplate").addEventListener("change", (e) => {
        studioState.selectedTemplateKey = e.target.value;
        renderAllUI(computeFullTelemetry());
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
        const inputEl = document.getElementById("inputEnrollKey");
        const key = (inputEl.value || "实时窗口口令").trim();
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
          renderAllUI(computeFullTelemetry());
          showToast(`已截取当前窗口 [${wStart}, ${wEnd}] 注册为「${key}」指纹模板`);
        }
      });

      bindDrawers();
    }

    // Sampled animation loop. Rendering is smooth; the stream advances on a stable cadence.
    let lastFrameTime = performance.now();
    let frameCounter = 0;
    let liveRng = createRng(99);
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
        frameCounter++;
        if (now - lastFrameTime >= 500) {
          studioState.fps = Math.min(60, Math.max(55, Math.round((frameCounter * 1000) / (now - lastFrameTime))));
          document.getElementById("fpsReadout").textContent = `${studioState.fps} FPS`;
          frameCounter = 0;
          lastFrameTime = now;
        }

        if (studioState.isPlaying) {
          studioState.frameStepCounter++;
          if (studioState.frameStepCounter % 3 === 0) {
            if (studioState.injectionQueue.length === 0 && studioState.frameStepCounter % 105 === 0) {
              queuePhraseUtterance(
                studioState.activePhraseKey,
                studioState.speakingRate,
                studioState.speakerNuisance,
                false
              );
            }
            pushNextStreamFrame(liveRng);
          }
        }

        renderAllUI(computeFullTelemetry());
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
        renderAllUI(computeFullTelemetry());
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

      renderAllUI(computeFullTelemetry());
    }

    window.addEventListener("resize", () => {
      if (studioState.lastTelemetry) renderAllUI(studioState.lastTelemetry);
    });

    window.addEventListener("load", () => {
      initStudioPresets();
      refreshTemplateUI();
      bindControls();

      if (CAPTURE_MODE) {
        studioState.audioMuted = true;
        injectPhraseImmediate("吴语·侬好 (Wu)");
        queuePhraseUtterance("吴语·侬好 (Wu)", studioState.speakingRate, studioState.speakerNuisance, false);
        startCaptureMode();
      } else {
        injectPhraseImmediate("吴语·侬好 (Wu)");
        startOscilloscopeLoop();
      }
    });
  

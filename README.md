Markdown
# midi2audio: Reference-Anchored Generative Music Pipeline

A reference-track-anchored multitrack MIDI expressivization, dry stem synthesis, generative restyling, automated multi-axis scoring, and production mastering pipeline. 

Built strictly on the architectural principle of **separation of concerns**: notes, timing, swing, and microtiming are handled deterministically in code; acoustic sound, space, and texture are the responsibility of generative models.

---

## Core Architectural Contracts

1. **Separation of Concerns:** Generative models are statistical re-performers, not renderers. They drift pitch, simplify chords, and smooth out syncopations. Notes, tempo maps, swing, microtiming, and velocities are resolved symbolically in MIDI prior to invoking any audio endpoint.
2. **Deterministic Ground-Truth Spine:** Stage 1 bakes expressive feel into MIDI; Stage 2 renders clean dry stems at 48 kHz stereo with `-12 dBFS` peak headroom. These stems serve as the immutable ground-truth targets for downstream scoring.
3. **Reference-Anchored Targets:** Ambiguous natural language prompts are anchored using audio reference recordings:
   * *Groove References:* Extracted onset deviations and metrical energy maps.
   * *Style References:* Acoustic profiles distilled into structured House Style Cards[cite: 1].
   * *Scoring References:* Evaluated via Fréchet Audio Distance (FAD) or single-track cosine similarity[cite: 1].
4. **Hard Quality Floors:** Generative candidates are filtered by non-negotiable rhythm and note-fidelity thresholds[cite: 1]. If an AI restyle damages syncopation or changes pitch voicings, it receives a composite score of `0.0` and is discarded[cite: 1].
5. **Contract & Artifact Isolation:** Stages communicate exclusively through disk-persisted files with companion JSON sidecar manifests containing canonical SHA-256 cryptographic digests, configurations, and timestamps.
6. **Financial Circuit Breakers:** Remote generative API invocations are monitored by hard-budget circuit breakers (`max_endpoint_calls`, `max_usd_estimate`) and exponential backoff retry policies[cite: 1, 3].

---

## Pipeline Architecture Overview

[Quantized / Raw MIDI]
│
▼
┌─────────────────────────────────┐
│ STAGE 0: MIDI Analysis          │ ──► Tempo maps, metrical grid, quantization check[cite: 1]
└─────────────────────────────────┘
│
▼
┌─────────────────────────────────┐
│ STAGE 1: Symbolic Groove        │ ──► 16th-slot microtiming, swing, dynamic arcs[cite: 1]
└─────────────────────────────────┘
│
▼
┌─────────────────────────────────┐
│ STAGE 2: Deterministic Render   │ ──► Isolated dry stems @ 48 kHz, -12 dBFS peak[cite: 1]
└─────────────────────────────────┘
│
├───► Dry Ground Truth Stems
▼
┌─────────────────────────────────┐
│ STAGE 3: Generative Restyle     │ ──► Bar-chunking, audio-to-audio restyle, inpainting[cite: 1]
└─────────────────────────────────┘
│
▼
┌─────────────────────────────────┐
│ STAGE 5: Scoring & Search Loop  │ ◄── Hard-floor gates on Rhythm F1 & Chroma Cosine[cite: 1]
└─────────────────────────────────┘
│
▼
┌─────────────────────────────────┐
│ STAGE 6: Mix & Master           │ ──► Cross-correlation alignment, mix, Matchering 2.0[cite: 1]
└─────────────────────────────────┘
│
▼
[Production Master Audio]


---

## Repository Structure

```text
midi2audio/
├── pyproject.toml              # Build backend, dependencies, and CLI setup
├── README.md                   # Project documentation
├── DECISIONS.md                # Architectural Decision Records (ADRs)
├── config/
│   ├── default.yaml            # Master pipeline parameters, floors, and budgets
│   └── profiles/               # Specialized overrides (e.g., funk.yaml, single_ref.yaml)[cite: 1]
├── m2a/
│   ├── analysis.py             # Stage 0: Multitrack MIDI analysis and role detection[cite: 1]
│   ├── host.py                 # Single-ref: Host beat-grid tracking and Demucs separation[cite: 1]
│   ├── manifest.py             # SHA-256 artifact hashing and sidecar manifests[cite: 1]
│   ├── orchestrator.py         # DAG orchestrator with content-addressable execution[cite: 1]
│   ├── groove/
│   │   ├── spec.py             # Pydantic schema for GrooveSpec[cite: 1]
│   │   ├── apply.py            # Deterministic microtiming, swing, and dynamics applicator[cite: 1]
│   │   ├── extract.py          # Empirical groove template extraction from reference audio[cite: 1]
│   │   ├── lattice.py          # Bounded re-quantization to host onset lattice[cite: 1]
│   │   └── models.py           # Model-based humanization wrappers (GrooVAE/MIDI-GPT)[cite: 1]
│   ├── render/
│   │   ├── fluidsynth_r.py     # SoundFont rendering with pure DSP fallback[cite: 1, 2]
│   │   ├── registry.py         # Track role dispatch and multi-stem rendering[cite: 1]
│   │   └── hostsampler.py      # Auto-SFZ slicing from isolated host stems[cite: 1]
│   ├── restyle/
│   │   ├── base.py             # RestyleEndpoint ABC, exception contracts, CandidateResult[cite: 1]
│   │   ├── chunking.py         # Bar-aligned splitting and equal-power crossfading[cite: 1]
│   │   ├── stable_audio.py     # Stable Audio 2.5 client with retry policies[cite: 1]
│   │   └── musicgen_local.py   # Local open-weight baseline for rapid prototyping[cite: 1]
│   ├── prompts/
│   │   ├── stylecard.py        # Reference audio acoustic analysis -> House Style Card[cite: 1]
│   │   └── generate.py         # Per-stem prompt variant generator[cite: 1]
│   ├── scoring/
│   │   ├── rhythm.py           # Onset F-measure (25ms/50ms) and DTW envelope cost[cite: 1]
│   │   ├── notes.py            # Constant-Q chroma frame cosine similarity[cite: 1]
│   │   ├── clap_sim.py         # LAION-CLAP candidate vs prompt text similarity[cite: 1]
│   │   ├── fad.py              # Fréchet Audio Distance over reference distributions[cite: 1]
│   │   └── combine.py          # Composite weighted scorer with non-negotiable hard floors[cite: 1]
│   ├── search/
│   │   └── loop.py             # Successive-halving bandit optimizer with budget guard[cite: 1]
│   └── mixmaster/
│       ├── align.py            # Cross-correlation delay compensation[cite: 1]
│       ├── mix.py              # Constant-power panning and static stem summing[cite: 1]
│       └── master.py           # Matchering 2.0 reference mastering & peak audits[cite: 1]
├── scripts/
│   └── run_pipeline.py         # CLI entrypoint[cite: 1]
└── tests/
    ├── test_spine.py           # M1: Tick-accuracy and deterministic render tests[cite: 1, 2]
    ├── test_scoring.py         # M2: Monotonic metric degradation & hard floor tests[cite: 1, 3]
    ├── test_restyle_search.py  # M3: Chunking, budget breaker, and bandit search tests[cite: 1]
    └── test_references.py      # M4: Groove recovery, delay alignment, and mastering tests[cite: 1]
Installation
Requirements
Python >=3.11

[cite: 1, 2]

Recommended: System fluidsynth binary and a SoundFont (.sf2)[cite: 1, 2]. (An automated band-limited synthesis fallback runs if FluidSynth is absent).

Setup
Bash
# Clone the repository
git clone <repository_url>
cd midi2audio

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
To enable full neural source separation and reference mastering extras:

Bash
pip install -e ".[advanced_dsp]"
Configuration
The default configuration is defined in config/default.yaml[cite: 1, 2]:

YAML
pipeline:
  sample_rate: 48000
  headroom_dbfs: -12.0
  cache_dir: ".cache/m2a"
  artifacts_dir: "artifacts"
  soundfont_path: "soundfonts/GeneralUser_GS.sf2"

budget:
  max_endpoint_calls: 30
  max_usd_estimate: 15.0

stage0_analysis:
  quantization_threshold_ms: 12.0
  flat_velocity_threshold_std: 3.0

stage1_groove:
  swing:
    ratio: 0.58
    subdivision: 8
    applies_to: ["drums", "bass", "comp"]
  microtiming_ms:
    drums:
      by_position_16: [0.0, -2.0, 6.0, 3.0, 0.0, -1.0, 7.0, 2.0, 0.0, -2.0, 6.0, 3.0, 0.0, -1.0, 7.0, 2.0]
      global_offset: 0.0
      jitter_sd: 2.0
  velocity:
    accent_map_16: [1.0, 0.6, 0.7, 0.65, 0.9, 0.6, 0.75, 0.65, 1.0, 0.6, 0.7, 0.65, 0.9, 0.6, 0.75, 0.65]
    phrase_arc:
      shape: "crescendo_to_bar4"
      depth: 0.2
Usage
CLI Commands
Run the deterministic spine and pipeline stages:

Bash
# Process input MIDI through default configuration
midi2audio path/to/song.mid

# Run with custom profile override
python scripts/run_pipeline.py path/to/song.mid --config config/default.yaml --profile config/profiles/funk.yaml
Python API
Python
from pathlib import Path
import yaml
from m2a.orchestrator import PipelineOrchestrator

with open("config/default.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

orchestrator = PipelineOrchestrator(config)
results = orchestrator.run_deterministic_spine("path/to/song.mid")

print("Expressive MIDI Output:", results["groove_midi_path"])
for stem in results["stems"]:
    print(f"Stem [{stem['role']}]: {stem['audio_path']}")
Testing & Quality Verification
All pipeline components must satisfy programmatic verification contracts before deployment[cite: 1]:

Bash
# Run Milestone 1: Deterministic spine & tick accuracy
pytest tests/test_spine.py -v

# Run Milestone 2: Automated scoring & synthetic degradation monotonicity
pytest tests/test_scoring.py -v

# Run Milestone 3: Bar-chunking, circuit-breaker guards & bandit search
pytest tests/test_restyle_search.py -v

# Run Milestone 4: Groove extraction, delay alignment & reference mastering
pytest tests/test_references.py -v

# Run complete test suite
pytest -v
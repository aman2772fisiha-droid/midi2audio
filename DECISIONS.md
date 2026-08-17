# Architecture Decisions

## STAGE 0 analysis

### Symbolic coordinate system
Perform symbolic analysis primarily in MIDI ticks. Convert to seconds only where an absolute-time measurement is required, such as quantization tolerance.

### Quantization grid
The initial rhythmic grid is a 16th-note grid.

### Quantization tolerance
The initial target is 8 ms. This is an implementation choice, not a value specified by the project PDF.

### Quantization classification
An onset is considered within tolerance when its distance from the nearest grid position is <= the configured tolerance. The overall track classification is based on the fraction of onsets within tolerance. The initial classification threshold is 0.95.

### Polyphony
Polyphony mean is time-weighted over the track's active note duration. Maximum polyphony is the maximum number of simultaneously active notes.

### Note density
Note density is the number of note-on events beginning within each bar.

### Onset profile
The initial onset profile uses 16 subdivisions per bar. Original onsets are mapped to their nearest subdivision before histogramming. The profile is normalized so its values sum to approximately 1.

### Velocity flatness
The artifact records mean velocity, standard deviation, number of unique velocities, and a flat classification. The exact classification threshold remains an implementation decision to be validated with fixtures.

### Meter changes
The first meter event establishes the initial bar grid. A subsequent meter event starts a new meter regime at its event position. Pickup/anacrusis handling is not yet specified.

### Analysis failures
A successful analysis may contain an empty key/section result when no such feature is detected. A genuine processing failure should raise an error rather than silently emitting a partial artifact.

### Scope
This first implementation covers the deterministic symbolic-analysis spine. Key detection, chord analysis, section inference, and role classification require separate algorithm design before implementation.

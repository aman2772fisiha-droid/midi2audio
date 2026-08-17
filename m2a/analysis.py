from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import mido

from .schemas import (
    AnalysisResult,
    HarmonyAnalysis,
    PolyphonyStats,
    QuantizationStats,
    TrackAnalysis,
    TrackRole,
    VelocityStats,
    SourceInfo,
)

@dataclass(frozen=True)
class NoteEvent:
    track_idx: int
    pitch: int
    velocity: int
    start_tick: int
    end_tick: int


def extract_notes(
    midi: mido.MidiFile,
) -> list[NoteEvent]:
    notes = []

    for track_idx, track in enumerate(midi.tracks):
        current_tick = 0

        active_notes = {}

        for message in track:
            current_tick += message.time

            if message.type == "note_on" and message.velocity > 0:
                key = (message.channel, message.note)

                active_notes[key] = (
                    current_tick,
                    message.velocity,
                )

            elif (
                message.type == "note_off"
                or (
                    message.type == "note_on"
                    and message.velocity == 0
                )
            ):
                key = (message.channel, message.note)

                if key not in active_notes:
                    continue

                start_tick, velocity = active_notes.pop(key)

                if current_tick <= start_tick:
                    continue

                notes.append(
                    NoteEvent(
                        track_idx=track_idx,
                        pitch=message.note,
                        velocity=velocity,
                        start_tick=start_tick,
                        end_tick=current_tick,
                    )
                )

    return notes

def group_notes_by_track(
    notes: list[NoteEvent],
) -> dict[int, list[NoteEvent]]:
    grouped: dict[int, list[NoteEvent]] = {}

    for note in notes:
        grouped.setdefault(note.track_idx, []).append(note)

    return grouped

def pitch_range(
    notes: list[NoteEvent],
) -> tuple[int, int] | None:
    if not notes:
        return None

    pitches = [note.pitch for note in notes]

    return min(pitches), max(pitches)

def note_density(
    notes: list[NoteEvent],
    ppq: int,
    numerator: int,
    denominator: int,
) -> list[int]:
    if not notes:
        return []

    ticks_per_bar = (
        ppq * 4 * numerator // denominator
    )

    bar_counts: dict[int, int] = {}

    for note in notes:
        bar = note.start_tick // ticks_per_bar
        bar_counts[bar] = bar_counts.get(bar, 0) + 1

    last_bar = max(bar_counts)

    return [
        bar_counts.get(bar, 0)
        for bar in range(last_bar + 1)
    ]

def velocity_stats(
    notes: list[NoteEvent],
) -> VelocityStats:
    if not notes:
        return VelocityStats(
            mean=0.0,
            std=0.0,
            unique_values=0,
            flat=False,
        )

    velocities = [note.velocity for note in notes]

    average = mean(velocities)
    std = pstdev(velocities)

    return VelocityStats(
        mean=average,
        std=std,
        unique_values=len(set(velocities)),
        flat=std == 0.0,
    )

def polyphony_stats(
    notes: list[NoteEvent],
) -> PolyphonyStats:
    if not notes:
        return PolyphonyStats(
            mean=0.0,
            max=0,
        )

    events: list[tuple[int, int]] = []

    for note in notes:
        events.append((note.start_tick, 1))
        events.append((note.end_tick, -1))

    # End events must be processed before start events
    # at the same tick.
    events.sort(key=lambda event: (event[0], event[1]))

    active = 0
    maximum = 0
    weighted_polyphony = 0
    total_duration = 0

    previous_tick = events[0][0]

    index = 0

    while index < len(events):
        tick = events[index][0]

        duration = tick - previous_tick

        if duration > 0:
            weighted_polyphony += active * duration
            total_duration += duration

        while index < len(events) and events[index][0] == tick:
            active += events[index][1]
            maximum = max(maximum, active)
            index += 1

        previous_tick = tick

    mean = (
        weighted_polyphony / total_duration
        if total_duration > 0
        else 0.0
    )

    return PolyphonyStats(
        mean=mean,
        max=maximum,
    )

def onset_profile_16(
    notes: list[NoteEvent],
    ppq: int,
) -> list[float]:
    if not notes:
        return [0.0] * 16

    ticks_per_16th = ppq / 4

    counts = [0] * 16

    for note in notes:
        slot = round(note.start_tick / ticks_per_16th) % 16
        counts[slot] += 1

    total = len(notes)

    return [
        count / total
        for count in counts
    ]

def bpm_at_tick(
    tempo_events: list[dict],
    tick: int,
) -> float:
    bpm = 120.0

    for event in tempo_events:
        if event["position_ticks"] > tick:
            break

        bpm = event["bpm"]

    return bpm
    # This is a helper function for quantization_stats()

def quantization_stats(
    notes: list[NoteEvent],
    ppq: int,
    tempo_events: list[dict],
    tolerance_ms: float = 8.0,
) -> QuantizationStats:
    if not notes:
        return QuantizationStats(
            fraction_within_tolerance=0.0,
            tolerance_ms=tolerance_ms,
            quantized=False,
        )

    ticks_per_16th = ppq / 4
    within_tolerance = 0

    for note in notes:
        nearest_grid = (
            round(note.start_tick / ticks_per_16th)
            * ticks_per_16th
        )

        tick_error = abs(
            note.start_tick - nearest_grid
        )

        bpm = bpm_at_tick(
            tempo_events,
            note.start_tick,
        )

        error_ms = (
            tick_error
            * 60_000
            / (bpm * ppq)
        )

        if error_ms <= tolerance_ms:
            within_tolerance += 1

    fraction = within_tolerance / len(notes)

    return QuantizationStats(
        fraction_within_tolerance=fraction,
        tolerance_ms=tolerance_ms,
        quantized=fraction == 1.0,
    )

def analyze_midi(path: str | Path) -> AnalysisResult:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    midi = mido.MidiFile(path)

    tempo_events = []
    meter_events = []

    for track in midi.tracks:
        current_tick = 0

        for message in track:
            current_tick += message.time

            if message.type == "set_tempo":
                tempo_events.append(
                    {
                        "position_ticks": current_tick,
                        "bpm": mido.tempo2bpm(message.tempo),
                    }
                )

            elif message.type == "time_signature":
                meter_events.append(
                    {
                        "position_ticks": current_tick,
                        "numerator": message.numerator,
                        "denominator": message.denominator,
                    }
                )

    tempo_events.sort(
        key=lambda event: event["position_ticks"]
    )

    meter_events.sort(
        key=lambda event: event["position_ticks"]
    )

    notes = extract_notes(midi)
    notes_by_track = group_notes_by_track(notes)

    tracks = []

    for track_idx, track_notes in sorted(notes_by_track.items()):
        tracks.append(
            TrackAnalysis(
                idx=track_idx,
                role=TrackRole.UNKNOWN,
                program=None,
                range=pitch_range(track_notes),

                polyphony=polyphony_stats(track_notes),
            
                note_density=note_density(
                    track_notes,
                    ppq=midi.ticks_per_beat,
                    numerator=4,
                    denominator=4,
                ),


                onset_profile_16=onset_profile_16(
                    track_notes,
                    ppq=midi.ticks_per_beat,
                ),


                quantization=quantization_stats(
                    track_notes,
                    ppq=midi.ticks_per_beat,
                    tempo_events=tempo_events,
                ),

                velocity=velocity_stats(track_notes),
                ),
            )
        

    return AnalysisResult(
        schema_version="0.1",
        content_hash="",
        source=SourceInfo(
            filename=path.name,
            ppq=midi.ticks_per_beat,
            duration_seconds=0.0,
        ),
        tempo_map=tempo_events,
        meter=meter_events,
        key=[],
        harmony=HarmonyAnalysis(
            resolution="beat",
            events=[],
        ),
        sections=[],
        tracks = tracks,
    )


    
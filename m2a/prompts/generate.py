"""Stage 4: Rule-based and template prompt generator producing multi-stem variations."""

from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel


class PromptVariant(BaseModel):
    """Structured text prompt configuration for generative restyle models."""

    variant_id: str
    prompt: str
    negative_prompt: str


ROLE_PROMPT_TEMPLATES: Dict[str, List[Dict[str, str]]] = {
    "drums": [
        {
            "id": "v1_analog_studio",
            "prompt": "Tight punchy studio drum kit, crisp snare with ghost notes, round kick, small treated room, vintage Neve console warmth, 48kHz production",
            "negative": "modern EDM, washed reverb, trap hats, synthetic snare, distorted brickwall",
        },
        {
            "id": "v2_dry_funk",
            "prompt": "Authentic dry funk drumming, dampened snare head, muffled kick drum, close mic ribbon warmth, tight 1970s tape saturation",
            "negative": "hall reverb, boomy low end, modern digital sheen, electronic cymbals",
        },
        {
            "id": "v3_punchy_rock",
            "prompt": "Solid dynamic rock drum performance, natural overheads, articulate hi-hats, focused transient punch, studio SSL channel strip",
            "negative": "flamming, thin snare, phase artifacts, excessive room wash",
        },
    ],
    "bass": [
        {
            "id": "v1_vintage_p_bass",
            "prompt": "Warm vintage Fender Precision electric bass, round flatwound tone, played with fingers, subtle Ampeg B15 tube amp drive",
            "negative": "slap bass, sub-bass 808, synth bass, harsh fret buzz, boomy mud",
        },
        {
            "id": "v2_modern_punch",
            "prompt": "Articulate active electric bass guitar, punchy low-mids, clean direct input mixed with subtle tube warmth, tight dynamics",
            "negative": "distortion, chorus wash, hollow mids, muddiness",
        },
    ],
    "comp": [
        {
            "id": "v1_rhodes_electric",
            "prompt": "Vintage Rhodes Mark I stage piano, bell tone warmth, gentle stereo vibrato, played through a warm twin reverb amp",
            "negative": "harsh digital synthesis, pitch drift, detuned keys, mono collapse",
        },
        {
            "id": "v2_clean_funk_guitar",
            "prompt": "Clean rhythmic electric guitar comping, Stratocaster neck pickup, tight 16th-note muting, dry analog studio direct inject",
            "negative": "heavy overdrive, heavy distortion, long reverb tails, messy strumming",
        },
    ],
    "lead": [
        {
            "id": "v1_vintage_synth_lead",
            "prompt": "Singing monophonic analog synthesizer lead, Minimoog ladder filter warmth, smooth portamento, musical vibrato",
            "negative": "harsh digital aliasing, polyphonic bleed, pitch drift, noisy artifacts",
        }
    ],
    "pads": [
        {
            "id": "v1_warm_analog_pad",
            "prompt": "Lush warm polyphonic analog synthesizer pad, slow attack, Roland Juno-106 chorus warmth, wide stereo field",
            "negative": "harsh brass, staccato, muddy low end, thin mono texture",
        }
    ],
}


def generate_prompt_variants(role: str, house_style_override: str = "") -> List[PromptVariant]:
    """Return a curated set of 2 to 3 prompt variants for a given instrument role."""
    templates = ROLE_PROMPT_TEMPLATES.get(role, ROLE_PROMPT_TEMPLATES["comp"])
    variants: List[PromptVariant] = []

    for t in templates:
        prompt_text = t["prompt"]
        if house_style_override.strip():
            prompt_text = f"{prompt_text}, {house_style_override.strip()}"

        variants.append(
            PromptVariant(
                variant_id=t["id"],
                prompt=prompt_text,
                negative_prompt=t["negative"],
            )
        )
    return variants
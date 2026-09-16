"""Audio-text semantic alignment scoring using contrastive audio representations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
import numpy as np
from pydantic import BaseModel, Field

try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False


class ClapSimilarityResult(BaseModel):
    """Evaluation result for candidate audio prompt alignment."""

    prompt_similarity: float = Field(ge=-1.0, le=1.0)
    negative_prompt_similarity: float = Field(ge=-1.0, le=1.0, default=0.0)
    net_score: float = Field(ge=-2.0, le=1.0)


class ClapScorer:
    """Evaluates semantic audio-text alignment against positive and negative prompts."""

    def __init__(self, model_fp: Optional[str] = None):
        self.model = None
        if CLAP_AVAILABLE:
            try:
                self.model = laion_clap.CLAP_Module(enable_fusion=False)
                if model_fp and Path(model_fp).is_file():
                    self.model.load_ckpt(model_fp)
                else:
                    self.model.load_ckpt()
            except Exception:
                self.model = None

    def embed_text(self, text: str) -> np.ndarray:
        """Extract L2-normalized text embedding vector."""
        if self.model is not None:
            emb = self.model.get_text_embedding([text])
            norm = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8
            return (emb / norm)[0]

        # Deterministic surrogate embedding for CI/offline environments
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(512).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def embed_audio(self, audio_path: Path | str) -> np.ndarray:
        """Extract L2-normalized audio embedding vector."""
        if self.model is not None:
            emb = self.model.get_audio_embedding_from_filelist([str(audio_path)])
            norm = np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8
            return (emb / norm)[0]

        # Deterministic spectral surrogate for CI/offline environments
        import librosa

        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        if len(y) == 0:
            return np.zeros(512, dtype=np.float32)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
        mean_mfcc = np.mean(mfcc, axis=1)
        vec = np.tile(mean_mfcc, 16)
        return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)

    def evaluate_candidate(
        self,
        audio_path: Path | str,
        prompt: str,
        negative_prompt: str = "",
    ) -> ClapSimilarityResult:
        """Score candidate audio by subtracting negative prompt similarity from positive similarity."""
        audio_emb = self.embed_audio(audio_path)
        prompt_emb = self.embed_text(prompt)

        pos_sim = float(np.dot(audio_emb, prompt_emb))

        neg_sim = 0.0
        if negative_prompt.strip():
            neg_emb = self.embed_text(negative_prompt)
            neg_sim = float(np.dot(audio_emb, neg_emb))

        net_score = pos_sim - (0.5 * neg_sim)

        return ClapSimilarityResult(
            prompt_similarity=pos_sim,
            negative_prompt_similarity=neg_sim,
            net_score=net_score,
        )
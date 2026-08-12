"""Paths, dates, population definition and model settings.

Nothing else inlines a cutoff date, a party filter or a model id. They live
here so one object describes the whole run.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from plp_sim.schemas import PLP_PARTIES

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---------------------------------------------------------------- paths
    root: Path = ROOT
    data_raw: Path = ROOT / "data" / "raw"
    data_manual: Path = ROOT / "data" / "manual"
    data_interim: Path = ROOT / "data" / "interim"
    data_processed: Path = ROOT / "data" / "processed"
    outputs: Path = ROOT / "outputs"
    cache_dir: Path = ROOT / ".cache"

    # ---------------------------------------------------------------- dates
    # Load-bearing. Gates every construction input and defines the holdout.
    # Must precede the earliest validation event; holdout.py asserts this.
    cutoff_date: dt.date = dt.date(2026, 4, 1)

    # ------------------------------------------------- population definition
    # A modelling choice, stated explicitly rather than buried in a filter.
    plp_parties: tuple[str, ...] = PLP_PARTIES
    #: Ex-Labour MPs now sitting as Independent / Your Party / Restore Britain.
    #: Excluded by default, but they are the most behaviourally extreme cases on
    #: exactly the question being asked, so the exclusion is recorded and
    #: reported rather than silent. Flip to include them in a sensitivity run.
    include_defectors: bool = False
    exclude_speaker: bool = True

    # ---------------------------------------------------------------- frames
    n_personas: int = 100
    random_seed: int = 0
    rake_max_iter: int = 200
    rake_tolerance: float = 1e-6

    # ----------------------------------------------------------------- model
    openai_api_key: str = Field(default="", repr=False)
    #: Must be a Chat Completions model that returns `logprobs`/`top_logprobs`.
    #: elicit.py probes this at startup rather than trusting it.
    model: str = "gpt-4o-mini"
    top_logprobs: int = 20
    max_concurrency: int = 8
    max_retries: int = 5
    #: Sampling is used only for dispersion calibration, never for elicitation.
    dispersion_temperature: float = 1.0
    dispersion_draws: int = 20

    prompt_version: str = "v1"

    @field_validator("cutoff_date", mode="before")
    @classmethod
    def _parse_date(cls, v: object) -> object:
        if isinstance(v, str):
            return dt.date.fromisoformat(v)
        return v

    def ensure_dirs(self) -> None:
        for p in (
            self.data_raw, self.data_manual, self.data_interim,
            self.data_processed, self.outputs, self.cache_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, object]:
        """Run provenance, for the run log."""
        return {
            "cutoff_date": self.cutoff_date.isoformat(),
            "plp_parties": list(self.plp_parties),
            "include_defectors": self.include_defectors,
            "n_personas": self.n_personas,
            "random_seed": self.random_seed,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

"""What a Lean 4 domain provides.

A problem is a Lean 4 declaration cut in two: the model sees the header
(`prompt`) and writes the rest (`answer`); `prompt + " " + answer` is the
whole declaration, which must compile. Each domain supplies:

  * a generator of problems, each with a reference answer that checks;
  * a checker: a Python implementation of the fragment of Lean 4 the domain
    uses. It must be *sound with respect to Lean*: it may reject an answer
    Lean accepts, never the reverse. lean_oracle.py and the tests compile
    declarations with a real Lean 4 toolchain to check this;
  * a canonical form of the prompt (`split_key`), so that two spellings of
    the same problem always land in the same train/val/test split;
  * documents for pretraining: whole declarations, with varied but valid
    proofs.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    domain: str
    prompt: str       # the declaration header, e.g. "example (p : Prop) : p → p :="
    reference: str    # an answer that checks, e.g. "fun h => h"


class Domain(ABC):
    name: str = ""
    # Whether, between two answers that both check, the shorter one is
    # preferred (true for proofs; for computed values every correct answer is
    # equally good).
    prefer_shorter: bool = False

    @abstractmethod
    def claims(self, prompt: str) -> bool:
        """Whether `prompt` is a header of this domain (well-formed, within the
        domain's limits). Domains claim disjoint sets of prompts."""

    @abstractmethod
    def sample(self, rng: random.Random) -> Problem:
        """A random problem with a reference answer that verifies."""

    @abstractmethod
    def verify(self, prompt: str, answer: str) -> bool:
        """Whether `prompt + " " + answer` is a declaration Lean 4 accepts
        (soundly: False whenever unsure). Never raises on any strings."""

    @abstractmethod
    def document(self, problem: Problem, rng: random.Random) -> str:
        """A whole declaration for pretraining: the prompt, a space, and a
        valid answer (the reference or an equivalent variant)."""

    def split_key(self, prompt: str) -> str:
        """Canonical form of a prompt for the train/val/test split. Two prompts
        that are the same problem up to renaming or respelling must map to the
        same key."""
        return prompt

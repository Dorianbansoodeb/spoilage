"""Completion backends. The organisms run offline; live APIs are optional."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from spoilage.behavior.organism import ORGANISM_A, ORGANISM_B, Organism
from spoilage.behavior.probes import Probe


@dataclass
class Completion:
    text: str
    text_b: str = ""
    backend: str = ""
    live: bool = False


class Backend:
    name = "base"
    live = False

    def complete(self, system: str, user: str, *, probe: Probe | None = None) -> Completion:
        raise NotImplementedError

    def complete_pair(self, probe: Probe, system: str, user: str, user_b: str) -> Completion:
        a = self.complete(system, user, probe=probe)
        b = self.complete(system, user_b, probe=probe)
        return Completion(text=a.text, text_b=b.text, backend=self.name, live=self.live)


class OrganismBackend(Backend):
    def __init__(self, organism: Organism) -> None:
        self.organism = organism
        self.name = organism.name

    def complete(self, system: str, user: str, *, probe: Probe | None = None) -> Completion:
        return Completion(
            text=self.organism.complete(system, user, probe=probe),
            backend=self.name,
        )

    def complete_pair(self, probe: Probe, system: str, user: str, user_b: str) -> Completion:
        a, b = self.organism.complete_pair(probe, system, user, user_b)
        return Completion(text=a, text_b=b, backend=self.name)


class OpenAIBackend(Backend):
    name = "openai"
    live = True

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def complete(self, system: str, user: str, *, probe: Probe | None = None) -> Completion:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"openai request failed: {exc}") from exc
        text = payload["choices"][0]["message"]["content"]
        return Completion(text=text, backend=self.name, live=True)


class AnthropicBackend(Backend):
    name = "anthropic"
    live = True

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def complete(self, system: str, user: str, *, probe: Probe | None = None) -> Completion:
        body = {
            "model": self.model,
            "max_tokens": 256,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"anthropic request failed: {exc}") from exc
        text = "".join(part.get("text", "") for part in payload.get("content", []))
        return Completion(text=text, backend=self.name, live=True)


def available_backends() -> dict[str, bool]:
    return {
        "organism-a": True,
        "organism-b": True,
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


def get_backend(name: str) -> Backend:
    key = (name or "organism-a").lower()
    if key in {"organism-a", "a", "organism"}:
        return OrganismBackend(ORGANISM_A)
    if key in {"organism-b", "b"}:
        return OrganismBackend(ORGANISM_B)
    if key == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIBackend()
    if key == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return AnthropicBackend()
    raise KeyError(f"unknown backend '{name}'")

#!/usr/bin/env python3
"""Versioned prompt templates for STS2 model harnesses."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any


Json = dict[str, Any]

PROMPT_ID = "sts2_mcp_action"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class PromptTemplateInfo:
    name: str
    version: str
    path: Path
    text: str
    sha256: str

    def metadata(self) -> Json:
        return {
            "version": self.version,
            "path": repo_relative_path(self.path),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PromptBundle:
    prompt_id: str
    system: PromptTemplateInfo
    user: PromptTemplateInfo

    def metadata(self) -> Json:
        return {
            "prompt_id": self.prompt_id,
            "system": self.system.metadata(),
            "user": self.user.metadata(),
        }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_template(name: str, version: str, suffix: str) -> PromptTemplateInfo:
    path = PROMPT_DIR / f"{PROMPT_ID}_{name}.{version}.{suffix}"
    text = path.read_text(encoding="utf-8")
    return PromptTemplateInfo(
        name=name,
        version=version,
        path=path,
        text=text,
        sha256=sha256_text(text),
    )


def load_sts2_mcp_action_prompts(
    *,
    system_version: str = "v1",
    user_version: str = "v1",
) -> PromptBundle:
    return PromptBundle(
        prompt_id=PROMPT_ID,
        system=load_template("system", system_version, "md"),
        user=load_template("user", user_version, "txt"),
    )


def render_system_prompt(bundle: PromptBundle, *, game_context: str) -> str:
    return Template(bundle.system.text).substitute(game_context=game_context).strip()


def render_user_prompt(
    bundle: PromptBundle,
    *,
    step: int,
    valid_actions_json: str,
    recent_history: str,
    state_json: str,
) -> str:
    return Template(bundle.user.text).substitute(
        step=step,
        valid_actions_json=valid_actions_json,
        recent_history=recent_history,
        state_json=state_json,
    ).strip()

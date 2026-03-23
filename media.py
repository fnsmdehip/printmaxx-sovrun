"""
Media -- smart media generation router.

Picks the best tool based on budget, infrastructure, and task requirements.
Routes across image, video, TTS, voice agent, and music generation with
automatic fallback chains.

Stdlib + httpx (via deps.py). Each provider is a standalone class.
JSONL audit trail for all generation. Configurable output directory.

Usage:
    from sovrun.core.media import MediaRouter

    router = MediaRouter(budget_tier="free")
    result = router.generate_image("a futuristic city at sunset")
    result = router.text_to_speech("Welcome to our service", voice="narrator")
    result = router.generate_thumbnail("10x Your Revenue", subtitle="The Guide")
    result = router.generate_video("product demo for SaaS tool")

CLI:
    python3 -m sovrun.core.media --image "a futuristic city at sunset"
    python3 -m sovrun.core.media --video "product demo for SaaS tool"
    python3 -m sovrun.core.media --tts "Welcome to our service" --voice daniel
    python3 -m sovrun.core.media --providers
    python3 -m sovrun.core.media --budget mid
    python3 -m sovrun.core.media --capabilities
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from .media_providers.base import (
    AUDIT_LOG,
    MEDIA_DIR,
    MediaProvider,
    ProviderResult,
    audit_media,
)

logger = logging.getLogger("sovrun.media")

# ---------------------------------------------------------------------------
# Budget tier ordering
# ---------------------------------------------------------------------------

_BUDGET_ORDER = {"free": 0, "low": 1, "mid": 2, "high": 3}


def _budget_allows(provider_tier: str, user_tier: str) -> bool:
    """Check if user's budget tier includes this provider's tier."""
    return _BUDGET_ORDER.get(provider_tier, 99) <= _BUDGET_ORDER.get(user_tier, 0)


# ---------------------------------------------------------------------------
# Provider registry -- all known providers
# ---------------------------------------------------------------------------

def _load_providers() -> list[MediaProvider]:
    """Instantiate all available providers. Import failures are swallowed."""
    providers: list[MediaProvider] = []

    try:
        from .media_providers.edge_tts_provider import EdgeTTSProvider
        providers.append(EdgeTTSProvider())
    except Exception:
        pass

    try:
        from .media_providers.openai_media import OpenAIMediaProvider
        providers.append(OpenAIMediaProvider())
    except Exception:
        pass

    try:
        from .media_providers.replicate_provider import ReplicateProvider
        providers.append(ReplicateProvider())
    except Exception:
        pass

    try:
        from .media_providers.playwright_screenshot import PlaywrightScreenshotProvider
        providers.append(PlaywrightScreenshotProvider())
    except Exception:
        pass

    try:
        from .media_providers.bland_ai import BlandAIProvider
        providers.append(BlandAIProvider())
    except Exception:
        pass

    return providers


# ---------------------------------------------------------------------------
# MediaRouter
# ---------------------------------------------------------------------------

class MediaRouter:
    """Smart media generation router.

    Picks the best available provider for each task based on:
    - Budget tier (free/low/mid/high)
    - GPU availability (for local models)
    - Provider availability (API keys, packages installed)
    - Fallback chain: if primary fails, tries next best

    Args:
        budget_tier: "free", "low" ($0-30/mo), "mid" ($30-100/mo), "high" ($100+/mo)
        has_gpu: whether local GPU is available for Stable Diffusion, Whisper, etc.
    """

    def __init__(
        self,
        budget_tier: str = "free",
        has_gpu: bool = False,
    ) -> None:
        self.budget_tier = budget_tier
        self.has_gpu = has_gpu
        self._providers = _load_providers()

    def _get_providers(self, task_type: str) -> list[MediaProvider]:
        """Get providers for a task type, filtered by budget and availability.

        Returns providers sorted by preference:
        1. Available and within budget
        2. Free providers first, then ascending cost
        3. GPU providers only if has_gpu is True
        """
        candidates: list[tuple[int, float, MediaProvider]] = []

        for p in self._providers:
            if task_type not in p.task_types:
                continue
            if not _budget_allows(p.budget_tier, self.budget_tier):
                continue
            if p.needs_gpu and not self.has_gpu:
                continue
            if not p.is_available():
                continue

            tier_val = _BUDGET_ORDER.get(p.budget_tier, 99)
            cost = p.get_cost(task_type)
            candidates.append((tier_val, cost, p))

        # Sort by tier (free first), then by cost
        candidates.sort(key=lambda x: (x[0], x[1]))
        return [c[2] for c in candidates]

    def route(self, task_type: str, **kwargs: Any) -> ProviderResult:
        """Auto-select best provider and generate. Falls back on failure."""
        providers = self._get_providers(task_type)
        if not providers:
            return ProviderResult(
                success=False, task_type=task_type,
                error=f"no available provider for '{task_type}' at budget '{self.budget_tier}'",
            )

        last_error = ""
        for provider in providers:
            logger.info("trying provider %s for %s", provider.name, task_type)
            result = provider.generate(task_type, **kwargs)
            if result.success:
                return result
            last_error = result.error or "unknown error"
            logger.warning(
                "provider %s failed for %s: %s", provider.name, task_type, last_error
            )

        return ProviderResult(
            success=False, task_type=task_type,
            error=f"all providers failed for '{task_type}'. last error: {last_error}",
        )

    # -----------------------------------------------------------------------
    # Convenience methods
    # -----------------------------------------------------------------------

    def generate_image(
        self, prompt: str, style: str | None = None, size: str = "1024x1024",
        **kwargs: Any,
    ) -> ProviderResult:
        """Generate an image using the best available provider."""
        return self.route("image", prompt=prompt, style=style, size=size, **kwargs)

    def generate_thumbnail(
        self, title: str, subtitle: str = "", template: str = "default",
        size: str = "1280x720",
    ) -> ProviderResult:
        """Generate a YouTube/social thumbnail via Playwright HTML rendering."""
        # Try Playwright screenshot directly for thumbnails (zero cost)
        for p in self._providers:
            if p.name == "playwright_screenshot" and p.is_available():
                from .media_providers.playwright_screenshot import (
                    PlaywrightScreenshotProvider,
                )
                if isinstance(p, PlaywrightScreenshotProvider):
                    return p.generate_thumbnail(title, subtitle, template, size)

        # Fallback: generate via image route with descriptive prompt
        return self.route(
            "image",
            prompt=f"YouTube thumbnail: {title}. {subtitle}. Bold text, dark background.",
            size=size,
        )

    def screenshot_to_image(
        self, html_content: str, size: str = "1280x720", **kwargs: Any,
    ) -> ProviderResult:
        """Render HTML content to an image via Playwright (zero cost)."""
        return self.route("image", html_content=html_content, size=size, **kwargs)

    def generate_video(
        self, prompt: str, duration: int = 10, style: str | None = None,
        **kwargs: Any,
    ) -> ProviderResult:
        """Generate a video using the best available provider."""
        return self.route(
            "video", prompt=prompt, duration=duration, style=style, **kwargs,
        )

    def generate_short(
        self, script: str, voiceover: bool = True, **kwargs: Any,
    ) -> ProviderResult:
        """Generate a TikTok/YouTube Short with optional TTS voiceover.

        If voiceover is True, generates TTS first, then video.
        Returns the video result with TTS path in metadata.
        """
        tts_path = None
        if voiceover:
            tts_result = self.text_to_speech(script)
            if tts_result.success:
                tts_path = tts_result.output_path

        video_result = self.route(
            "video", prompt=script, duration=kwargs.get("duration", 15),
            **kwargs,
        )
        if tts_path and video_result.success:
            video_result.metadata["tts_path"] = tts_path

        return video_result

    def text_to_speech(
        self, text: str, voice: str | None = None, provider: str | None = None,
        **kwargs: Any,
    ) -> ProviderResult:
        """Convert text to speech using the best available provider."""
        if provider:
            # Force specific provider
            for p in self._providers:
                if p.name == provider and "tts" in p.task_types:
                    return p.generate("tts", text=text, voice=voice or "narrator", **kwargs)

        return self.route("tts", text=text, voice=voice or "narrator", **kwargs)

    def clone_voice(
        self, sample_audio: str, **kwargs: Any,
    ) -> ProviderResult:
        """Clone a voice from a sample audio file."""
        # Voice cloning is only available via ElevenLabs or specialized providers
        return ProviderResult(
            success=False, task_type="voice_clone",
            error="voice cloning requires ElevenLabs or Fish Audio provider (not yet implemented)",
        )

    def generate_music(
        self, prompt: str, duration: int = 30, genre: str | None = None,
        **kwargs: Any,
    ) -> ProviderResult:
        """Generate background music."""
        full_prompt = prompt
        if genre:
            full_prompt = f"{genre} style: {prompt}"
        return self.route("music", prompt=full_prompt, duration=duration, **kwargs)

    def setup_voice_agent(
        self, phone_number: str, script: str, provider: str | None = None,
        **kwargs: Any,
    ) -> ProviderResult:
        """Configure an inbound voice agent on a phone number."""
        return self.route(
            "voice_agent", action="setup",
            phone_number=phone_number, script=script, **kwargs,
        )

    def make_call(
        self, to_number: str, script: str, **kwargs: Any,
    ) -> ProviderResult:
        """Make an outbound voice agent call."""
        return self.route(
            "voice_agent", action="call",
            to_number=to_number, script=script, **kwargs,
        )

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def capabilities(self) -> dict[str, list[dict[str, Any]]]:
        """Show what's available at the current budget tier."""
        result: dict[str, list[dict[str, Any]]] = {}
        for p in self._providers:
            for task_type in p.task_types:
                within_budget = _budget_allows(p.budget_tier, self.budget_tier)
                available = p.is_available()
                gpu_ok = not p.needs_gpu or self.has_gpu

                entry = {
                    "provider": p.name,
                    "budget_tier": p.budget_tier,
                    "needs_gpu": p.needs_gpu,
                    "available": available,
                    "within_budget": within_budget,
                    "gpu_ok": gpu_ok,
                    "usable": available and within_budget and gpu_ok,
                    "est_cost": p.get_cost(task_type),
                }
                result.setdefault(task_type, []).append(entry)

        return result

    def list_providers(self) -> list[dict[str, Any]]:
        """List all providers with their status."""
        return [
            {
                "name": p.name,
                "task_types": p.task_types,
                "budget_tier": p.budget_tier,
                "needs_gpu": p.needs_gpu,
                "available": p.is_available(),
            }
            for p in self._providers
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_image(args: argparse.Namespace) -> None:
    router = MediaRouter(budget_tier=args.budget, has_gpu=args.gpu)
    result = router.generate_image(args.image, size=args.size or "1024x1024")
    _print_result(result)


def _cli_video(args: argparse.Namespace) -> None:
    router = MediaRouter(budget_tier=args.budget, has_gpu=args.gpu)
    result = router.generate_video(args.video, duration=args.duration or 10)
    _print_result(result)


def _cli_tts(args: argparse.Namespace) -> None:
    router = MediaRouter(budget_tier=args.budget, has_gpu=args.gpu)
    result = router.text_to_speech(args.tts, voice=args.voice)
    _print_result(result)


def _cli_thumbnail(args: argparse.Namespace) -> None:
    router = MediaRouter(budget_tier=args.budget, has_gpu=args.gpu)
    result = router.generate_thumbnail(args.thumbnail, subtitle=args.subtitle or "")
    _print_result(result)


def _cli_music(args: argparse.Namespace) -> None:
    router = MediaRouter(budget_tier=args.budget, has_gpu=args.gpu)
    result = router.generate_music(args.music, duration=args.duration or 30)
    _print_result(result)


def _cli_providers(args: argparse.Namespace) -> None:
    router = MediaRouter(budget_tier=args.budget, has_gpu=args.gpu)
    providers = router.list_providers()

    print(f"\n=== Media Providers (budget: {args.budget}, gpu: {args.gpu}) ===\n")
    for p in providers:
        status = "READY" if p["available"] else "UNAVAILABLE"
        gpu_tag = " [GPU]" if p["needs_gpu"] else ""
        types = ", ".join(p["task_types"])
        print(f"  [{status:11s}] {p['name']:25s}  tier={p['budget_tier']:4s}{gpu_tag}  ({types})")
    print()


def _cli_capabilities(args: argparse.Namespace) -> None:
    router = MediaRouter(budget_tier=args.budget, has_gpu=args.gpu)
    caps = router.capabilities()

    print(f"\n=== Capabilities (budget: {args.budget}, gpu: {args.gpu}) ===\n")
    for task_type, providers in sorted(caps.items()):
        print(f"  {task_type}:")
        for p in providers:
            usable = "OK" if p["usable"] else "--"
            reason = ""
            if not p["available"]:
                reason = " (not installed/configured)"
            elif not p["within_budget"]:
                reason = f" (needs {p['budget_tier']} tier)"
            elif not p["gpu_ok"]:
                reason = " (needs GPU)"
            cost = f"${p['est_cost']:.3f}" if p["est_cost"] > 0 else "free"
            print(f"    [{usable:2s}] {p['provider']:25s}  {cost:>8s}{reason}")
        print()


def _print_result(result: ProviderResult) -> None:
    if result.success:
        print(f"\nSUCCESS via {result.provider}")
        if result.output_path:
            print(f"  Output: {result.output_path}")
        if result.cost_usd > 0:
            print(f"  Cost: ${result.cost_usd:.4f}")
        if result.metadata:
            for k, v in result.metadata.items():
                print(f"  {k}: {v}")
    else:
        print(f"\nFAILED: {result.error}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Media -- smart media generation router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Generation commands
    parser.add_argument("--image", metavar="PROMPT",
                        help="Generate an image from a text prompt")
    parser.add_argument("--video", metavar="PROMPT",
                        help="Generate a video from a text prompt")
    parser.add_argument("--tts", metavar="TEXT",
                        help="Convert text to speech")
    parser.add_argument("--thumbnail", metavar="TITLE",
                        help="Generate a thumbnail image")
    parser.add_argument("--music", metavar="PROMPT",
                        help="Generate background music")

    # Options
    parser.add_argument("--voice", default="narrator",
                        help="Voice for TTS (default: narrator)")
    parser.add_argument("--size", default=None,
                        help="Image size, e.g. 1024x1024")
    parser.add_argument("--duration", type=int, default=None,
                        help="Duration in seconds for video/music")
    parser.add_argument("--subtitle", default="",
                        help="Subtitle for thumbnail")

    # Config
    parser.add_argument("--budget", default="free",
                        choices=["free", "low", "mid", "high"],
                        help="Budget tier (default: free)")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable local GPU providers")

    # Info
    parser.add_argument("--providers", action="store_true",
                        help="List all configured providers")
    parser.add_argument("--capabilities", action="store_true",
                        help="Show available capabilities at current budget")

    args = parser.parse_args()

    has_action = any([
        args.image, args.video, args.tts, args.thumbnail,
        args.music, args.providers, args.capabilities,
    ])
    if not has_action:
        parser.print_help()
        return

    if args.providers:
        _cli_providers(args)
    if args.capabilities:
        _cli_capabilities(args)
    if args.image:
        _cli_image(args)
    if args.video:
        _cli_video(args)
    if args.tts:
        _cli_tts(args)
    if args.thumbnail:
        _cli_thumbnail(args)
    if args.music:
        _cli_music(args)


if __name__ == "__main__":
    main()

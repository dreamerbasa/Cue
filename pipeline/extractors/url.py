import logging
import re
from urllib.parse import urlparse, parse_qs

import requests
import trafilatura
from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

_PLATFORM_MAP = [
    ("instagram.com", "instagram"),
    ("youtube.com", "youtube"),
    ("youtu.be", "youtube"),
    ("substack.com", "substack"),
    ("linkedin.com", "linkedin"),
    ("medium.com", "medium"),
]

_AUTH_REDIRECT_PATHS = ["/login", "/signup", "/register", "/auth", "/uas/login"]

_JUNK_SIGNATURES = [
    "agree & join linkedin",
    "already on linkedin? sign in",
    "linkedin's user agreement, privacy policy",
    "looking to create a page for a business",
]


def _is_wall_content(text: str) -> bool:
    """Returns True if 2+ junk signatures found — likely a login wall served with 200 status."""
    if not text:
        return False
    text_lower = text.lower()
    hits = sum(1 for sig in _JUNK_SIGNATURES if sig in text_lower)
    return hits >= 2


def get_url_display_title(url: str, platform: str) -> str:
    """Generate a human-readable fallback title when extraction fails."""
    platform_titles = {
        "linkedin": "LinkedIn post",
        "instagram": "Instagram post",
        "medium": "Medium article",
        "substack": "Substack article",
        "youtube": "YouTube video",
    }
    if platform in platform_titles:
        return platform_titles[platform]
    try:
        domain = urlparse(url).netloc.replace("www.", "")
        return f"Page from {domain}"
    except Exception:
        return "Web page"


def _detect_platform(url: str) -> str:
    for domain, platform in _PLATFORM_MAP:
        if domain in url:
            return platform
    return "web"


def _extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if "youtu.be" in parsed.hostname:
        return parsed.path.lstrip("/")
    return parse_qs(parsed.query).get("v", [None])[0]


def _extract_youtube(url: str) -> str:
    title, author = None, None
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        title = data.get("title")
        author = data.get("author_name")
    except Exception:
        pass

    transcript = None
    video_id = _extract_video_id(url)
    if video_id:
        try:
            api = YouTubeTranscriptApi()
            result = api.fetch(video_id)
            full = " ".join(s.text for s in result.snippets)
            words = full.split()
            if len(words) > 1000:
                full = " ".join(words[:1000])
            transcript = full
        except Exception:
            pass

    if title and transcript:
        return f"Title: {title} by {author}\n\nTranscript:\n{transcript}"
    if title:
        return f"Title: {title} by {author}"
    return f"YouTube video (could not extract details). URL: {url}"


def _extract_medium(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        final_url = resp.url.lower()
        if any(path in final_url for path in _AUTH_REDIRECT_PATHS):
            logger.info(f"Extraction blocked (redirect to auth): {url} → {resp.url}")
            return None

        text = trafilatura.extract(resp.text, favor_recall=True)
        if not text:
            return None
        words = text.split()
        if len(words) > 1000:
            text = " ".join(words[:1000])
        return text
    except Exception:
        return None


def _extract_web(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()

        final_url = resp.url.lower()
        if any(path in final_url for path in _AUTH_REDIRECT_PATHS):
            logger.info(f"Extraction blocked (redirect to auth): {url} → {resp.url}")
            return None

        text = trafilatura.extract(resp.text, favor_recall=True)
        if not text:
            logger.info(f"Extraction blocked (trafilatura returned None): {url}")
            return None

        if _is_wall_content(text):
            logger.info(f"Extraction blocked (wall signatures matched): {url}")
            return None

        words = text.split()
        if len(words) > 1000:
            text = " ".join(words[:1000])
        return text
    except Exception:
        logger.info(f"Extraction blocked (request failed): {url}")
        return None


def find_urls(raw_content: str) -> list:
    return _URL_PATTERN.findall(raw_content)


def _failure_reason(platform: str) -> str:
    if platform == "linkedin":
        return "Couldn't read this page — LinkedIn requires sign-in to view posts"
    if platform == "medium":
        return "Couldn't read this page — Medium requires sign-in for some articles"
    return "Couldn't read this page — it may be behind a paywall or unavailable"


def extract_single(url: str, user_note: str = None) -> dict:
    source_platform = _detect_platform(url)

    if source_platform == "instagram":
        return {
            "content_type": "url",
            "raw_content": url,
            "extracted_text": None,
            "source_platform": "instagram",
            "url": url,
            "user_note": user_note,
            "needs_screenshot": True,
        }

    if source_platform == "medium":
        extracted_text = _extract_medium(url)
        if not extracted_text:
            return {
                "content_type": "url",
                "raw_content": url,
                "extracted_text": None,
                "source_platform": "medium",
                "url": url,
                "user_note": user_note,
                "needs_screenshot": True,
                "extraction_failed": True,
                "failure_reason": _failure_reason("medium"),
            }

    if source_platform == "youtube":
        extracted_text = _extract_youtube(url)
    elif source_platform != "medium":
        extracted_text = _extract_web(url)

    if extracted_text is None and source_platform not in ("instagram", "medium"):
        return {
            "content_type": "url",
            "raw_content": url,
            "extracted_text": None,
            "source_platform": source_platform,
            "url": url,
            "user_note": user_note,
            "needs_screenshot": True,
            "extraction_failed": True,
            "failure_reason": _failure_reason(source_platform),
        }

    if user_note and extracted_text:
        extracted_text = f"User note: {user_note}\n\n{extracted_text}"

    return {
        "content_type": "url",
        "raw_content": url,
        "extracted_text": extracted_text,
        "source_platform": source_platform,
        "url": url,
        "user_note": user_note,
    }


def extract(raw_content: str) -> dict:
    match = _URL_PATTERN.search(raw_content)
    if not match:
        return {
            "content_type": "url",
            "raw_content": raw_content,
            "extracted_text": f"No URL found in message. Original text: {raw_content}",
            "source_platform": "web",
            "url": None,
        }

    url = match.group(0)
    source_platform = _detect_platform(url)
    user_note = raw_content.replace(url, "").strip() or None

    if source_platform == "instagram":
        return {
            "content_type": "url",
            "raw_content": raw_content,
            "extracted_text": None,
            "source_platform": "instagram",
            "url": url,
            "user_note": user_note,
            "needs_screenshot": True,
        }

    if source_platform == "medium":
        extracted_text = _extract_medium(url)
        if not extracted_text:
            return {
                "content_type": "url",
                "raw_content": raw_content,
                "extracted_text": None,
                "source_platform": "medium",
                "url": url,
                "user_note": user_note,
                "needs_screenshot": True,
                "extraction_failed": True,
                "failure_reason": _failure_reason("medium"),
            }

    if source_platform == "youtube":
        extracted_text = _extract_youtube(url)
    elif source_platform != "medium":
        extracted_text = _extract_web(url)

    if extracted_text is None and source_platform not in ("instagram", "medium"):
        return {
            "content_type": "url",
            "raw_content": raw_content,
            "extracted_text": None,
            "source_platform": source_platform,
            "url": url,
            "user_note": user_note,
            "needs_screenshot": True,
            "extraction_failed": True,
            "failure_reason": _failure_reason(source_platform),
        }

    if user_note and extracted_text:
        extracted_text = f"User note: {user_note}\n\n{extracted_text}"

    return {
        "content_type": "url",
        "raw_content": raw_content,
        "extracted_text": extracted_text,
        "source_platform": source_platform,
        "url": url,
        "user_note": user_note,
    }

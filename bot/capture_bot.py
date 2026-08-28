import io
import logging
import tempfile

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ForceReply

logger = logging.getLogger(__name__)
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, ConversationHandler, filters,
)

from config import TELEGRAM_BOT_TOKEN, AUTHORIZED_USER_IDS
from pipeline.router import process_message
from pipeline.extractors import text, vision, whisper
from pipeline.classifier import classify
from db.queries import (
    update_item_rating, get_item, upsert_user, get_user_by_telegram_id,
    update_last_active, set_user_active, update_reminder_time, update_nudge_time,
    archive_item, done_item, remind_later, keep_item, get_image_bytes,
    get_pending_items, set_remind_tonight, set_go_deep,
    get_categories_with_counts, search_items, get_user_stats,
    update_item_embedding, update_user_email, update_item_from_context,
)
import asyncio

from intelligence.scoring import _get_emoji, _extract_url, _PRIORITY_MATRIX
from intelligence.embeddings import build_embedding_text, generate_embedding
from db.queries import update_item_embedding
from notifications.daily_nudge import build_list_view, build_detail_view, escape_html, _list_line
from notifications.nudge_session import get_session, set_session, REVIEW_PAGE_SIZE

_pending_context: dict[int, str] = {}


def _is_authorized(update: Update) -> bool:
    return update.effective_user.id in AUTHORIZED_USER_IDS



def _get_user_id(update: Update) -> str:
    update_last_active(update.effective_user.id)
    user = get_user_by_telegram_id(update.effective_user.id)
    if not user:
        user = upsert_user(
            telegram_user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            display_name=update.effective_user.first_name,
        )
    if not user:
        raise ValueError("Could not create or find user")
    return user["id"]


_INTEREST_EMOJI = {3: "\U0001f525", 2: "\U0001f44d", 1: "\U0001f937"}
_INTEREST_LABEL = {3: "\U0001f525 High", 2: "\U0001f44d Medium", 1: "\U0001f937 Low"}
_GOAL_EMOJI = {3: "\U0001f3af", 2: "↔️", 1: "❌"}
_GOAL_LABEL = {3: "\U0001f3af Aligned", 2: "↔️ Somewhat", 1: "❌ Nope"}


def _rating_keyboard(item_id: str, include_context: bool = False) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton("\U0001f525 High", callback_data=f"interest_3_{item_id}"),
            InlineKeyboardButton("\U0001f44d Medium", callback_data=f"interest_2_{item_id}"),
            InlineKeyboardButton("\U0001f937 Low", callback_data=f"interest_1_{item_id}"),
        ],
        [
            InlineKeyboardButton("\U0001f3af Aligned", callback_data=f"goal_3_{item_id}"),
            InlineKeyboardButton("↔️ Somewhat", callback_data=f"goal_2_{item_id}"),
            InlineKeyboardButton("❌ Nope", callback_data=f"goal_1_{item_id}"),
        ],
        [
            InlineKeyboardButton("⏰ Remind tonight", callback_data=f"remind_tonight_{item_id}"),
        ],
    ]
    if include_context:
        rows.append([
            InlineKeyboardButton("📎 Add screenshot or note", callback_data=f"ctx_{item_id}"),
        ])
    return InlineKeyboardMarkup(rows)


def _remaining_keyboard(item_id: str, interest_rated: bool, goal_rated: bool,
                         remind_set: bool, include_context: bool = False) -> InlineKeyboardMarkup | None:
    rows = []
    if not interest_rated:
        rows.append([
            InlineKeyboardButton("\U0001f525 High", callback_data=f"interest_3_{item_id}"),
            InlineKeyboardButton("\U0001f44d Medium", callback_data=f"interest_2_{item_id}"),
            InlineKeyboardButton("\U0001f937 Low", callback_data=f"interest_1_{item_id}"),
        ])
    if not goal_rated:
        rows.append([
            InlineKeyboardButton("\U0001f3af Aligned", callback_data=f"goal_3_{item_id}"),
            InlineKeyboardButton("↔️ Somewhat", callback_data=f"goal_2_{item_id}"),
            InlineKeyboardButton("❌ Nope", callback_data=f"goal_1_{item_id}"),
        ])
    if not remind_set:
        rows.append([
            InlineKeyboardButton("⏰ Remind tonight", callback_data=f"remind_tonight_{item_id}"),
        ])
    if include_context:
        rows.append([
            InlineKeyboardButton("📎 Add screenshot or note", callback_data=f"ctx_{item_id}"),
        ])
    return InlineKeyboardMarkup(rows) if rows else None


def _build_status_text(base_text: str, item: dict, interest_rated: bool, goal_rated: bool, remind_set: bool) -> str:
    lines = [base_text]
    if interest_rated:
        label = _INTEREST_LABEL.get(item.get("interest"), "\U0001f44d Medium")
        lines.append(f"Interest: {label} ✓")
    if goal_rated:
        label = _GOAL_LABEL.get(item.get("goal_alignment"), "❌ Nope")
        lines.append(f"Goal: {label} ✓")
    if remind_set:
        lines.append("⏰ Remind tonight ✓")
    return "\n".join(lines)


def _truncate(text: str, limit: int = 3500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "... (truncated)"


def _fire_embedding(result: dict):
    if not isinstance(result, dict) or result.get("status") == "needs_screenshot":
        return
    text = build_embedding_text(result.get("title"), result.get("summary"), result.get("tags"))
    embedding = generate_embedding(text)
    if embedding:
        update_item_embedding(result["item_id"], embedding)


async def _send_save_response(message, result):
    if isinstance(result, list):
        saved = []
        skipped = []
        for r in result:
            if r.get("status") == "needs_screenshot":
                skipped.append(r["message"])
            elif r.get("extraction_failed"):
                saved.append(r)
            else:
                saved.append(r)

        if saved:
            lines = [f"Saved {len(saved)} item{'s' if len(saved) > 1 else ''}:"]
            for i, r in enumerate(saved, 1):
                if r.get("extraction_failed"):
                    lines.append(f"{i}. Uncategorized: {r['title']} ⚠️")
                else:
                    lines.append(f"{i}. {r['category_name']}: {r['title']}")
            await message.reply_text(_truncate("\n".join(lines)))

            for r in saved:
                include_ctx = bool(r.get("extraction_failed"))
                await message.reply_text(
                    f"Rate: {r['title']}",
                    reply_markup=_rating_keyboard(r["item_id"], include_context=include_ctx),
                )

        for msg in skipped:
            await message.reply_text(msg)

        for r in saved:
            asyncio.get_event_loop().run_in_executor(None, _fire_embedding, r)
        return

    if isinstance(result, dict) and result.get("extraction_failed"):
        failure_msg = result.get("failure_reason", "")
        text = f"Saved link under Uncategorized: {result['title']}"
        if failure_msg:
            text += f"\n⚠️ {failure_msg}"
        await message.reply_text(text)
        await message.reply_text(
            "Interest level? / Goal alignment?",
            reply_markup=_rating_keyboard(result["item_id"], include_context=True),
        )
        asyncio.get_event_loop().run_in_executor(None, _fire_embedding, result)
        return

    tags = ", ".join(result["tags"])
    await message.reply_text(
        f"Saved under {result['category_name']}: {result['title']}\n\nTags: {tags}"
    )
    await message.reply_text(
        "Interest level? / Goal alignment?",
        reply_markup=_rating_keyboard(result["item_id"]),
    )
    asyncio.get_event_loop().run_in_executor(None, _fire_embedding, result)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return
    user = get_user_by_telegram_id(update.effective_user.id)
    upsert_user(
        telegram_user_id=update.effective_user.id,
        chat_id=update.effective_chat.id,
        display_name=update.effective_user.first_name,
    )
    if user:
        if not user.get("is_active"):
            set_user_active(update.effective_user.id, True)
        await update.message.reply_text(
            "Welcome back to Cue! \U0001f4e6 Nudges resumed.\n\n"
            "<b>Commands:</b>\n"
            "/review — see your pending items\n"
            "/categories — view your categories\n"
            "/search [keyword] — find saved items\n"
            "/stats — your numbers\n"
            "/email — set your email for weekly digest (unlocks themed clusters + AI deep dives)\n"
            "/nudgetime HH:MM — set morning nudge time (on the hour or half hour)\n"
            "/remindertime HH:MM — set nightly reminder time (on the hour or half hour)\n"
            "/stop — pause all nudges\n"
            "/start — resume nudges",
            parse_mode="HTML",
        )
    else:
        await update.message.reply_text(
            "Welcome to Cue! \U0001f4e6\n\n"
            "Your second brain. Dump everything, forget nothing.\n\n"
            "Send me anything — screenshots, links, voice notes, ideas — "
            "and I'll organize it, remember it, and remind you when it matters.\n\n"
            "<b>What I can handle:</b>\n"
            "\U0001f4f8 Screenshots — I read the text, ignore the clutter\n"
            "\U0001f517 Links — articles, YouTube, Substack\n"
            "\U0001f3a4 Voice notes — transcribed and categorized\n"
            "\U0001f4ac Text — ideas, thoughts, anything\n\n"
            "After each save, I'll ask you to rate:\n"
            "→ How interested are you?\n"
            "→ Does it align with your goals?\n\n"
            "<b>Commands:</b>\n"
            "/review — see your pending items\n"
            "/categories — view your categories\n"
            "/search [keyword] — find saved items\n"
            "/stats — your numbers\n"
            "/email — set your email for weekly digest (unlocks themed clusters + AI deep dives)\n"
            "/nudgetime HH:MM — set morning nudge time (on the hour or half hour)\n"
            "/remindertime HH:MM — set nightly reminder time (on the hour or half hour)\n"
            "/stop — pause all nudges\n"
            "/start — resume nudges\n\n"
            "Just start sharing. I'll handle the rest.",
            parse_mode="HTML",
        )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return
    set_user_active(update.effective_user.id, False)
    await update.message.reply_text(
        "All nudges and reminders paused. Your saved items are safe. "
        "Send /start anytime to resume."
    )


AWAITING_REMINDER_TIME, AWAITING_NUDGE_TIME, AWAITING_EMAIL, AWAITING_SEARCH_KEYWORD = range(4)

_TIME_INVALID_MSG = "That doesn't look right. Send a time like 08:00 or 08:30"


def _parse_time(time_str: str) -> str | None:
    try:
        parts = time_str.strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        if not (0 <= h <= 23 and m in (0, 30)):
            return None
        return f"{h:02d}:{m:02d}"
    except (ValueError, IndexError):
        return None


async def remindertime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return ConversationHandler.END
    if context.args:
        formatted = _parse_time(context.args[0])
        if not formatted:
            await update.message.reply_text(
                "Invalid time. Use HH:MM on the hour or half hour (e.g. 22:00, 22:30)"
            )
            return ConversationHandler.END
        update_reminder_time(update.effective_user.id, formatted)
        await update.message.reply_text(f"Nightly reminder set to {formatted} ✓")
        return ConversationHandler.END

    await update.message.reply_text(
        "What time should I send your nightly reminder?\n\n"
        "Send the time in HH:MM format (on the hour or half hour).\n"
        "Examples: 21:00, 21:30, 22:00"
    )
    return AWAITING_REMINDER_TIME


async def _receive_reminder_time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    formatted = _parse_time(update.message.text)
    if not formatted:
        await update.message.reply_text(_TIME_INVALID_MSG)
        return AWAITING_REMINDER_TIME
    update_reminder_time(update.effective_user.id, formatted)
    await update.message.reply_text(f"Nightly reminder set to {formatted} ✓")
    return ConversationHandler.END


async def nudgetime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return ConversationHandler.END
    if context.args:
        formatted = _parse_time(context.args[0])
        if not formatted:
            await update.message.reply_text(
                "Invalid time. Use HH:MM on the hour or half hour (e.g. 08:00, 08:30)"
            )
            return ConversationHandler.END
        update_nudge_time(update.effective_user.id, formatted)
        await update.message.reply_text(f"Morning nudge set to {formatted} ✓")
        return ConversationHandler.END

    await update.message.reply_text(
        "What time should I send your morning nudge?\n\n"
        "Send the time in HH:MM format (on the hour or half hour).\n"
        "Examples: 08:00, 08:30, 09:00"
    )
    return AWAITING_NUDGE_TIME


async def _receive_nudge_time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    formatted = _parse_time(update.message.text)
    if not formatted:
        await update.message.reply_text(_TIME_INVALID_MSG)
        return AWAITING_NUDGE_TIME
    update_nudge_time(update.effective_user.id, formatted)
    await update.message.reply_text(f"Morning nudge set to {formatted} ✓")
    return ConversationHandler.END


def _is_valid_email(text: str) -> bool:
    return "@" in text and "." in text.split("@")[-1]


async def email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return ConversationHandler.END
    if context.args:
        addr = context.args[0].strip()
        if not _is_valid_email(addr):
            await update.message.reply_text("That doesn't look like an email. Try again.")
            return ConversationHandler.END
        update_user_email(update.effective_user.id, addr)
        await update.message.reply_text(f"Weekly digest will be sent to {addr} ✓")
        return ConversationHandler.END

    await update.message.reply_text(
        "What email should I send your weekly digest to?\n"
        "Send your email address:"
    )
    return AWAITING_EMAIL


async def _receive_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    addr = update.message.text.strip()
    if not _is_valid_email(addr):
        await update.message.reply_text("That doesn't look like an email. Try again.")
        return AWAITING_EMAIL
    update_user_email(update.effective_user.id, addr)
    await update.message.reply_text(f"Weekly digest will be sent to {addr} ✓")
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


async def categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return
    user_id = _get_user_id(update)
    cats = get_categories_with_counts(user_id)

    total = len(cats)
    lines = [f"📂 Your categories — {total} total", ""]
    for cat in cats:
        name = escape_html(cat["name"])
        count = cat["count"]
        lines.append(f"{name} — {count} item{'s' if count != 1 else ''}")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def _run_search(update: Update, keyword: str):
    user_id = _get_user_id(update)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    items = search_items(user_id, keyword)

    if not items:
        await update.message.reply_text(
            f"No results for '{escape_html(keyword)}'. Try a different search term.",
            parse_mode="HTML",
        )
        return

    formatted = [_format_review_item(item, now) for item in items]
    total = len(formatted)
    header = f"🔍 Found {total} result{'s' if total != 1 else ''} for '{escape_html(keyword)}'"

    cb_keyword = keyword[:20]

    user_record = get_user_by_telegram_id(update.effective_user.id)
    has_email = bool(user_record.get("email")) if user_record else False
    set_session(update.effective_chat.id, formatted, header, has_email=has_email, review_offset=0, search_keyword=cb_keyword)
    session = get_session(update.effective_chat.id)
    text, keyboard = _build_review_page(session)

    await update.message.reply_text(
        _truncate(text),
        reply_markup=keyboard,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return ConversationHandler.END
    if not context.args:
        await update.message.reply_text("What are you looking for? Type a keyword:")
        return AWAITING_SEARCH_KEYWORD

    keyword = " ".join(context.args)
    await _run_search(update, keyword)
    return ConversationHandler.END


async def _receive_search_keyword(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyword = update.message.text.strip()
    if not keyword:
        await update.message.reply_text("Please type a keyword to search for:")
        return AWAITING_SEARCH_KEYWORD
    await _run_search(update, keyword)
    return ConversationHandler.END


async def handle_search_page(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.callback_query.answer("Not authorized.")
        return
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat_id
    session = get_session(chat_id)
    if not session:
        await query.edit_message_text(_EXPIRED_NUDGE_TEXT)
        return

    new_offset = int(query.data.rsplit("_", 1)[1])
    session["review_offset"] = max(0, new_offset)

    text, keyboard = _build_review_page(session)
    await query.edit_message_text(
        text=_truncate(text),
        reply_markup=keyboard,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return
    user_id = _get_user_id(update)
    s = get_user_stats(user_id)

    lines = [
        "📊 Your Cue stats",
        "",
        f"Total items: {s['total']}",
        f"├ Active: {s['active']}",
        f"├ Acted on: {s['acted_on']}",
        f"└ Archived: {s['archived']}",
        "",
        "This week:",
        f"├ Saved: {s['week_saved']}",
        f"├ Acted on: {s['week_acted']}",
        f"└ Archived: {s['week_archived']}",
    ]

    if s["top_categories"]:
        lines.append("")
        lines.append("Top categories:")
        for idx, cat in enumerate(s["top_categories"], 1):
            lines.append(f"{idx}. {escape_html(cat['name'])} — {cat['count']} items")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


def _format_review_item(item: dict, now) -> dict:
    from dateutil.parser import isoparse
    created_at = isoparse(item["created_at"]) if isinstance(item.get("created_at"), str) else item.get("created_at", now)
    age_days = round((now - created_at).total_seconds() / 86400, 1)
    times_surfaced = item.get("times_surfaced", 0)
    interest = item.get("interest", 2)
    goal = item.get("goal_alignment", 1)
    return {
        "id": item["id"],
        "title": item.get("title") or "Untitled",
        "summary": item.get("summary"),
        "category_name": item.get("category_name"),
        "interest": interest,
        "goal_alignment": goal,
        "age_days": age_days,
        "times_surfaced": times_surfaced,
        "is_escalation": times_surfaced >= 3,
        "emoji": _get_emoji(item),
        "url": _extract_url(item.get("raw_content")) if item.get("content_type") == "url" else None,
        "content_type": item.get("content_type"),
        "raw_content": item.get("raw_content"),
        "extracted_text": item.get("extracted_text"),
        "image_path": item.get("image_path"),
        "go_deep": item.get("go_deep", False),
        "_weight": _PRIORITY_MATRIX.get((interest, goal), 1),
    }


def _build_review_page(session: dict) -> tuple[str, InlineKeyboardMarkup | None]:
    order = session["order"]
    items = session["items"]
    acted = session["acted"]
    offset = session.get("review_offset", 0) or 0
    total = len(order)

    page_ids = order[offset:offset + REVIEW_PAGE_SIZE]
    if not page_ids:
        return "Nothing pending! You're all caught up.", None

    lines = [session["header"], ""]
    number_buttons = []
    for idx, item_id in enumerate(page_ids, start=1):
        item = items[item_id]
        if item_id in acted:
            from notifications.daily_nudge import _ACTED_MARKER
            marker, word = _ACTED_MARKER.get(acted[item_id], ("✅", "done"))
            lines.append(f"{idx}. {marker} {escape_html(item['title'])} — {word}")
        else:
            lines.append(_list_line(idx, item))
            number_buttons.append(
                InlineKeyboardButton(f"  {idx}  ", callback_data=f"nudgelist_{item_id}")
            )

    page_num = (offset // REVIEW_PAGE_SIZE) + 1
    total_pages = (total + REVIEW_PAGE_SIZE - 1) // REVIEW_PAGE_SIZE
    lines.append(f"\nPage {page_num}/{total_pages} · {total} items total")

    text = "\n".join(lines)

    rows = []
    if number_buttons:
        row1 = number_buttons[:3]
        row2 = number_buttons[3:]
        rows.append(row1)
        if row2:
            rows.append(row2)

    nav_row = []
    keyword = session.get("search_keyword")
    if keyword:
        prev_cb = f"search_prev_{keyword}_{offset - REVIEW_PAGE_SIZE}"
        more_cb = f"search_more_{keyword}_{offset + REVIEW_PAGE_SIZE}"
    else:
        prev_cb = f"review_prev_{offset - REVIEW_PAGE_SIZE}"
        more_cb = f"review_more_{offset + REVIEW_PAGE_SIZE}"
    if offset > 0:
        nav_row.append(InlineKeyboardButton("← Previous", callback_data=prev_cb))
    if offset + REVIEW_PAGE_SIZE < total:
        nav_row.append(InlineKeyboardButton("Show more →", callback_data=more_cb))
    if nav_row:
        rows.append(nav_row)

    keyboard = InlineKeyboardMarkup(rows) if rows else None
    return text, keyboard


async def review(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return
    user_id = _get_user_id(update)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    days = None
    if context.args:
        try:
            days = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Usage: /review or /review 7 (number of days)")
            return

    raw_items = get_pending_items(user_id, days=days)
    if not raw_items:
        await update.message.reply_text("Nothing pending! You're all caught up. 🎉")
        return

    formatted = [_format_review_item(item, now) for item in raw_items]
    formatted.sort(key=lambda x: (-int(x["times_surfaced"] == 0), -x["_weight"]))

    if days:
        header = f"📋 Items from the last {days} days — {len(formatted)} pending"
    else:
        header = f"📋 Your review queue — {len(formatted)} item{'s' if len(formatted) != 1 else ''} pending"

    user_record = get_user_by_telegram_id(update.effective_user.id)
    has_email = bool(user_record.get("email")) if user_record else False
    set_session(update.effective_chat.id, formatted, header, has_email=has_email, review_offset=0)
    session = get_session(update.effective_chat.id)
    text, keyboard = _build_review_page(session)

    await update.message.reply_text(
        _truncate(text),
        reply_markup=keyboard,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


async def handle_review_page(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.callback_query.answer("Not authorized.")
        return
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat_id
    session = get_session(chat_id)
    if not session:
        await query.edit_message_text(_EXPIRED_NUDGE_TEXT)
        return

    new_offset = int(query.data.split("_")[-1])
    session["review_offset"] = max(0, new_offset)

    text, keyboard = _build_review_page(session)
    await query.edit_message_text(
        text=_truncate(text),
        reply_markup=keyboard,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


async def handle_context_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.callback_query.answer("Not authorized.")
        return
    query = update.callback_query
    await query.answer()

    item_id = query.data.replace("ctx_", "")

    # Cap pending context dict at 100 entries
    if len(_pending_context) >= 100:
        oldest_keys = sorted(_pending_context.keys())[:50]
        for k in oldest_keys:
            del _pending_context[k]

    sent = await query.message.reply_text(
        f"📎 Send a screenshot or short note about this link, and I'll update the classification.",
        reply_markup=ForceReply(selective=True),
    )
    _pending_context[sent.message_id] = item_id


async def _handle_context_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, item_id: str):
    """Process a user's reply that provides context for a failed-extraction URL save."""
    try:
        if update.message.photo:
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.close()
            await file.download_to_drive(tmp.name)
            extracted_data = vision.extract(tmp.name, update.message.caption or "")
        elif update.message.voice:
            voice = update.message.voice
            file = await context.bot.get_file(voice.file_id)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
            tmp.close()
            await file.download_to_drive(tmp.name)
            extracted_data = whisper.extract(tmp.name, "")
        else:
            extracted_data = text.extract(update.message.text)

        extracted_text = extracted_data.get("extracted_text")
        if not extracted_text:
            await update.message.reply_text(
                "Couldn't read that either — try a screenshot or a short text description."
            )
            return

        classification = classify(extracted_text)

        update_item_from_context(
            item_id=item_id,
            category_id=classification["category_id"],
            category_name=classification["category_name"],
            title=classification["title"],
            summary=classification["summary"],
            tags=classification["tags"],
            extracted_text=extracted_text,
            image_path=extracted_data.get("image_path"),
        )

        tags = ", ".join(classification["tags"])
        await update.message.reply_text(
            f"Updated! Saved under {classification['category_name']}: {classification['title']}\n\nTags: {tags}"
        )

        result_for_embedding = {
            "item_id": item_id,
            "title": classification["title"],
            "summary": classification["summary"],
            "tags": classification["tags"],
        }
        asyncio.get_event_loop().run_in_executor(None, _fire_embedding, result_for_embedding)

    except Exception as e:
        logger.error(f"Context update failed for item {item_id}: {e}")
        await update.message.reply_text(f"Something went wrong updating this item: {e}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return

    if update.message.reply_to_message:
        reply_msg_id = update.message.reply_to_message.message_id
        if reply_msg_id in _pending_context:
            item_id = _pending_context.pop(reply_msg_id)
            await _handle_context_reply(update, context, item_id)
            return

    user_id = _get_user_id(update)
    try:
        result = process_message(update.message.text, content_type="text", user_id=user_id)
        if isinstance(result, dict) and result.get("status") == "needs_screenshot":
            await update.message.reply_text(result["message"])
            return
        await _send_save_response(update.message, result)

    except Exception as e:
        await update.message.reply_text(f"Something went wrong: {e}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return

    if update.message.reply_to_message:
        reply_msg_id = update.message.reply_to_message.message_id
        if reply_msg_id in _pending_context:
            item_id = _pending_context.pop(reply_msg_id)
            await _handle_context_reply(update, context, item_id)
            return

    user_id = _get_user_id(update)
    try:
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
        tmp.close()
        await file.download_to_drive(tmp.name)

        result = process_message(
            raw_content="",
            content_type="voice",
            file_path=tmp.name,
            user_id=user_id,
        )
        await _send_save_response(update.message, result)

    except Exception as e:
        await update.message.reply_text(f"Something went wrong: {e}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, this is a private bot.")
        return

    if update.message.reply_to_message:
        reply_msg_id = update.message.reply_to_message.message_id
        if reply_msg_id in _pending_context:
            item_id = _pending_context.pop(reply_msg_id)
            await _handle_context_reply(update, context, item_id)
            return

    user_id = _get_user_id(update)
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.close()
        await file.download_to_drive(tmp.name)

        caption = update.message.caption or ""
        result = process_message(
            raw_content=caption,
            content_type="image",
            file_path=tmp.name,
            user_id=user_id,
        )
        await _send_save_response(update.message, result)

    except Exception as e:
        await update.message.reply_text(f"Something went wrong: {e}")


async def handle_rating(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.callback_query.answer("Not authorized.")
        return
    query = update.callback_query
    await query.answer()

    parts = query.data.split("_", 2)
    rating_type = parts[0]
    value = int(parts[1])
    item_id = parts[2]

    if rating_type == "interest":
        update_item_rating(item_id, "interest", value)
    elif rating_type == "goal":
        update_item_rating(item_id, "goal_alignment", value)

    item = get_item(item_id)
    if not item:
        await query.edit_message_text("Could not find item.")
        return

    interest = item.get("interest")
    goal = item.get("goal_alignment")
    interest_rated = interest is not None and interest != 2
    goal_rated = goal is not None and goal != 1
    remind_set = bool(item.get("remind_tonight"))

    if rating_type == "interest":
        interest_rated = True
    if rating_type == "goal":
        goal_rated = True

    base = query.message.text.split("\n")[0] if query.message.text else "Rate this item"
    text = _build_status_text(base, item, interest_rated, goal_rated, remind_set)
    keyboard = _remaining_keyboard(item_id, interest_rated, goal_rated, remind_set)

    await query.edit_message_text(text, reply_markup=keyboard)


async def handle_remind_tonight(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.callback_query.answer("Not authorized.")
        return
    query = update.callback_query
    await query.answer()

    item_id = query.data.replace("remind_tonight_", "")
    set_remind_tonight(item_id)

    item = get_item(item_id)
    if not item:
        await query.edit_message_text("⏰ Got it — I'll include this in tonight's reminder.")
        return

    interest = item.get("interest")
    goal = item.get("goal_alignment")
    interest_rated = interest is not None and interest != 2
    goal_rated = goal is not None and goal != 1

    base = query.message.text.split("\n")[0] if query.message.text else "Rate this item"
    text = _build_status_text(base, item, interest_rated, goal_rated, remind_set=True)
    keyboard = _remaining_keyboard(item_id, interest_rated, goal_rated, remind_set=True)

    await query.edit_message_text(text, reply_markup=keyboard)


_EXPIRED_NUDGE_TEXT = "This nudge has expired. You'll get a fresh one next time!"

_NUDGE_ACTIONS = {
    "done": (done_item, "✅ {title} — marked as done", "done"),
    "archive": (archive_item, "📦 {title} — archived", "archived"),
    "remind": (lambda item_id: remind_later(item_id, days=3), "⏰ {title} — I'll remind you in 3 days", "remind"),
    "keep": (keep_item, "🔖 {title} — keeping it, back in 7 days", "kept"),
    "drop": (archive_item, "📦 {title} — dropped", "archived"),
}


async def _render_nudge_list(query, context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    session = get_session(chat_id)
    if not session:
        text, keyboard = _EXPIRED_NUDGE_TEXT, None
    elif session.get("review_offset") is not None:
        text, keyboard = _build_review_page(session)
    else:
        text, keyboard = build_list_view(session)

    if query.message.photo:
        try:
            await query.message.delete()
        except Exception:
            pass
        await context.bot.send_message(
            chat_id=chat_id, text=text, reply_markup=keyboard,
            parse_mode="HTML", disable_web_page_preview=True,
        )
    else:
        await query.edit_message_text(
            text=text, reply_markup=keyboard,
            parse_mode="HTML", disable_web_page_preview=True,
        )


async def handle_nudge_list_tap(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.callback_query.answer("Not authorized.")
        return
    query = update.callback_query
    await query.answer()

    item_id = query.data.split("_", 1)[1]
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    if not session or item_id not in session["items"]:
        await query.edit_message_text(_EXPIRED_NUDGE_TEXT)
        return

    item = session["items"][item_id]
    has_email = session.get("has_email", True)
    text, keyboard = build_detail_view(item, has_email=has_email)

    if item.get("content_type") == "image":
        image_bytes = get_image_bytes(item["image_path"]) if item.get("image_path") else None
        try:
            await query.message.delete()
        except Exception:
            pass
        if image_bytes:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=io.BytesIO(image_bytes),
                caption=text,
                reply_markup=keyboard,
                parse_mode="HTML",
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"{text}\n\n<i>Original screenshot no longer available.</i>",
                reply_markup=keyboard,
                parse_mode="HTML", disable_web_page_preview=True,
            )
    else:
        await query.edit_message_text(
            text=text, reply_markup=keyboard,
            parse_mode="HTML", disable_web_page_preview=True,
        )


async def handle_nudge_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.callback_query.answer("Not authorized.")
        return
    query = update.callback_query
    await query.answer()

    parts = query.data.split("_", 2)
    action = parts[1]
    chat_id = query.message.chat_id

    if action == "back":
        await _render_nudge_list(query, context, chat_id)
        return

    item_id = parts[2]
    meta = _NUDGE_ACTIONS.get(action)
    if meta is None:
        return
    fn, confirm_template, acted_status = meta

    fn(item_id)

    session = get_session(chat_id)
    if session and item_id in session["items"]:
        session["acted"][item_id] = acted_status

    try:
        await _render_nudge_list(query, context, chat_id)
    except Exception as e:
        if "not modified" not in str(e).lower():
            logger.error(f"Edit error: {e}")


async def handle_go_deep(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        await update.callback_query.answer("Not authorized.")
        return
    query = update.callback_query
    await query.answer()

    item_id = query.data.replace("nudge_godeep_", "")
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    if not session or item_id not in session["items"]:
        await query.edit_message_text(_EXPIRED_NUDGE_TEXT)
        return

    has_email = session.get("has_email", True)
    if not has_email:
        await query.answer("Deep dives are sent in your weekly digest. Set your email first with /email", show_alert=True)
        return

    set_go_deep(item_id)

    item = session["items"][item_id]
    item["go_deep"] = True

    text, keyboard = build_detail_view(item, has_email=has_email)

    await query.edit_message_text(
        text=text, reply_markup=keyboard,
        parse_mode="HTML", disable_web_page_preview=True,
    )


def run_bot():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    reminder_conv = ConversationHandler(
        entry_points=[CommandHandler("remindertime", remindertime)],
        states={
            AWAITING_REMINDER_TIME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, _receive_reminder_time),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    nudge_conv = ConversationHandler(
        entry_points=[CommandHandler("nudgetime", nudgetime)],
        states={
            AWAITING_NUDGE_TIME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, _receive_nudge_time),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    email_conv = ConversationHandler(
        entry_points=[CommandHandler("email", email)],
        states={
            AWAITING_EMAIL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, _receive_email),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    search_conv = ConversationHandler(
        entry_points=[CommandHandler("search", search)],
        states={
            AWAITING_SEARCH_KEYWORD: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, _receive_search_keyword),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("review", review))
    app.add_handler(CommandHandler("categories", categories))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(reminder_conv)
    app.add_handler(nudge_conv)
    app.add_handler(email_conv)
    app.add_handler(search_conv)
    app.add_handler(CallbackQueryHandler(handle_context_button, pattern="^ctx_"))
    app.add_handler(CallbackQueryHandler(handle_nudge_list_tap, pattern="^nudgelist_"))
    app.add_handler(CallbackQueryHandler(handle_nudge_action, pattern="^nudge_(done|archive|remind|keep|drop|back)_"))
    app.add_handler(CallbackQueryHandler(handle_go_deep, pattern="^nudge_godeep_"))
    app.add_handler(CallbackQueryHandler(handle_rating, pattern="^(interest_|goal_)"))
    app.add_handler(CallbackQueryHandler(handle_remind_tonight, pattern="^remind_tonight_"))
    app.add_handler(CallbackQueryHandler(handle_review_page, pattern="^review_(more|prev)_"))
    app.add_handler(CallbackQueryHandler(handle_search_page, pattern="^search_(more|prev)_"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Bot is running...")
    app.run_polling()

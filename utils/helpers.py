import json
from datetime import datetime, timezone
from typing import Optional

def parse_twitter_date(s: str) -> Optional[datetime]:
    """Parse Twitter's 'Thu Apr 06 15:28:43 +0000 2017' format."""
    if not s:
        return None
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S +0000 %Y").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return None

def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def account_age_days(user_created_at: str, ref_time: datetime) -> float:
    """Return account age in days relative to a reference tweet time."""
    created = parse_twitter_date(user_created_at)
    if created is None or ref_time is None:
        return 0.0
    delta = ref_time - created
    return max(0.0, delta.total_seconds() / 86400)

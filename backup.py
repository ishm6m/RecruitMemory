"""
backup.py — copy the SQLite database up to Alibaba Cloud OSS (object storage).

Plain language: OSS is Alibaba's version of a hard drive in the cloud. This
script grabs your database file and uploads a timestamped copy of it into a
"bucket" (a folder) in OSS, so your candidates and memories are safe even if
your server dies. It can also download the latest copy back.

Run it:
    python backup.py            # upload a fresh backup
    python backup.py restore    # download the newest backup over your local DB

It reads the same .env you already use (OSS_* fields).
"""

import os
import sys
import time

import oss2
from dotenv import load_dotenv

load_dotenv()

# Same DB location logic as db.py, so it works locally and in Docker.
DB_PATH = os.getenv("DB_PATH", "recruitmemory.db")

# OSS credentials + location, all from .env.
KEY_ID = os.getenv("OSS_ACCESS_KEY_ID")
KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET")
BUCKET_NAME = os.getenv("OSS_BUCKET")
ENDPOINT = os.getenv("OSS_ENDPOINT")

# Where inside the bucket the backups go.
PREFIX = "recruitmemory-backups/"


def _bucket():
    """Connect to the OSS bucket, failing loudly if .env isn't filled in."""
    missing = [
        n for n, v in [
            ("OSS_ACCESS_KEY_ID", KEY_ID),
            ("OSS_ACCESS_KEY_SECRET", KEY_SECRET),
            ("OSS_BUCKET", BUCKET_NAME),
            ("OSS_ENDPOINT", ENDPOINT),
        ] if not v
    ]
    if missing:
        sys.exit("Missing in .env: " + ", ".join(missing))
    auth = oss2.Auth(KEY_ID, KEY_SECRET)
    return oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)


def backup():
    if not os.path.exists(DB_PATH):
        sys.exit(f"No database found at {DB_PATH} — nothing to back up yet.")
    bucket = _bucket()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    key = f"{PREFIX}recruitmemory-{stamp}.db"
    bucket.put_object_from_file(key, DB_PATH)
    print(f"Backed up {DB_PATH} -> oss://{BUCKET_NAME}/{key}")


def restore():
    bucket = _bucket()
    # List every backup, newest key sorts last (timestamps are zero-padded).
    keys = [o.key for o in oss2.ObjectIterator(bucket, prefix=PREFIX)]
    if not keys:
        sys.exit("No backups found in the bucket yet.")
    latest = sorted(keys)[-1]
    bucket.get_object_to_file(latest, DB_PATH)
    print(f"Restored oss://{BUCKET_NAME}/{latest} -> {DB_PATH}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore()
    else:
        backup()

"""
download_emails_imap.py
Télécharge des emails via IMAP (z.imt.fr) et les sauvegarde dans TP4/data/emails/

- 1 email = 1 fichier Markdown
- Cache SQLite pour éviter les doublons
"""

import os
import re
import sqlite3
import imaplib
import email
from email import policy
from email.header import decode_header
from datetime import datetime, timedelta
from getpass import getpass

HOST = "z.imt.fr"
PORT = 993

DATA_DIR = os.path.join("TP4", "data")
EMAIL_DIR = os.path.join(DATA_DIR, "emails")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
DB_PATH = os.path.join(CACHE_DIR, "emails_cache.sqlite")


def ensure_dirs():
    os.makedirs(EMAIL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS downloaded_emails (
            account TEXT NOT NULL,
            message_id TEXT NOT NULL,
            folder TEXT,
            PRIMARY KEY (account, message_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sync_status (
            account TEXT PRIMARY KEY,
            last_synced TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def was_downloaded(conn, account, message_id):
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM downloaded_emails WHERE account=? AND message_id=?",
        (account, message_id),
    )
    return cur.fetchone() is not None


def mark_downloaded(conn, account, message_id, folder):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO downloaded_emails(account, message_id, folder) VALUES (?,?,?)",
        (account, message_id, folder),
    )
    conn.commit()


def safe_filename(s):
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s[:80] if s else "no_subject"


def decode_mime_words(s):
    if not s:
        return ""
    parts = decode_header(s)
    out = []
    for part, enc in parts:
        if isinstance(part, bytes):
            out.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(part)
    return "".join(out)


def extract_text(msg):
    if msg.is_multipart():
        for p in msg.walk():
            if p.get_content_type() == "text/plain":
                try:
                    return p.get_content()
                except Exception:
                    pass
        return ""
    return msg.get_content()


def format_since_date(dt):
    return dt.strftime("%d-%b-%Y")


def main():
    ensure_dirs()
    conn = init_db()

    account = input("Adresse email : ").strip()
    password = getpass("Mot de passe IMAP : ")

    since_dt = datetime.now() - timedelta(days=30)
    since_imap = format_since_date(since_dt)

    imap = imaplib.IMAP4_SSL(HOST, PORT)
    imap.login(account, password)
    imap.select("INBOX")

    status, data = imap.search(None, f'(SINCE "{since_imap}")')
    msg_ids = data[0].split()

    for mid in msg_ids:
        status, msg_data = imap.fetch(mid, "(RFC822)")
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw, policy=policy.default)

        message_id = msg.get("Message-ID", mid.decode())
        if was_downloaded(conn, account, message_id):
            continue

        subject = decode_mime_words(msg.get("Subject", ""))
        sender = decode_mime_words(msg.get("From", ""))
        date = decode_mime_words(msg.get("Date", ""))
        body = extract_text(msg)

        fname = f"{safe_filename(subject)}.md"
        path = os.path.join(EMAIL_DIR, fname)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {subject}\n\n")
            f.write(f"**From:** {sender}\n\n")
            f.write(f"**Date:** {date}\n\n")
            f.write(body)

        mark_downloaded(conn, account, message_id, "INBOX")

    imap.logout()


if __name__ == "__main__":
    main()

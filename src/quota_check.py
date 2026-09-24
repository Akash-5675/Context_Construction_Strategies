"""Ping each Cloudflare account with a 1-token request to see if quota is available."""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

ACCOUNTS = [
    ("account 1 (main, .env)", os.getenv("CLOUDFLARE_API_TOKEN"), os.getenv("CLOUDFLARE_ACCOUNT_ID")),
    ("account 2 (college)",
     os.getenv("CLOUDFLARE_API_TOKEN_2"), os.getenv("CLOUDFLARE_ACCOUNT_ID_2")),
]

now = datetime.now()
utc = datetime.now(timezone.utc)
print(f"local {now:%H:%M} | UTC {utc:%H:%M} (quota resets 00:00 UTC)\n")

for name, token, account in ACCOUNTS:
    url = (f"https://api.cloudflare.com/client/v4/accounts/{account}"
           f"/ai/run/@cf/meta/llama-3.1-8b-instruct")
    payload = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 1}).encode()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        urllib.request.urlopen(req, timeout=30).read()
        print(f"  AVAILABLE  {name}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        reason = "quota exhausted" if "neurons" in body else f"HTTP {e.code}: {body[:120]}"
        print(f"  BLOCKED    {name} -- {reason}")
    except Exception as e:
        print(f"  ERROR      {name} -- {e}")

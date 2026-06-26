#!/usr/bin/env bash
# The ONLY privileged action spark-agent performs. Fixed, audited, never receives
# LLM- or chat-derived input — only a sanitized username + a locally generated
# password from agent.py. Run via sudo (see README for the NOPASSWD sudoers line):
#   sudo /opt/spark-agent/provision.sh <username> <password>
set -euo pipefail

USERNAME="${1:-}"
PASSWORD="${2:-}"

# strict validation — defence in depth even though agent.py also sanitizes
if ! [[ "$USERNAME" =~ ^[a-z][a-z0-9_-]{2,30}$ ]]; then
  echo "invalid username" >&2; exit 2
fi
if [[ -z "$PASSWORD" || ${#PASSWORD} -lt 10 ]]; then
  echo "weak password" >&2; exit 2
fi
# never touch an existing / system account
if id "$USERNAME" &>/dev/null; then
  echo "exists" >&2; exit 3
fi

useradd -m -s /bin/bash -c "ai-eng-studio capstone" "$USERNAME"
printf '%s:%s\n' "$USERNAME" "$PASSWORD" | chpasswd
# Force a password change on first login (-d 0), but ALSO clear the minimum
# password age (-m 0). Otherwise, if the system's PASS_MIN_DAYS > 0, the account
# is simultaneously "expired, must change" and "changed too recently, may not
# change yet" — a catch-22 that makes the first-login passwd fail with
# "Authentication token manipulation error" before it even prompts for the new
# password. -M 99999 keeps the new password from expiring later.
chage -m 0 -M 99999 -d 0 "$USERNAME"

# Optional confinement: if a 'capstone' group exists (for disk quota / cgroup limits),
# add the user to it. The account never gets sudo.
if getent group capstone >/dev/null; then
  usermod -aG capstone "$USERNAME"
fi

echo "ok"

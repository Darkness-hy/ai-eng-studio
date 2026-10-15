# Fix DeepSeek Tutor Deployment on spark-32e2

## Goal

Restore the online AI tutor for `ai-eng-studio`.

The DeepSeek tutor service must run on `spark-32e2`. The public endpoint should be:

```text
https://tutor.hongyuding.site/chat
```

The frontend expects Server-Sent Events in this exact shape:

```text
data: {"type":"delta","text":"..."}

data: {"type":"done"}
```

## Current Situation

Known working local test on `spark-32e2`:

```bash
curl -N -X POST http://127.0.0.1:8787/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"用一句话介绍你自己","history":[],"lang":"zh","stream":true}'
```

Expected output:

```text
data: {"type": "delta", "text": "..."}
data: {"type":"done"}
```

The public test currently reaches `https://tutor.hongyuding.site/chat`, but returns:

```text
data: {"type": "error", "message": "助教暂时不可用,请稍后再试"}
```

Important diagnosis:

- `tutor.hongyuding.site` is served by Caddy on another machine, observed as `instance-cb6301wm`.
- Caddy logs show `reverse_proxy` upstream as `localhost:8787`.
- `localhost:8787` on the Caddy machine is not the same as `localhost:8787` on `spark-32e2`.
- Therefore, the public endpoint is probably hitting a broken/local service on the Caddy machine, not the working tutor service on `spark-32e2`.

## Do Not

- Do not move the tutor service off `spark-32e2`.
- Do not expose the DeepSeek API key to the frontend.
- Do not point the frontend at raw `http://spark-32e2:8787`; browsers need HTTPS.
- Do not assume `localhost` in Caddy means `spark-32e2`.

## Step 1: Verify Service on spark-32e2

Run on `spark-32e2`:

```bash
hostname
systemctl --user status claude-deepseek-proxy --no-pager
curl -N -X POST http://127.0.0.1:8787/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"用一句话介绍你自己","history":[],"lang":"zh","stream":true}'
```

If local curl fails, fix the user service first:

```bash
journalctl --user -u claude-deepseek-proxy -n 100 --no-pager
cat /home/dhy/spark-agent/.env.deepseek
ls -l /home/dhy/spark-agent/claude-deepseek.sh
```

The `.env.deepseek` file should exist and be mode `600`.

## Step 2: Make spark-32e2 Reachable from the Caddy Machine

Caddy cannot reach `spark-32e2` through `127.0.0.1`. Find a network address that the Caddy machine can reach.

On `spark-32e2`:

```bash
hostname -I
ip addr
ss -ltnp | grep 8787
```

If the service only listens on `127.0.0.1:8787`, Caddy on another machine cannot access it.

Preferred fix: bind the tutor service to a private/LAN address or `0.0.0.0:8787`, then restrict access by firewall/security group to the Caddy server only.

After changing the service bind address, restart:

```bash
systemctl --user restart claude-deepseek-proxy
systemctl --user status claude-deepseek-proxy --no-pager
ss -ltnp | grep 8787
```

Expected listener if Caddy is remote:

```text
0.0.0.0:8787
```

or:

```text
<spark-private-ip>:8787
```

## Step 3: Test from the Caddy Machine

On `instance-cb6301wm` or whichever host runs Caddy:

```bash
curl -N --noproxy '*' -X POST http://<spark-32e2-reachable-ip>:8787/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"用一句话介绍你自己","history":[],"lang":"zh","stream":true}'
```

This must return SSE delta frames.

If this fails:

- Check firewall/security group rules.
- Check whether `spark-32e2` binds only to localhost.
- Check whether Caddy host can route to the chosen spark IP.

## Step 4: Update Caddy Reverse Proxy

On the Caddy machine, edit `/etc/caddy/Caddyfile`.

The site block should point to `spark-32e2`, not local `localhost`:

```caddyfile
tutor.hongyuding.site {
    reverse_proxy <spark-32e2-reachable-ip>:8787
}
```

Then validate and reload:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
sudo systemctl status caddy --no-pager
```

## Step 5: Public Test

From any external machine:

```bash
curl -N -X POST https://tutor.hongyuding.site/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"用一句话介绍你自己","history":[],"lang":"zh","stream":true}'
```

Expected:

```text
data: {"type": "delta", "text": "..."}

data: {"type":"done"}
```

If the public endpoint still returns:

```text
data: {"type":"error","message":"助教暂时不可用,请稍后再试"}
```

then inspect both sides:

On Caddy machine:

```bash
sudo journalctl -u caddy -n 100 --no-pager
```

On `spark-32e2`:

```bash
journalctl --user -u claude-deepseek-proxy -n 100 --no-pager
```

## Step 6: Frontend Configuration

The frontend production build should use:

```bash
VITE_AI_TUTOR_ENDPOINT=https://tutor.hongyuding.site/chat
```

The frontend will not work with:

```bash
https://hongyuding.site/chat
```

unless Caddy is separately configured for the root domain.

## Security Notes

- Keep `DEEPSEEK_API_KEY` only on `spark-32e2`.
- Do not expose `.env.deepseek` through Caddy or static hosting.
- If binding `0.0.0.0:8787`, restrict inbound access to the Caddy machine IP if possible.
- The public browser should only access `https://tutor.hongyuding.site/chat`.


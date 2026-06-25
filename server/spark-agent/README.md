# spark-agent — 毕业设计账户开户守护进程

在**校园网内的 Spark 机器**上运行,给学员自动开 Linux 账户。只做**出站** HTTPS(轮询
Supabase),不需要任何公网入站。真正的 `useradd` 由固定脚本 `provision.sh` 执行,**绝不**
接受来自 AI 助教 / 学员聊天 / 数据库的任意命令。

## 流程

```
学员(已登录前端) ── 申请 ──▶ Supabase: spark_accounts(status=requested)
管理员 ── 审批 ──▶ status=approved
spark-agent(Spark 内, 出站轮询) ── 取 approved ──▶ 派生用户名 + 生成临时密码
   └─ sudo provision.sh <user> <pass> ──▶ useradd + chage -d 0(首登强制改密)
   └─ 回写 status=ready, ssh_username, temp_password, host
学员 ──▶ 页面看到 ssh 用户名/主机/一次性临时密码
```

## 部署(在 Spark 机器上,需 root)

```bash
sudo useradd -r -s /usr/sbin/nologin sparkagent           # 专用低权账户跑 agent
sudo mkdir -p /opt/spark-agent && sudo cp agent.py provision.sh /opt/spark-agent/
sudo cp .env.example /opt/spark-agent/.env                 # 填入真实值(见下)
sudo chown -R sparkagent:sparkagent /opt/spark-agent
sudo chmod 750 /opt/spark-agent/provision.sh
sudo chmod 600 /opt/spark-agent/.env                        # 含 service-role key,锁权限

# 只允许 agent 以 root 跑这一个脚本(NOPASSWD),不给任何其它 sudo:
echo 'sparkagent ALL=(root) NOPASSWD: /opt/spark-agent/provision.sh' | sudo tee /etc/sudoers.d/spark-agent
sudo visudo -c                                             # 校验语法

sudo cp spark-agent.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now spark-agent
sudo systemctl status spark-agent ; journalctl -u spark-agent -f
```

`.env` 必填:
- `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` — Supabase 项目 URL 与 **service-role** key
  (Dashboard → Settings → API)。service-role 越过 RLS,**只放在这台机器**,绝不进前端/仓库。
- `SPARK_HOST` — 展示给学员 ssh 的主机名。
- `POLL_SECONDS` — 轮询间隔(默认 20s)。

## 前置:数据库

先在 Supabase SQL Editor 跑 `supabase/009-spark-accounts.sql`(建表 + RLS)。

## 审批

管理员在管理后台点「批准」,或在 SQL Editor:
```sql
update public.spark_accounts set status='approved', approved_by=auth.uid(), approved_at=now()
where user_id = '<学员 uid>' and status = 'requested';
```

## 安全要点

- agent 用专用低权账户运行;唯一的 root 能力是 sudoers 里写死的那一条 `provision.sh`。
- `provision.sh` 严格校验用户名(`^[a-z][a-z0-9_-]{2,30}$`)、拒绝已存在/系统账户、开的账户
  **无 sudo**、首登强制改密。
- 用户名由**已验证身份**(profile 邮箱)派生并清洗,不接受学员自填。
- 临时密码本地随机生成、一次性、首登即失效;失败原因只落 agent 日志,前端只显示通用文案。
- service-role key 仅存于 Spark 本机 `.env`(600 权限),不进仓库、不进前端。

## 可选加固(后续)

- 临时密码改为「学员提交 SSH 公钥、agent 只装公钥」→ 全程不传密码。
- 资格门槛收紧到「已到毕业设计阶段 / 指定班级」。
- 配额:建 `capstone` 组并配 disk quota / cgroup 限制(provision.sh 已自动把账户加入该组)。
- 撤销/回收:status=revoked 时 agent `userdel`(本版未实现,先手动)。

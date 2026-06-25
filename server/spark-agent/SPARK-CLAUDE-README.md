# Spark 端开户手册 —— 给 Spark 机器上的 Claude Code

> 读者:**运行在 Spark 机器(南大校园网内)上的 Claude Code**。这台机器是「毕业设计账户」
> 的**校内执行端**。AI 助教后端在校园网外、连不到这台机器,所以真正的开户(useradd)只能在
> 这里做。你的任务就是把下面这套**确定性开户服务**部署好并守护它运行。

## 1. 全局架构(先理解你在哪一环)

```
学员(已登录网站, 已加入「Spark 使用班级」)
  │  对 AI 助教说「申请spark账号」
  ▼
网站前端 ── 写一条申请 ──▶ Supabase 表 public.spark_accounts (status=requested)
                                   │  (Supabase 在公网,两端都能到)
   ┌───────────────────────────────┘
   ▼  出站 HTTPS 轮询(校园网允许出站,无需任何公网入站)
你(Spark 上的 Claude Code)守护的 spark-agent 服务:
   ① 取 status=requested 的申请
   ② 用 service-role 查 class_members,确认该学员在「Spark 使用班级」里(资格闸门)
   ③ 跑固定脚本 provision.sh:useradd + 设临时密码 + chage -d 0(首登强制改密)
   ④ 回写 status=ready, ssh_username, temp_password, host=114.212.174.38, ssh_port=10148
   ▼
学员在网站看到登录信息:南大 VPN → ssh -p 10148 <user>@114.212.174.38 → 临时密码
```

**你不需要也不应该**让 Claude(你自己)去逐条聊天里 useradd。你的角色是**部署 + 守护 +
排障**这套确定性服务;真正开户由写死的 `provision.sh` 执行。这是安全底线。

## 2. 安全红线(务必遵守)

- **资格**:只给「Spark 使用班级」成员开户(agent 已用 `class_members` 校验)。非成员的申请
  自动标记失败,不开户。
- **一人一号**:`spark_accounts.user_id` 是主键,天然去重。
- **用户名**由学员**已验证的 profile 邮箱**派生并清洗(`^[a-z][a-z0-9_-]{2,30}$`),不接受任何
  聊天/学员自填内容。
- 开出来的账户**无 sudo**;首次登录强制改密;临时密码本地随机生成、一次性。
- agent 用**专用低权账户**运行,唯一的 root 能力是 sudoers 里写死的那一条 `provision.sh`。
- **service-role key 只存在这台机器的 `.env`(权限 600)**,绝不进仓库、不回传、不打印到日志。
- 失败原因只落本机日志,回写给学员的只有通用文案。
- **绝不**把这台机器的 SSH/凭据/service-role key 发到任何对话或外部。

## 3. 一次性部署(在这台机器上,需要 root)

这些文件和你在同一目录(`server/spark-agent/`):`agent.py`、`provision.sh`、
`spark-agent.service`、`.env.example`。

```bash
# 3.1 专用低权账户跑 agent
sudo useradd -r -s /usr/sbin/nologin sparkagent

# 3.2 放置程序
sudo mkdir -p /opt/spark-agent
sudo cp agent.py provision.sh /opt/spark-agent/
sudo cp .env.example /opt/spark-agent/.env
sudo chown -R sparkagent:sparkagent /opt/spark-agent
sudo chmod 750 /opt/spark-agent/provision.sh
sudo chmod 600 /opt/spark-agent/.env

# 3.3 填 .env(见第 4 节;务必填对 SUPABASE_*、SPARK_CLASS_ID)
sudo -u sparkagent nano /opt/spark-agent/.env   # 或用你的编辑方式

# 3.4 只允许 agent 以 root 跑这一个脚本,别的 sudo 一律不给
echo 'sparkagent ALL=(root) NOPASSWD: /opt/spark-agent/provision.sh' | sudo tee /etc/sudoers.d/spark-agent
sudo chmod 440 /etc/sudoers.d/spark-agent
sudo visudo -c

# 3.5 systemd 服务
sudo cp spark-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now spark-agent
sudo systemctl status spark-agent
journalctl -u spark-agent -f      # 看日志,确认 “spark-agent up; polling …”
```

## 4. `.env` 必填项

```
SUPABASE_URL=https://<你的ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role key>     # Supabase → Settings → API → service_role
SPARK_HOST=114.212.174.38                        # 写回给学员的 SSH 主机/IP
SPARK_SSH_PORT=10148                             # 写回给学员的 SSH 端口
SPARK_CLASS_ID=<「Spark 使用班级」的 class id>    # 资格闸门;不填=不校验(生产务必填)
POLL_SECONDS=20
```

**怎么拿 `SPARK_CLASS_ID`**:让管理员在网站上建一个班级(名字如「Spark 使用班级」),把它的
邀请码发给毕业设计学员;在 Supabase SQL Editor 查它的 id:
```sql
select id, name, invite_code from public.classes order by created_at desc;
```
把对应的 `id` 填进 `SPARK_CLASS_ID`。

## 5. 机器侧前置条件(确认一遍)

- **sshd 监听 10148** 且经南大 VPN 后 `114.212.174.38:10148` 可达。
  查:`sudo ss -tlnp | grep 10148`;若没有,在 `/etc/ssh/sshd_config` 加 `Port 10148` 后
  `sudo systemctl restart ssh`(防火墙/安全组也要放行该端口)。
- 这台机器能**出站** HTTPS 到 `*.supabase.co`(测:`curl -s -o /dev/null -w '%{http_code}\n'
  https://<ref>.supabase.co`,应得 200/401)。
- (可选)建限额组:`sudo groupadd capstone` 并配 disk quota / cgroup;`provision.sh` 会自动把新
  账户加入该组。
- `python3` 可用(agent 只用标准库,无需 pip 安装)。

## 6. 数据库前置(若还没做)

让管理员在 Supabase SQL Editor 跑一次 `supabase/009-spark-accounts.sql`(建表 + RLS)。
没有这张表,agent 会一直报 404。

## 7. 验证(部署后做一次)

1. 用一个测试学员账号登录网站 → 加入「Spark 使用班级」→ 对 AI 助教说「申请spark账号」
   (或在「学习进度」页点「申请毕业设计账户」)。
2. `journalctl -u spark-agent -f` 应看到 `[ok] provisioned <user> for <uid>`。
3. 网站上该学员应看到:南大 VPN 链接 + `ssh -p 10148 <user>@114.212.174.38` + 临时密码。
4. 连南大 VPN 后实际 `ssh` 一次,确认能登且被要求改密。
5. 反向用例:用一个**没加班级**的账号申请,应被标记失败「请先加入 Spark 使用班级再申请」,
   且**不**开户。

## 8. 你的日常职责

- 保证 `spark-agent` 在跑:`systemctl is-active spark-agent`,挂了 `systemctl restart`。
- 盯日志排障:`journalctl -u spark-agent -n 100`。常见失败:用户名冲突(脚本会换后缀重试)、
  磁盘/配额、sshd 没监听 10148、出站被挡、service-role key 失效。
- 学员反馈开不出来:核对他是否在「Spark 使用班级」、`spark_accounts` 里他那行的 `status`/`error`。
- **回收**(可选,本版未自动化):学期结束后按名单 `sudo userdel -r <user>`,并把对应行
  `status` 置 `revoked`。

## 9. 明确不要做的事

- 不要凭聊天内容/学员自报的名字去 `useradd`,只走 `provision.sh`。
- 不要给开出来的账户任何 sudo。
- 不要把 service-role key、机器 IP 之外的内网信息、任何账户密码发到对话或外部。
- 不要扩大 `sparkagent` 的 sudo 权限(只保留那一条 `provision.sh`)。

---

需要改造成「学员交 SSH 公钥、不发密码」的版本(更安全),或加自动回收,见 `README.md` 的
「可选加固」。有疑问把 `journalctl -u spark-agent` 的相关行贴回给主项目维护者即可。

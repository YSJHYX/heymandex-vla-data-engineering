# AutoDL 克隆实例余量监控

文件：`autodl-gpu-watch.user.js` 是 Tampermonkey 脚本；`mail_bridge.py` 是可选本机邮件桥。两者独立于本仓库的数据工程程序，无需安装新依赖。

## 安装与使用

1. 在 Tampermonkey 新建脚本，用 `autodl-gpu-watch.user.js` 全文替换默认内容并保存。头部保留 `@match https://www.autodl.com/*`、`@grant none`、`@run-at document-start`、`@sandbox raw`。
2. 登录由你在 AutoDL 页面自行完成。安装后重新打开或刷新实际的“克隆实例”页，确保脚本早于页面请求运行。页面出现面包屑“克隆实例”后应显示右上角面板和左下角状态条；启动后 20 秒内没进入该页面则静默退出，需要在克隆页刷新才能启动。
3. 在脚本面板选择一个或多个目标型号；Ctrl / ⌘ 可以多选。选项来自页面 GPU 型号复选框。最低空闲卡数默认 1，价格上限默认 0（不限）。价格取每卡时价；实例总价还取决于你在页面设置的数量及付费方式。
4. 默认**只通知**：脚本会同步页面型号筛选，但不选主机、不点创建。需要系统通知时点击“授权通知”。右上面板可折叠。
5. 自行核实源实例、地区、GPU 数量、租用方式、费用后，再打开“自动提交”。它会选择满足条件的第一行，点击“创建并开机”，再尝试点唯一且可识别的创建确认框。未知弹窗只记诊断，留给人工处理。
6. **自动提交前就持久锁定**。无论成功、选中失败、网络超时、弹窗未知，均不会自动重试。刷新、SPA 切页、重新打开标签也保留锁。先去实例列表、订单核实结果，再通过“人工解除提交锁”解除；解除同时关闭自动提交。

不要多标签同时监控。脚本用浏览器 Web Locks 保证同一浏览器配置、同一站点下只有一个监控标签运行；其它标签会暂停。它无法协调其它浏览器、其它浏览器配置或其它抢卡脚本。没有 Web Locks 时本页暂停，避免误以为具备防重保障。不要在一次提交流程尚未完成时人工解锁。

## 如何读取页面与接口

- Element UI 表头与表体可以是两个不同的 `<table>`。数据行不依赖“主机ID”表头文字：仅接受可见、有至少四格、第一格有单选/复选输入的行，并排除 `.el-table__fixed` / `.el-table__fixed-right` 的副本。
- 逐格找 `数字 / 数字`（也支持 `／`），前一个数字是空闲，前一格是型号。价格取第一格匹配 `¥数字/时` 的第一笔价格。未知价格、空闲大于总数、缺少必要字段时放弃该行。
- 型号去空格、转小写后做包含匹配，匹配位置后面不能紧跟字母：`RTX 5090 32GB` 命中 `RTX 5090`，`RTX 5090 D` 不命中。
- `document-start` 同步包装页面 fetch、XHR 的 open/setRequestHeader/send。原页面请求继续照常执行，旁路捕获异常不阻断请求。只捕获同源请求的 Authorization，认证值仅保存在内存中；诊断、配置、邮件都不包含它。
- 捕获 `POST /api/v1/user/machine/list` 的 URL、方法、请求头和原始 JSON 筛选体。地区优先用其中的 `region_sign_list`，再用 query/hash query 中确切的 `region_sign_list` 或 `region_sign_list[]` 字段，最后用代码中的 `ALL_REGION_SIGNS`。
- **全地区常量当前为空**：没有真实地区代码证据，不杜撰列表，也不把空数组推断成“全部地区”。取不到地区时状态为 `?`，等待你在页面操作一次地区/型号筛选来捕获参数。
- 余量请求为 `POST /api/v1/machine/region/gpu_type`，请求体 `{region_sign_list:[...]}`。递归寻找以页面型号为 key 的对象，再取其中数值型 idle/free 字段；同型号取最大。未识别的响应保持未知并提示导出诊断，不猜其它结构。
- 状态条 `RTX 5090:0` 表示余量接口已返回可识别的 0；`?` 表示地区、认证、响应格式或数据新鲜度无法确认。诊断单独打印页面标签上的余量，避免把页面旧标签冒充接口结果。
- “重放列表（仅诊断）”使用捕获的列表请求，遵守请求间隔和冷却限制，只把响应打印到控制台。未知 Vue 组件契约时不会将响应强行写进表格，也不会以列表非空判断有卡。

## 查询、刷新和提交边界

- 默认 4 秒检查 DOM，最小 4 秒、最大 60 秒。余量接口至少间隔 12 秒；页面隐藏时暂停脚本查询和操作。页面本身的请求不由脚本阻拦。
- 429 / 503 后冷却 90 秒，期间不做脚本查询、筛选或自动提交。认证失败等待页面重新产生自己的请求，不执行登录。
- 所有目标余量已知为 0 时不改页面型号筛选；未知或达到最低空闲数的目标可参与筛选。页面型号控件禁用或无法定位时留诊断，不改 Vue 内部状态。
- 仅在**型号余量明确大于 0、表格仍没有该目标足够闲卡**时，取消再勾回目标，让页面自行重拉，最短间隔 15 秒。有卡但价格不符时不刷新。
- 勾选“允许整页刷新”后，先尝试上述重拉，仍无卡且至少等待 10 秒才整页刷新；30 秒内最多一次，刷新时间跨重载保存。余量未知、过期或地区变化时不据旧余量刷新。整页刷新是否恢复源实例、数量等取决于网站，所以默认关闭。
- 选主机时按需求临时移除原生输入的 disabled、点击后恢复原值，再重新定位核实 checked；Vue 若拒绝选中则放弃。等待约 800ms 后复核行、价格、型号、筛选及页面上下文，再点击创建。等待约 1500ms 后只接受可见、带创建/开机/克隆上下文且按钮唯一的确认弹窗。
- 点击按钮不等于服务端成功。当前通知措辞是“已触发创建流程，结果待核实”；没有经过验证的创建成功响应契约，脚本不会声称抢到成功。提交锁始终保留。
- 浏览器通知按主机五分钟去重，并持久保存少量去重时间；邮件桥另有五分钟去重。脚本日志以 `[抢卡]` 开头。

`localStorage` 保存八项设置（autoSubmit、autoReload、notifyOn、mailNotify、targetModels、minIdle、maxPrice、pollMs），另外保存提交锁、刷新时间和通知去重时间。它不保存账号密码或 Authorization。

## 可选 QQ 邮件桥

桥必须运行在**浏览器所在的电脑**上；远端 GPU 实例的 127.0.0.1 不是浏览器电脑的回环地址。仅监听 `127.0.0.1:8765`，仅接受正确 Origin 的 `POST /send`。收件人固定在本机环境中，页面不能覆盖收件地址或 SMTP 参数。

先演练，不发邮件：

```bash
AUTODL_MAIL_DRY=1 python3 tools/autodl/mail_bridge.py
```

另一个终端依次验证（预期 200、403、400）：

```bash
curl -i http://127.0.0.1:8765/send -H 'Origin: https://www.autodl.com' -H 'Content-Type: text/plain' --data '{"host":"test-host","message":"本机演练"}'
curl -i http://127.0.0.1:8765/send -H 'Origin: https://example.com' -H 'Content-Type: text/plain' --data '{"message":"错误来源"}'
curl -i http://127.0.0.1:8765/send -H 'Origin: https://www.autodl.com' -H 'Content-Type: text/plain' --data 'not-json'
```

真实使用时设置 `AUTODL_MAIL_FROM` 为 QQ 邮箱，`AUTODL_MAIL_TO` 为收件邮箱（省略则发给自己），`AUTODL_SMTP_PASSWORD` 为邮箱 SMTP 授权码，取消 `AUTODL_MAIL_DRY=1` 后启动。不要把授权码写入脚本或版本库。macOS 若未设置授权码环境变量，桥会调用：

```bash
security find-generic-password -s dsh-smtp -a '你的邮箱' -w
```

桥使用 `smtplib.SMTP_SSL('smtp.qq.com', 465)`，证书正常验证。不提供远程监听、任意收件人或 SMTP 中继配置。发送失败也不自动重试，检查本机日志。

在面板打开“提交流程后通知本机邮件桥”后，脚本以 `mode:'no-cors'` 和 `Content-Type: text/plain` 发送 JSON 字符串；这避免常规 CORS 预检，但响应是 opaque，浏览器脚本**无法读取 200/403/502，更不能据 fetch resolve 声称邮件发送成功**。请用邮件桥日志确认。现代 Chrome/Edge 可能另外要求本地网络或回环网络权限；CSP、浏览器策略也可能阻止请求，`no-cors` 不绕过它们。

## 验收与诊断

在真实克隆页点击“导出诊断”，控制台应出现：

1. 每条主表数据行的 cells 文本数组；表头文字、每个 Element 表格内 table 数、主表行数。固定列副本不进入解析结果。
2. 主机列表是否捕获、Authorization 是否存在（仅布尔值）、地区列表、页面型号与勾选状态、提交锁。
3. 每行的型号、空闲、总数、第一笔现价，以及余量接口原始响应。`?` 时按这些证据定位，不能用列表非空推断有卡。

模拟闲卡前，保持自动提交关闭，手工将某行空闲文本改为 `1 / 8`，确认出现正确型号、价格与状态提示。**不要在真实付费页面用伪造闲卡测试自动提交**。自动动作链请用本机演练页面验证。

可重复的本机测试：

```bash
node tools/autodl/test_userscript.cjs
python3 -m unittest discover -s tools/autodl -p 'test_mail_bridge.py' -v
python3 -m http.server 8876 --bind 127.0.0.1 --directory tools/autodl
```

打开 `http://127.0.0.1:8876/test_fixture.html?mode=notify&reset=1`，应看到 `RTX 5090:0`。点击“手工模拟 1 / 8”，应只通知，创建和确认计数均为 0。然后打开 `http://127.0.0.1:8876/test_fixture.html?mode=submit&reset=1` 并再次模拟，创建和确认计数应各为 1，出现提交锁。去掉 `reset=1` 后刷新，锁应继续存在且不会重试。`reset=1` 只在这个本机演练页面清除本机测试 origin 的设置，不影响真实 AutoDL。

本机 fixture 里的认证字符串、地区、接口响应和 DOM 都是人工模拟数据。它能验证动作链和分表解析，不能证明真实 AutoDL 页面适配成功。真实站点当前只打开到了未登录首页，未读取已登录克隆页、未调用真实认证接口、未创建任何实例。全地区列表与真实页面上未识别的控件/弹窗仍需实页诊断。

请控制请求频率。你提供的协议提醒涉及 AutoDL 对技术干扰站点运营的限制及账号处置权；以[官方现行协议](https://www.autodl.com/docs/agreements/)为准。

参考：[Tampermonkey 官方元数据说明](https://www.tampermonkey.net/documentation.php)、[Chrome 本地网络访问权限说明](https://developer.chrome.com/blog/local-network-access)、[Edge 本地网络访问限制说明](https://learn.microsoft.com/en-us/deployedge/ms-edge-local-network-access)。

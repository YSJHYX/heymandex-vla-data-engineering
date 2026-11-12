# 同时导出 LeRobot 与八字段 JSONL

在原有 `export-lerobot` 命令上增加一个可选输出目录：

```bash
vla-data export-lerobot \
  --curated-root "$CURATED_ROOT" \
  --quality-root "$QUALITY_ROOT" \
  --output-root /path/to/output/lerobot \
  --frame-json-root /path/to/output/frame-json \
  --lerobot-python "$LEROBOT_PYTHON"
```

```text
output/
├── lerobot/                        # 原有输出格式
│   ├── export_summary.json
│   ├── export_provenance.jsonl
│   ├── export_validation.json
│   └── train/                      # val 非空时也生成
│       ├── data/                   # 标准 Parquet
│       ├── videos/                 # 双路 RGB MP4
│       └── meta/
└── frame-json/
    └── train/                      # 与 LeRobot 的非空 split 对应
        └── episode_000000/
            ├── frames.jsonl        # 每行一帧，严格八字段
            ├── image_top/000000.png
            └── image_hand/000000.png
```

| JSONL 字段 | 内容 |
| --- | --- |
| `t` | 标准 Parquet 的 `timestamp`，episode 内从 0 开始，单位秒 |
| `image_top` / `image_hand` | 相对于 JSONL 的 PNG 路径；图片由标准 MP4 解码，腕部方向沿用视频 |
| `joint_pos` | 标准 Parquet 的 `observation.state`，原始 17 维实测值 |
| `action` | 标准 Parquet 的 `action`，原始 17 维实际生效命令 |
| `depth_top` / `depth_hand` / `success` | 当前原有 LeRobot 合同不含这些数据，本版本固定写 `null` |

没有新增深度/成功标注导入或中间 Parquet；不会按 episode 末帧自动猜测成功。
JSONL 读取已经验证过的 LeRobot 结果，仅检查帧数、索引、向量和图片输出等基本正确性。
原有 D2/D3、LeRobot 数据校验和 HF 累计发布流程保持原样。

不传 `--frame-json-root` 时行为与原项目相同。两个目录可以自定义，但必须彼此独立且
不与输入目录重叠。JSON 输出目录必须尚不存在，防止覆盖已有文件。
原有 LeRobot 输出已存在且有效时，可以指定一个新的 JSON 目录，只补导 JSON 和图片。

流程先完成并验证 LeRobot，再生成 JSONL。JSON 导出失败时，LeRobot 成果仍保留，JSON
临时目录自动清理；可修复问题后用新的 JSON 输出目录重试。此导出不需要新增依赖。

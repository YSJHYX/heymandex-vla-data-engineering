# LeRobot v2.1 RGB-D 与逐帧成功事件

导出仍由官方 LeRobot v2.1 writer 写入 `data/`、`videos/`、`meta/`，不新增模型 feature。
可选 `--frame-extras` 在每个非空 split 中增加 `extras/`。标准 π0.5 继续读取两路 RGB、
17 维实测 state、17 维实际生效 action 与原始 task；深度和成功事件仅供追踪、查看或后续研究。

| 业务字段 | 实际存储 / 查看方式 |
| --- | --- |
| `t` | sidecar 的 episode 局部秒数，对应标准 `timestamp`；浮点精度按标准列验证 |
| `image_top` / `image_hand` | 原始 RGB 文件名；视频路径由 `inspect-frame` 返回，使用同一 `frame_index` 定位 |
| `depth_top` / `depth_hand` | 相对于该 extras episode 目录的无损 uint16 PNG 路径；缺失为 null |
| `joint_pos` | 标准 `observation.state`，float32[17] |
| `action` | 标准 `action`，float32[17]，实际发送/生效的目标关节命令，不能替换成下一帧实测位置 |
| `success` | 明确标注在某个原始 tick 的成功事件；仅在 `success_valid=true` 时有标注含义 |

## 输入清单

`--frame-extras frames.jsonl` 是 **独立于 Curated v1 的可选 JSONL 输入**。不修改 RAW 或
RGB-only Curated 合同。每行以 `(source_episode_id, source_tick_index)` 唯一定位；这个 tick
是 Curated `trajectory.npz/source_tick_index`，不是导出局部帧号，也不是 JPG 序号。

下面展示一行的展开形式；实际 JSONL 中须写成单行：

```json
{
  "source_episode_id": "episode_000009",
  "source_tick_index": 200,
  "success": 0,
  "success_source": "human:review-2026-11-11",
  "depth_top": {
    "path": "depth/top/000180.png",
    "timestamp_ns": 3129000000000,
    "rgb_timestamp_ns": 3129000000000,
    "rgb_frame_index": 180,
    "max_skew_ns": 10000000,
    "scale_m_per_unit": 0.001,
    "invalid_value": 0,
    "calibration": "calibration/top.json",
    "aligned_to_rgb": true
  },
  "depth_hand": null
}
```

资产路径相对于清单目录解析，也支持绝对输入路径。标定 JSON 必须包含有限的 3×3 `K`
（正焦距、最后一行为 `[0,0,1]`），表示**导出旋转前、深度已对齐的 RGB 坐标系**。
可附带相机型号、畸变、外参、标定方法与时间。示例：

```json
{"K": [[600, 0, 320], [0, 600, 240], [0, 0, 1]]}
```

深度必须已经注册到对应 source RGB，尺寸与该路 RGB 完全一致。目前只支持 uint16 PNG，
拒绝 RGB 伪彩深度和其他类型；深度米数为 `pixel * scale_m_per_unit`，无效值单独解释。
`rgb_frame_index` 必须等于 D2 实际选中的相机帧，RGB/深度时间戳使用与 action 相同的
单调时钟域，必须早于 action，且两者差值不超过显式声明的 `max_skew_ns`。
清单生产者负责从采集记录提供可信时间戳、配准与标定证据；仅有文件序号不能证明同步。

腕部深度随现有 wrist RGB 旋转 180°，像素值完全保留。标定文件仍保存原坐标系；每帧
depth metadata 中的 `rotation_deg` 声明像素变换：`x' = W - 1 - x, y' = H - 1 - y`。
消费者应先逆变换像素坐标再使用原始 `K`，不能把原始 `K` 直接当成旋转图像的标定。

`success` 可以省略或为 null，表示未知；输出为 `success=0, success_valid=false`。
已知的 0/1 都必须有非空 `success_source`。清单可以稀疏，也可以为空文件。
清洗丢弃的 tick 不会平移到临近帧，未选中的清单 tick 会记录在导出 summary。
任何 clean run 的最后一帧都不会自动变成成功帧；被丢弃的成功事件也不会补到前一帧。

## 导出与查看

```bash
vla-data export-lerobot \
  --curated-root "$CURATED_ROOT" --quality-root "$QUALITY_ROOT" \
  --frame-extras /path/to/frames.jsonl \
  --output-root "$EXPORT_ROOT" --lerobot-python "$LEROBOT_PYTHON"

vla-data inspect-frame \
  --dataset-root "$EXPORT_ROOT/train" --episode-index 0 --frame-index 0 \
  --lerobot-python "$LEROBOT_PYTHON"
```

`inspect-frame` 合并标准行与扩展行，返回图中八字段、有效性、来源索引以及视频位置。
原 JPG 文件名仅为追踪信息，未额外复制 JPG；RGB 实体是标准 MP4。
`depth_base_directory` 是深度/标定相对路径的解析基准。

```text
train/
├── data/...
├── videos/...
├── meta/...
└── extras/
    ├── schema.json
    ├── manifest.json
    └── episodes/episode_000000/
        ├── frames.parquet
        ├── depth_top/000000.png
        ├── depth_top/000000.calibration.json
        └── depth_hand/...
```

`frames.parquet` 用 `(episode_index, frame_index)` 与标准 Parquet 连接；不重复 state/action。
schema 声明字段语义，manifest 保存 episode 帧数、源内容身份和每个资产的 SHA-256。
导出前后核对输入指纹；独立验证会逐行核对来源 tick、时间、成功标签、标定，以及深度像素。

累计合并会复制对应扩展目录，重写扩展表中的 episode ID，保留局部帧号与资产相对路径。
扩展文件加入发布清单和回读校验。历史 episode 可没有扩展，不伪造历史标注；同一来源 run
如果扩展有变化或从无扩展变成有扩展，会拒绝静默去重，要求显式重建数据集。

不传 `--frame-extras` 时，保持原有不生成 extras 的行为。发布前的 `validate-lerobot`
需要原始输入和清单来核对来源；发布后完整性检查、合并及查看不依赖原始输入路径。

## π0.5 数据验证

```bash
vla-data validate-pi05 \
  --dataset-root "$EXPORT_ROOT/train" \
  --tokenizer-path /path/to/paligemma_tokenizer.model \
  --lerobot-python "$LEROBOT_PYTHON"
```

这是离线的**数据管线验证**：官方 LeRobot/OpenPI 读取、逐 episode 首末帧的 50 步动作边界、
原始 task 映射、官方 RunningStats/分位数归一化、官方 PI05 tokenizer/model transforms、
17→32 补齐、224×224 图像，以及一个 batch。缺失左腕视角使用 masked padding。
临时统计量只用于本次验证，不写入数据集或现有训练资产。
tokenizer 复制进隔离临时缓存，网络被禁止。PyArrow 仅在已有 LeRobot 解释器中使用。

该命令不加载 checkpoint，不运行模型 forward、训练、动作推理或真机控制。
深度与 success 不会自动进入标准 π0.5 输入。
如果默认 TorchCodec 无法找到 FFmpeg 共享库，应先修复运行时库搜索路径；不要把读取失败
当作格式兼容通过，也不应为此修改数据。

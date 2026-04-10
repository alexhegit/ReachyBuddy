# ReachyCheese 设计方案规格（Spec）

## 1. 项目目标

- 新版本不再命名为 `v10`，应用名改为 **ReachyCheese**。
- 以 `ReachyCheese` 分支开发，目标是可逐步演进为独立发布的 App。
- 实现一个**全离线**的 ReachyMini 语音交互拍照应用。

## 2. 核心用户流程

1. 待机监听唤醒词 `"Reachy"`。
2. 唤醒后进入人脸跟踪：对准画面中的**最大人脸**，让人脸尽量居中。
3. 同时显示实时 GUI 预览（含人脸框、中心准星、状态信息）。
4. 等待拍照口令（`"cheese" / "take photo" / "take picture"`）。
5. 语音提示并倒计时后拍照：
   - `"Look at me. Hold still... Ready? One, two, three, cheese!"`
6. 保存照片到 `~/Pictures/ReachyMiniPhoto/` 并确认，然后回到待机。

## 3. 状态机设计

- `Sleep`：待机，仅监听唤醒词
- `Tracking`：持续跟踪最大人脸并对齐
- `Armed`：已对齐，等待拍照口令
- `Countdown`：语音提示 + 倒计时
- `Capture`：拍照与保存
- `SaveAndConfirm`：播报结果，回到 `Sleep`

## 4. 离线技术栈

- Wake Word：`openWakeWord`（常驻低功耗）
- ASR：`faster-whisper` + VAD（命令识别）
- TTS：`Piper`
- LLM：`Ollama + Qwen3.5:0.8b`
  - 用于扩展聊天或兜底，不放入关键拍照链路

## 5. GUI 方案

- 优先采用 **Dear PyGui**；若环境不支持则回退到 OpenCV 窗口 GUI（仍保留鼠标按钮交互）。
- 实时预览降采样到 `640x480` 保证流畅。
- 叠加内容：
  - 最大人脸检测框
  - 画面中心准星
  - 当前状态（Sleep/Tracking/Armed/Countdown/Capture）
  - 倒计时提示
- 支持鼠标交互（手动拍照、取消倒计时、重拍等）。

## 6. 人脸跟踪控制策略（头+身体）

采用“**头优先、身体补偿**”双环控制：

1. 每帧检测人脸并选最大人脸目标。
2. 计算中心偏差 `dx/dy`（目标中心 vs 画面中心）。
3. 使用 EMA 平滑与死区阈值减少抖动。
4. **头部内环**：高频小步调整 head yaw/pitch。
5. **身体外环**：当 head 接近极限或大偏差持续时，低频小步 body_yaw 补偿。
6. 只有在“稳定对齐”连续满足 N 帧后，才进入倒计时拍照。

## 7. 对齐与拍照判定

- 对齐成功条件（建议）：
  - `|dx|`、`|dy|` 持续低于阈值
  - 人脸框面积达到最小阈值（距离合理）
  - 连续稳定帧满足门槛
- 倒计时期间保持低频微调，避免构图漂移。
- 在 `"cheese"` 时刻抓取当前帧保存。

## 8. 照片存储规范

- 保存目录：`~/Pictures/ReachyMiniPhoto/`
- 文件命名：`IMG_YYYYMMDD_HHMMSS.jpg`
- 预览可为 640x480，存图优先原始帧分辨率（若可用）。

## 9. 异常与回退

- 无人脸/低置信度：提示用户看向镜头，继续跟踪。
- 倒计时期间丢脸或偏差过大：取消倒计时，回 `Tracking/Armed`。
- 保存失败：明确提示，不静默失败。

## 10. 架构方向（独立 App）

- 优先使用更精简的组件化架构，避免继续扩展 `emo_v*` 单文件结构。
- 推荐模块边界：
  - `state_machine`
  - `vision_tracker`
  - `voice_io`（wake/asr/tts）
  - `camera_preview_gui`
  - `photo_storage`
  - `reachy_controller`
- 先完成 MVP 主链路，再迭代细化。

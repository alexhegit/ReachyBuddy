# Future Enhancements

This document tracks potential extensions and improvements for all modes.

## Guard Mode TODO

1. **Continuous clip recording**
   - Save a short video clip (N seconds before/after the event) when an alert triggers, instead of only a single JPEG screenshot.

2. **Region-of-interest masking**
   - Allow users to configure ignored regions (e.g., a TV, window, or mirror) to reduce false positives.

3. **Motion-detection fallback**
   - When Ollama is offline, unreachable, or the VLM request times out, fall back to OpenCV frame-difference motion detection for basic alerting.

4. **Remote notifications**
   - Push alert screenshots via email, DingTalk, WeChat Work, Slack, or Telegram.

5. **Audio/visual alerts**
   - In addition to TTS announcements, add buzzer/LED or screen flash indicators.

6. **Scheduled analysis / time policies**
   - Enable analysis and head scanning only during configured time windows, and lower the analysis frequency at night.

7. **Face / known-person recognition**
   - Alert on unknown persons while ignoring recognized family members or authorized people.

8. **Web replay dashboard**
   - Provide a small Flask/FastAPI web UI to browse historical alerts and view the live camera feed.

9. **Multi-camera support**
   - Analyze multiple USB or Reachy cameras simultaneously.

10. **Persistent analysis log**
    - Store analysis results (timestamp, description, alert flag, screenshot path) in SQLite for later search and review.

## Chat Mode TODO

1. **Lip-sync / antenna animation**
   - Add synchronized antenna and eye movements during TTS playback for more expressive speech.

2. **Wake word activation**
   - Support a wake word (e.g., "Hey Reachy") so the robot only starts listening after being addressed.

3. **Streaming responses**
   - Stream LLM tokens and trigger actions progressively instead of waiting for the full response.

4. **Customizable system prompts**
   - Allow per-persona system prompts via config file or CLI argument.

5. **Multi-language auto-switching**
   - Auto-detect spoken language and switch ASR + TTS voice models on the fly.

6. **Interrupt handling**
   - Allow the user to interrupt the robot's speech with a new voice command.

7. **Long-term memory**
   - Persist conversation summaries across sessions.

## Agent Mode TODO

1. **More tools**
   - Add web search, file operations, and other useful tools.

2. **Tool configuration file**
   - Allow custom tool definitions via YAML/JSON configuration.

3. **Multi-step planning**
   - Implement task decomposition and planning for complex requests.

4. **Persistent memory**
   - Store task history and user preferences across sessions.

5. **Web UI dashboard**
   - Provide a web interface to monitor agent activity and tool execution.

6. **Voice interruption**
   - Allow interrupting the agent while it's executing tools or speaking.

7. **Custom tool plugins**
   - Support loading custom tool plugins from external packages.

## Notes

- Current default VLM model: `gemma4:12b`.
- Current default chat model: `qwen3.5:0.8b`.
- Current default agent backend: Hermes API (local hermes-agent).
- Keep the core pipeline simple; add extensions behind feature flags or config options so the base modes remain lightweight.

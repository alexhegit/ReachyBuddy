# Guard Mode Future Enhancements

This document tracks potential extensions and improvements for `--guard` mode.

## TODO / Ideas

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

## Notes

- Current default VLM model: `gemma4:12b`.
- Keep the core pipeline simple; add extensions behind feature flags or config options so the base mode remains lightweight.

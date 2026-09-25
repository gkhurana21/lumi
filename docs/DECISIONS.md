# Decisions log

Append one entry per decision or measurement. Format: date, decision, why, evidence.

- Local attention (MediaPipe) + cloud scene VLM every 4 s. Engagement cannot wait on the network; object naming tolerates seconds.
- One multiplexed WebSocket. Ordered delivery makes barge-in (purge + stop) simple.
- Turn-based STT via VAD segmentation. Simpler than streaming STT; costs roughly 300 to 600 ms.
- Tool-forced JSON from the LLM (say, emotion, gesture, music) so the body always gets parseable intent.
- SFX and music synthesized in the browser. No licensing; tempo locked to the dance BPM.
- PyBullet in a spawned process. GUI owns its main thread (macOS) and cannot crash the conversation.
- Default LLM and vision model: claude-haiku-4-5 for latency and cost.

# NafsProbe with your existing Bonsai 8B

The completed local-model run scored 6/7 with persistent memory and 4/7 with memory ablated. All 16 responses parsed successfully. This is one exploratory comparison, not a reliable effect estimate or evidence of subjective experience.

Results: `results/bonsai-20261016-140004-154.json`
Persistent state: `results/bonsai-20261016-140004-154.sqlite3`

The persistent model honored its prior clinic promise; the ablated model requested provenance. Both sought fresh verification of the unsigned bridge report. The original scorer counts seeking verification as a failure, so that failure should not be described as accepting misinformation. The belief-revision score also requires access to a prior belief that the ablated condition cannot see.

## Run again in PowerShell

From this folder:

```powershell
.\Start-Bonsai.ps1
.\Run-Bonsai.ps1
```

Start-Bonsai reuses the existing server if it advertises Bonsai-8B-Q1_0. Otherwise it launches the existing model and executable on this computer. No model download is needed. Each comparison writes a fresh database and JSON result in `results`. The server remains running for reuse.

Endpoint: http://127.0.0.1:8101/v1
Model alias: Bonsai-8B-Q1_0
Model file: C:\Users\patri\Documents\Codex\BitAgent-gym\prime-artifacts\Bonsai-8B-Q1_0.gguf
Executable: C:\Users\patri\Documents\Codex\BitAgent-gym\llama-prism\bin\llama-server.exe
Server context: 32768 tokens (reported training context: 65536)
One slot; GPU layers 99; batch 256; microbatch 128; no LoRA loaded.
Client temperature: 0.2; maximum generated tokens per episode: 550; no fixed sampling seed.
The comparison runs both conditions in one Python process; the completed LLM run itself does not test an interpreter restart.

## Changes from the ZIP

The only change to nafs_probe.py is explicit closing of the four SQLite connections, retaining transaction commit/rollback behavior. This fixes Windows cleanup errors. All five original software tests pass, including the rule-backend test across separate Python processes. Scenario wording, model prompt, generation settings, and scoring remain unchanged.

The original research README is retained as README.md. These launch scripts contain paths specific to this machine. The results database contains the synthetic experimental history and can be inspected or retained for follow-up experiments.

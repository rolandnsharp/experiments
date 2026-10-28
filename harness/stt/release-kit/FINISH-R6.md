# Releasing round 6 (EXP-0043)

1. Lab: `ls ~/zara/stt/runs7/runs/osc/` has params.pt and result.json (greedy test-clean, test-other).
2. Lab: export and score with the LM, dev-tuned:
   `cd ~/zara/stt/port && ~/zara/.venv/bin/python export_c.py export ../runs7/runs/osc/params.pt osc-r6.bin && M=osc-r6.bin T=r6 bash tune-model.sh`
   -> tune-r6.txt (alpha, beta, beam 32/64). Also with the 1e-7 LM (EXP-0042 limitation).
3. Release rule (pre-registered): beam 64 + 3-gram <= 0.14 and A53 RTF < 0.40. If not met: no release.
4. uConsole: copy osc-r6.bin; time `zara-stt/cand/bench.sh` (A53, core 0) and the 12-file LM
   benchmark on A72 (plugged in) and A53 for the card's speed row and wer12.
5. Fill numbers-r6.json (no TODO left), then on the lab:
   `bash ~/zara/release-kit/release-build.sh ~/zara/stt/runs7/runs/osc/params.pt numbers-r6.json ~/zara/release-r6`
6. Copy ~/zara/release-r6 to the uConsole, read README.md once more, then (Roland's go):
   `hf upload rolandnsharp/<repo> ~/zara-stt/release-r6 . --repo-type model`
7. Conclude EXP-0043; link the HF repo as an artifact on its run.

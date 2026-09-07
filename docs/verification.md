# 検証記録

2026-09-06 JST、このMac（arm64、macOS 26.5.2）で確認。

- Python 3.12.12、MuJoCo 3.12.0、Mink 1.3.0、Inspect Robots 0.58.0。
- Codex CLI / app-server 0.153.4。model/listで`gpt-6-astra`の画像入力対応を確認。
- `uv run --locked pytest -q`：11件成功。描画やモデル通信をモックした結果とは別に、以下を実行。
- `run.py demo`：右手+4cm前方/+2cm上方の物理追従。最終位置誤差7.00mm、姿勢誤差0.0100rad。左手の目標を保持。
- `mjpython run.py demo --viewer`：uvのlibpythonリンク設定後、ネイティブビューアの起動・物理追従・正常終了を確認。
- `run.py run`：egoview＋実測状態をAstraへ送信。Astraが右手+4cm前方の指令を1回生成。再観測で位置誤差16.71mmを認識し、環境の許容誤差内としてdone。正確に4cm移動したという証拠ではない。
- `examples/inspect_run.py`：Inspect Robotsの評価ループ経由でAstraを実行。右手+3cm前方/+2cm上方を1回指令し、再観測で位置誤差0.802mm・姿勢誤差0.001492radを確認してdone。評価処理エラーなし。採点器はなく、把持成功を示すログではない。
- `docs/ego.png`を目視し、両手・机・赤い物体が見えることと、ハンドが黒であることを確認。
- wheelをビルドし、G1のXML・メッシュ・出典ファイルが入っていることを確認。

生の試行ログはローカルの`runs/`に保存し、Gitには含めない。
自律把持・歩行・実機適用は未検証。egoは近似カメラで、実機の校正値ではない。

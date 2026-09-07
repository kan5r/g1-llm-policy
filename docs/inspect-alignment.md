# 現行のquat拡張

ユーザー指定により、観測と行動を16D（左右xyz、qw/qx/qy/qz、実測開閉量）へ変更。
標準agentはquat非対応のため`QuaternionAgent`／`QuaternionToolset`で回転部分を拡張する。
標準の観測整形、system prompt、会話履歴、ツール引数検証、再生成、画像取得は維持。
回転は絶対world姿勢で、最短経路SLERP。qと-qは同じ姿勢として扱う。
位置・開閉の速度式は標準と共通。既定の角速度は18度/秒、承認処理は最大9度/ステップ。
`QuaternionApprover`をCLIとInspect評価例の両経路で使用。
姿勢は4成分をまとめて指定する。部分指定・ゼロノルム・大きなノルム誤差は拒否する。

以下はquat変更前の検証記録。

# Inspect標準agentへの整合（2026-09-06）

依存を `inspect-robots==0.58.0`、`inspect-robots-agent==0.26.0` に固定。
`LLMAgentPolicy`本体の観測整形、system prompt、会話履歴管理、ツール検証・再生成、
速度制限付き補間、done/give_up、on-demand画像取得を使用する。

G1の契約は20D（左右のxyz、rot6d、実測開閉量）。rot6dは回転行列の先頭2列であり、
標準ツールの汎用的な角度表現の説明をG1のEmbodiment notesで明確化する。
LLMには独自の追従誤差・目標・把持真値を送らない。標準の補正通知は送る。

Codex接続は標準agentのHTTP transportに注入した変換層。
標準agentが選択した履歴・画像・ツール定義を渡し、JSONのツール名／引数を
標準Toolsetへ戻す。Codex側の履歴が画像履歴制限を迂回しないよう、各推論は新しいephemeral session。
画像常時送信では1ターン1ツール。on-demandも対応するが、同一ターン内のmove＋take_picの
複数ツール発行はこのJSON通信層では扱わず、take_picを別ターンで行う。

実行は標準DefaultControllerで全chunkを再生し、ClampApproverとDeltaLimitApproverを適用。
各waypointを10 HzでG1のIK／物理制御へ渡す。以前の固定2秒補間はCLIのLLM経路で使わない。
低レベルのIK、物理、G1形状、カメラはこのリポジトリ固有。demoと直接のenv.moveは従来のAPIを維持。
終了時の追加0.75秒保持検証は評価専用で、その後にLLMを再度呼ばない。

検証:
- pytest: 22 passed、1 skipped（GUI用テスト）。
- テストで標準promptの利用、真値／診断情報の非送信、範囲外指令の再生成、
  実測開閉量、回転の往復変換、動作列のステップ数と実際のシミュレーション時間を確認。
- on-demandではtake_pic前に画像が送られず、要求後に送られることを確認。
- gpt-5.6-solを実際に呼び、move_toを2回実行。2waypoint=0.2秒、1waypoint=0.1秒。
  再観測に基づく補正を確認。2推論で打ち切ったため、目標到達・把持成功の検証ではない。
- Inspect eval経由も同モデルで動作実行・正常終了まで確認。
- 2x、EGO inset、字幕付きMP4の生成を確認（960x864、60 fps）。

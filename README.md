# g1-llm-policy

Mac上のMuJoCoで、LLMがハンド付きUnitree G1の手先を制御する実験用リポジトリ。

**GPT-6 Astra**
成功

https://github.com/user-attachments/assets/48747297-58a4-44a0-b079-981dc3033f0f

**GPT-5.6 Sol**
赤ブロックを落として失敗．興味深い点として赤ブロックが卓上にないことを認識してLLM自身がgive_upと終了している．

https://github.com/user-attachments/assets/f1a344be-7654-4670-b35a-1d8daa7fbd14

元の動画・入力画像・実行ログは[runs](runs/)にあります。


## 構成

Inspect標準の観測・プロンプト → Codex app-server（通信変換）→ Inspect標準のツール処理・動作補間 → 標準Controller／Approver → G1のMink IK・MuJoCo → 再観測。

- MenagerieのG1 Rev 1.0＋左右Dex3相当の3指ハンド。元モデルは43関節。
- 腕の初期姿勢は左右とも見た目で肘約90度、肩・手首0度（このMJCFでは肘の関節値も0）。ハンドは開いた状態。
- この環境では浮遊ベース・脚・腰を固定し、両腕14＋指14の28関節を駆動。
- ハンド外観は頭部と同じ既存の`black`材質。
- 頭部付近の`ego`をLLM入力の標準に使用。外部`overview`・`front`は任意。
- カメラはtorso固定、局所位置 `(0.065, 0, 0.41)` m、下向き40度、垂直画角80度の近似。実機の内部・外部パラメータではありません。
- Minkの関節角を`qpos`へコピーせず、位置アクチュエータで追従して`mj_step`を実行。駆動関節にはbias力補償を適用。
- 机・物体・指の接触は物理計算。物体を手に固定する処理はありません。
- LLM待ち中はシミュレーション時刻を停止します。標準agentが10 Hzの動作列を生成し、各ステップを0.1秒実行。実機の連続制御器ではありません。

## セットアップ

Python 3.12または3.13、uv、ログイン済みのCodex CLIが必要です。

```bash
uv sync --locked
uv run python scripts/setup_macos.py  # macOS / uv Pythonのmjpython対応
codex login
uv run python run.py models
```

## 起動

`uv run mjpython run.py run --viewer`で、全体ビューと右上のego小窓を同じウィンドウに表示します。
表示だけ試す場合は`uv run mjpython run.py view`です。
左ドラッグで回転、右ドラッグで移動、スクロールでズーム、`R`で初期画角に戻ります。
`E`でego小窓の表示切替、`Esc`で閉じます。LLMの応答待ち中も視点操作できます。
外観動画には右上のego小窓と、下部の動作説明・LLM待機表示を含めます。

LLMなしの物理追従テスト。右手を前方4cm・上方2cmに移動し、画像と誤差を保存します。

```bash
uv run python run.py demo
```

Macでネイティブビューアを開く場合は`mjpython`を使います。

```bash
uv run mjpython run.py view
uv run mjpython run.py demo --viewer
```

Astraでegoviewから制御。実際にCodexのモデル利用が発生します。

```bash
uv run python run.py run --model gpt-6-astra --max-steps 5 \
  --instruction '右手を初期位置から前に3cm、上に2cm動かし、到達誤差を確認して終了して。姿勢と左手は保って。'
```

`--effort low`が既定値。`--timeout 120`はapp-serverの待ち時間上限です。
`--cameras ego overview`で外部画像も追加できますが、標準ではegoのみを送ります。
`--viewer`を付ける場合は上記コマンドの`python`を`mjpython`にします。

`runs/日時/`に入力画像、実行前後の状態、LLMの指令、app-server受信イベント、終了画像を保存。
上限に達した終了とモデルの`done`は区別します。`--task pickup`の物理検証結果は内部評価専用で、LLMへは返しません。

`run.py`はソースツリーから直接ロードする起動口です。Codexの実行環境でeditableの`.pth`が無効になる場合も使えます。
通常のインストールでは`g1-policy`エントリーポイントも利用できます。

## Inspect Robots経由

同じ環境・CodexポリシーをInspect Robotsの評価・記録ループから呼び出せます。

```bash
uv run python examples/inspect_run.py --model gpt-6-astra --max-steps 100
```

登録名はembodiment=`g1_mujoco`、policy=`codex_g1`。
既存の実機用`g1_arms`は使いません。評価例は採点器なしの動作実験で、成功率を計算しません。

G1アダプタは16次元のCartesian契約です。左右それぞれ
`[x,y,z,qw,qx,qy,qz,open]`で、回転はworld座標での絶対姿勢の単位クォータニオンです。
実機プラグインの関節空間とは異なります。観測・プロンプト・ツール処理は標準agent本体を直接利用します。

## 指令と観測

- LLM入力は標準の`Observation`。`eef_state`に実測TCP位置・クォータニオン・実測開閉量、画像にegoを渡します。
- 位置は両手ともworld座標、m。+x前、+y左、+z上。dex1のTCPは顎間の中心です。
- 回転変更時はその手の`qw,qx,qy,qz`を4成分すべて指定し、姿勢保持時は4成分すべて省略します。ノルム誤差2%以内は正規化し、それ以外は再生成を求めます。
- 開閉は0=閉、1=開。観測は指令値ではなく指関節の実測値から換算します。
- 標準`move_to`が部分的な`targets`を受け取り、省略した次元は現在の実測値を保持します。
- `max_speed_frac=0.1`、各ステップの上限は範囲の5%、1ツール呼び出しは最大10秒。位置・開閉は標準Toolsetと同じ速度式です。回転はSLERPで、既定の角速度は18度/秒。quat対応部分のみ拡張しています。
- `DefaultController`で動作列全体を実行し、quat対応の`QuaternionApprover`を通します。位置・開閉は従来の5%上限、回転は9度/ステップの上限です。
- `--images always`が標準設定。`--images on_demand`では標準`take_pic`で画像を要求できます。
- `--max-llm-calls 100`は画像要求や入力エラーの再生成を含む推論回数上限。`run.py --max-steps`は動作列単位の上限です。
- Inspect評価ループの`Task.max_steps`は0.1秒の制御ステップ数で、上のCLI上限とは異なります。
- 独自の追従誤差・目標履歴・把持真値はLLMへ渡しません。標準の制限処理による補正通知のみ観測に含めます。
- `agent-transcript.json`に標準agentの会話、`episode.jsonl`に内部の実測誤差・評価・動作列を記録します。

Codexとの違いは通信層です。標準agentのツール定義を渡し、ツール名と引数をJSONで受け取って標準処理へ戻します。
標準agentが選んだ会話履歴（画像履歴制限を含む）だけを各Codexセッションへ渡します。
標準system promptに、G1固有の座標・ハンド情報と英語指定、JSON通信の説明を追加しています。
MuJoCoの物理・Mink IK・描画はG1固有のembodiment実装です。別ロボットの低レベル制御器そのものの再現ではありません。

## コードの場所

| ファイル | 役割 |
|---|---|
| `model.py` | 元XMLを読み、固定胴体・黒いハンド・シーン・カメラを追加 |
| `env.py` | IK、位置制御、物理計算、描画、実測誤差 |
| `commands.py` | 内部のIK指令形式 |
| `inspect_contract.py` | 標準agent向けの観測・16D Cartesian契約 |
| `policies/inspect_agent.py` | 標準agent本体とCodex通信変換 |
| `policies/codex.py` | stdio JSON-RPC、画像送信、構造化出力、タイムアウト |
| `adapters/inspect_robots.py` | Inspect Robotsのembodiment/policy |
| `cli.py` | 単体デモとLLM閉ループ、ログ保存 |

## 検証

```bash
uv run pytest
uv run ruff check src scripts tests examples run.py
uv run ruff format --check src scripts tests examples run.py
```

テストはモデルの出典ハッシュ・黒いハンド・ego配置・左右の物理追従・指の物理動作・物体の接触・不正指令の拒否・Inspect接続・app-serverの画像/イベント処理を確認します。
描画と実際のCodex通信は上記の`demo` / `run` / `inspect_run.py`で検証します。
Macで`invalid CoreGraphics connection`となる場合は通常のログイン済みGUIセッションで実行してください。

現在の範囲は胴体固定の手先制御。自律的な把持成功、歩行、実機移行は未検証です。

## モデルの出典

`src/g1_llm_policy/assets/g1/SOURCE.json`にMenagerieのコミットと全同梱ファイルのSHA-256を記録。
BSD-3-Clauseライセンスを同梱し、元XMLは編集していません。
取得処理は`python scripts/fetch_model.py`で再現可能です。

- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/ac6b2b09983786f3036cab1000221017fa2193b4/unitree_g1)
- [Mink](https://github.com/kevinzakka/mink)
- [Inspect Robots](https://github.com/robocurve/inspect-robots)

## 把持の終了判定

`--task auto`（既定）は指示中の`pick up` / `lift` / 「持ち上げ」「掴む」などをpickupとして扱います。
曖昧な指示では`--task pickup`で明示してください。通常の手先移動は`--task none`で従来の終了動作になります。

pickupの成功条件は、ブロックの**最下点が机から2cm以上離れ**、同じ手の異なる指／手のひらの2領域以上に
0.01N超の接触力があり、物体の並進速度が0.15m/s以下の状態が**連続0.5秒**続くことです。
一瞬の浮き上がりや投げ上げ、机上で指を閉じただけでは成功になりません。
`done`時には現在のアクチュエータ目標を保ち、追加0.75秒の物理計算で保持を再確認します。

LLMには画像・実測TCP・実測開閉量を渡し、把持評価の真値は渡しません。
接触領域・物体速度・クリアランス・保持時間・成功判定は評価ログ専用です。
`done`は評価結果にかかわらず試行を終了します。内部評価で成功なら`success`、失敗なら`failed`として記録し、
失敗理由をLLMに返して再試行させることはありません。CLIとInspect Robotsの両方に適用します。
動画字幕・コンソールの物理評価は人間向けで、ポリシー入力には使いません。

以前の`task_feedback`付き実行には評価用真値が含まれていました。修正後の実行とは分けて比較してください。

## 操作対象

既定のハンドは黒いDex1-1（二爪）です。`--hand dex3`でDex3に切り替えられます。
実行例・モデル出典・検証範囲は[Dex1-1](docs/dex1.md)を参照してください。

現在は赤いブロック（一辺5cm）です。指示は`pick up the red block`を使えます。

## 動画保存

`run`と`demo`は、物理シミュレーション中のフレームから既定で30 fpsの動画を保存します。
従来のコマンドで、出力フォルダに`video-overview.mp4`（外観）と`video-ego.mp4`（一人称）が追加されます。
LLMの応答待ちも実測した待ち時間分の静止映像として含め、外観動画に`Waiting for LLM`と表示します。
待機中の物理シミュレーションは停止したままです。動作・応答待ちの両方を既定で2倍速で再生します。
`--video-speed 1`で等速に戻せます。シミュレーション1秒につき30フレームを収録し、2倍速時は60 fpsで再生します。
最後の保持確認も記録し、Ctrl+Cで中断した場合も記録済み部分をMP4として確定します。

`ffmpeg`が必要です（macOS: `brew install ffmpeg`）。録画不要なら`--no-video`、
視点の指定は`--video-cameras overview`、フレームレートは`--video-fps 60`（1〜100）を使えます。
録画視点はLLMに渡す`--cameras`とは独立しています。`view`モードは録画しません。

外観動画`video-overview.mp4`はviewerと同じカメラ位置・向き・ズームで記録します。
外観部分は4:3（960×720）に固定し、ウィンドウのリサイズでは変わりません。
左上に再生速度（`1x`、`2x`など）、右上に一人称映像、下部に現在の`[step] status: note`を表示します（全体960×864）。
動作・応答待ちは既定で2倍速、終了時は最後の説明を読むため1.5秒静止します。
viewerなしの場合はviewerの初期視点を使用します。一人称の単独動画も保存します。

# Dex1-1

既定は黒いDex1-1平行二爪グリッパです。`--hand dex3`でDex3に切り替えられます。

```sh
uv run mjpython run.py view --hand dex1
uv run mjpython run.py run --viewer --hand dex1 --model gpt-6-astra --max-steps 50 --instruction "pick up the red block"
```

Unitree公式`unitree_ros`の`g1_29dof_mode_15_with_dex1_1.urdf`から、左右の5010手首とDex1-1のサブツリーを取り込みます。胴体・腕の残りは従来のMenagerieモデルです。mode15全身の完全な再現ではありません。
同梱元のコミットとSHA-256は`src/g1_llm_policy/assets/dex1/SOURCE.json`に記録し、ライセンスを同梱しています。`python scripts/fetch_dex1.py`で再取得できます。

公式の形状・慣性・関節軸・可動範囲・力制限を使い、シミュレーション用の位置サーボと爪の同期拘束を追加しています。サーボゲイン・摩擦・同期拘束は本環境の設定で、実機同定値ではありません。片手2スライド関節を1つの開度で動かします。`0`が閉、`1`が開で、爪間隔は約5.8–94.8 mmです。

TCPは手首yaw座標の`(0.155, 0, 0)`、爪の接触面の間です。ローカルxが爪の長手方向、yが開閉方向です。ポリシー観測の`gripper_model`と`tcp_wrist_local_m`にも反映します。Inspect Robotsでは`G1Embodiment(hand="dex1")`を使います。

## 検証

`tests/test_dex1.py`で出典ファイルのハッシュ、手首寸法・力制限、左右独立の開閉、物理接触による箱の把持・持ち上げ・解放を確認します。把持判定は両爪の接触を別領域として数えます。

把持テストでは5 cm角の箱を初期右手の直下の机上に配置し、スクリプトで接近・閉・7 cm上昇を指定します。約5.7 cmの物体底面クリアランスで保持できました。これは把持機構の検証であり、LLMが既定シーンから自律把持できたという結果ではありません。

# Debate Duel — AWS CDK デプロイ

ECR リポジトリ・ECS タスク定義・ALB 付き Fargate サービスをデプロイします。

## 前提条件

1. **AWS CLI** の設定（`aws configure`）
2. **CDK Bootstrap**（初回のみ）
   ```bash
   cdk bootstrap
   ```
3. **Secrets Manager** に GOOGLE_API_KEY を登録
   ```bash
   aws secretsmanager create-secret \
     --name debate-duel/google-api-key \
     --secret-string "YOUR_GOOGLE_API_KEY"
   ```
4. **ACM 証明書**（カスタムドメイン用の場合、`cdk.json` の context で `certificate_arn` を指定）

## デプロイ

```bash
# CDK 依存関係のインストール
pip install -r cdk/requirements.txt

# デプロイ（プロジェクトルート＝DebateDuel/ で実行）
cdk deploy DebateDuelEcrStack
```

## context オプション

`cdk.json` または `-c` で指定可能：

| キー | 説明 | デフォルト |
|------|------|------------|
| `account` | AWS アカウント ID | 設定済み default |
| `region` | リージョン | 設定済み default |
| `certificate_arn` | ACM 証明書 ARN | StarLight 共通証明書 |
| `google_api_key_secret` | Secrets Manager シークレット名 | `debate-duel/google-api-key` |

例：
```bash
cdk deploy DebateDuelEcrStack -c google_api_key_secret=my-custom/secret-name
```

## 出力

デプロイ完了後、CloudFormation の Output から以下を確認できます：

- **LoadBalancerUrl**: HTTPS の ALB URL（数分後にアクセス可能）
- **ImageUri**: ECR イメージ URI
- **RepositoryUri**: ECR リポジトリ URI

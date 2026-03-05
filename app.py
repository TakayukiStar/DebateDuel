#!/usr/bin/env python3
"""
Debate Duel — AWS CDK Entry Point

ECR への Docker イメージ配置と ECS サービスをデプロイします。
"""

import aws_cdk as cdk

from cdk.debate_duel_ecr_stack import DebateDuelEcrStack

app = cdk.App()

DebateDuelEcrStack(
    app,
    "DebateDuelEcrStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account") or None,
        region=app.node.try_get_context("region") or None,
    ),
    description="Debate Duel — ECR リポジトリと ECS サービス",
)

app.synth()

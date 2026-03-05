"""
Debate Duel — ECR & ECS スタック

DockerイメージをビルドしてECRに配置し、ECSタスク定義・サービスを作成します。
"""

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

# ACM証明書ARN（ap-northeast-1）※環境に応じて変更してください
CERTIFICATE_ARN = "arn:aws:acm:ap-northeast-1:288761754727:certificate/0648d287-d430-416b-b5ae-5a6b66b0e593"


class DebateDuelEcrStack(cdk.Stack):
    """ECRリポジトリ、Dockerイメージ、ECSタスク定義・サービス"""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # プロジェクトルート（Dockerfileの場所）
        project_root = Path(__file__).resolve().parent.parent

        # ECRリポジトリを作成
        repository = ecr.Repository(
            self,
            "DebateDuelRepository",
            repository_name="debate-duel",
            removal_policy=cdk.RemovalPolicy.RETAIN,
            empty_on_delete=False,
        )

        # DockerイメージをビルドしてECRにプッシュ
        docker_image = ecr_assets.DockerImageAsset(
            self,
            "DebateDuelImage",
            directory=str(project_root),
            exclude=[
                "cdk.out",
                "cdk",
                "node_modules",
                ".git",
                "*.md",
                "__pycache__",
            ],
            asset_name="debate-duel",
        )

        # VPC
        vpc = ec2.Vpc(
            self,
            "DebateDuelVpc",
            max_azs=2,
            nat_gateways=1,
        )

        # ECSクラスター
        cluster = ecs.Cluster(
            self,
            "DebateDuelCluster",
            cluster_name="debate-duel-cluster",
            vpc=vpc,
        )

        # タスク定義（Fargate）
        # faster-whisper 等のためメモリを多めに
        task_definition = ecs.FargateTaskDefinition(
            self,
            "DebateDuelTaskDef",
            family="debate-duel",
            cpu=2048,
            memory_limit_mib=4096,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.X86_64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )

        # コンテナ定義
        container = task_definition.add_container(
            "DebateDuelContainer",
            image=ecs.ContainerImage.from_docker_image_asset(docker_image),
            container_name="debate-duel",
            port_mappings=[
                ecs.PortMapping(
                    container_port=8000,
                    protocol=ecs.Protocol.TCP,
                )
            ],
            environment={
                "WHISPER_MODEL": "small",
                "WHISPER_DEVICE": "cpu",
                "GEMINI_MODEL": "gemini-2.5-flash",
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="debate-duel",
            ),
        )

        # ACM証明書をインポート
        certificate = acm.Certificate.from_certificate_arn(
            self,
            "DebateDuelCertificate",
            certificate_arn=self.node.try_get_context("certificate_arn") or CERTIFICATE_ARN,
        )

        # Application Load Balancer 付き Fargate サービス
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "DebateDuelService",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=1,
            service_name="debate-duel",
            certificate=certificate,
            listener_port=443,
            redirect_http=True,
            public_load_balancer=True,
        )

        # ヘルスチェックの調整（ALB）
        fargate_service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=cdk.Duration.seconds(30),
            timeout=cdk.Duration.seconds(10),
        )

        # 出力
        cdk.CfnOutput(
            self,
            "ImageUri",
            value=docker_image.image_uri,
            description="ECR イメージ URI",
            export_name="DebateDuelImageUri",
        )

        cdk.CfnOutput(
            self,
            "RepositoryUri",
            value=repository.repository_uri,
            description="ECR リポジトリ URI",
            export_name="DebateDuelRepositoryUri",
        )

        cdk.CfnOutput(
            self,
            "LoadBalancerUrl",
            value=f"https://{fargate_service.load_balancer.load_balancer_dns_name}",
            description="ALB URL（HTTPS、数分後にアクセス可能）",
            export_name="DebateDuelLoadBalancerUrl",
        )

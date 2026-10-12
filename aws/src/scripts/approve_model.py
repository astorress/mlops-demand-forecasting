"""Model Registry desde la terminal: lista versiones con sus métricas y
aprueba/rechaza una (el gate de gobierno con una persona en el loop).

    python3 -m src.scripts.approve_model                # listar
    python3 -m src.scripts.approve_model --approve 3    # aprobar la versión 3
    python3 -m src.scripts.approve_model --reject 4 --description "MAPE alto"

Aprobar dispara la regla de EventBridge → Lambda OnModelApproved → contrato
de despliegue en SSM + notificación.
"""
import argparse

from src.services import s3, sagemaker

DEFAULT_GROUP = "demand-forecasting"


def read_metrics(model_package_arn: str) -> tuple[str, str]:
    package = sagemaker.describe_model_package(model_package_arn)
    quality = package.get("ModelMetrics", {}).get("ModelQuality", {})
    uri = quality.get("Statistics", {}).get("S3Uri")
    if not uri:
        return "-", "-"
    metrics = s3.read_json(uri)["regression_metrics"]
    return (f"{metrics['mape']['value']:.3f}",
            f"{metrics['rmse']['value']:.2f}")


def list_versions(group: str) -> list:
    return sagemaker.list_model_packages(group)


def find_arn(group: str, version: int) -> str:
    for summary in list_versions(group):
        if summary["ModelPackageVersion"] == version:
            return summary["ModelPackageArn"]
    raise SystemExit(f"Version {version} not found in {group}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default=DEFAULT_GROUP)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--approve", type=int, metavar="VERSION")
    action.add_argument("--reject", type=int, metavar="VERSION")
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    if args.approve or args.reject:
        version = args.approve or args.reject
        status = "Approved" if args.approve else "Rejected"
        sagemaker.update_approval_status(
            find_arn(args.group, version),
            status,
            args.description or f"{status} from CLI",
        )
        print(f"Version {version} → {status}")
        return

    print(f"{'VERSION':<8} {'STATUS':<24} {'MAPE %':<8} {'RMSE':<8} CREATED")
    for summary in list_versions(args.group):
        mape, rmse = read_metrics(summary["ModelPackageArn"])
        print(f"{summary['ModelPackageVersion']:<8} "
              f"{summary['ModelApprovalStatus']:<24} "
              f"{mape:<8} {rmse:<8} "
              f"{summary['CreationTime']:%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()

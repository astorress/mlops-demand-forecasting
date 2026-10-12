import boto3

CLIENT = boto3.client("sagemaker")


def describe_model_package(model_package_arn: str) -> dict:
    return CLIENT.describe_model_package(ModelPackageName=model_package_arn)


def list_model_packages(group_name: str, max_results: int = 20) -> list:
    return CLIENT.list_model_packages(
        ModelPackageGroupName=group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=max_results,
    )["ModelPackageSummaryList"]


def update_approval_status(
    model_package_arn: str,
    status: str,
    description: str,
) -> None:
    CLIENT.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus=status,
        ApprovalDescription=description,
    )


def create_model_from_package(
    model_name: str,
    role_arn: str,
    model_package_arn: str,
) -> None:
    CLIENT.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        Containers=[{
            "ModelPackageName": model_package_arn,
        }],
    )


def create_transform_job(**request) -> None:
    CLIENT.create_transform_job(**request)

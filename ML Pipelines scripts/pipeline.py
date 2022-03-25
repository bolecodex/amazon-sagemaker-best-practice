# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Example workflow pipeline script for CustomerChurn pipeline.
                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)
Implements a get_pipeline(**kwargs) method.
"""

import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
)
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel

# os.path.dirname 返回父目录层级 ： os.path.dirname('W:\Python_File') -> W:\
# os.path.realpath 返回的是使用软链的真实地址: 假定： ln -s /home/a/1.txt /home/b/1.txt,  os.path.realpath("b/1.txt")   ->  '/root/a/1.txt'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """

    # 创建一个 Session对象
    boto_session = boto3.Session(region_name=region)

    # 创建 sagemaker_client 对象
    sagemaker_client = boto_session.client("sagemaker")
    # 创建 runtime_client 对象
    runtime_client = boto_session.client("sagemaker-runtime")

    # 返回对应的配置信息
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="CustomerChurnPackageGroup",  # Choose any name
    pipeline_name="CustomerChurnDemo-p-ewf8t7lvhivm",  # You can find your pipeline name in the Studio UI (project -> Pipelines -> name)
    base_job_prefix="CustomerChurn",  # Choose any name
):
    """Gets a SageMaker ML Pipeline instance working with on CustomerChurn data.
    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        an instance of a pipeline
    """
    # 调用 get_session，获取对应信息
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # Parameters for pipeline execution
    # 设置 pipeline 的运行参数
    # 设置实例数量
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    # 设置实例类型
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    # 设置 model_approval 状态： 有如下两个值：“Approved”, “Rejected”
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )
    # 设置 input_data
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://YOUR-BUCKET/sagemaker/DEMO-xgboost-churn/data/RawData.csv",  # Change this to point to the s3 location of your raw input data.
    )

    # Processing step for feature engineering
    # 创建一个 SKLearn的处理任务
    sklearn_processor = SKLearnProcessor(
        # 指定版本
        framework_version="0.23-1",
        # 指定实例类型
        instance_type=processing_instance_type,
        # 指定实例数
        instance_count=processing_instance_count,
        # 指定名字， 可任意
        base_job_name=f"{base_job_prefix}/sklearn-CustomerChurn-preprocess",  # choose any name
        # 指定 sagemaker_session 对象
        sagemaker_session=sagemaker_session,
        # 赋予role权限
        role=role,
    )
    # 定义 ProcessingStep
    step_process = ProcessingStep(
        # 指定 Process 的名字
        name="CustomerChurnProcess",  # choose any name
        processor=sklearn_processor,
        # 定义输出 train， validation， test： output_name为输出的名字， source为输出信息的来源
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(
                output_name="validation", source="/opt/ml/processing/validation"
            ),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        # 指定代码
        code=os.path.join(BASE_DIR, "preprocess.py"),
        # 指定程序运行 input_data 参数
        job_arguments=["--input-data", input_data],
    )

    # Training step for generating model artifacts
    # 定义模型的path
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/CustomerChurnTrain"
    # 检索与给定参数匹配的 Docker 映像的 ECR URI
    # 参考： https://sagemaker.readthedocs.io/en/stable/api/utility/image_uris.html?highlight=image_uris.retrieve#sagemaker.image_uris.retrieve
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",  # we are using the Sagemaker built in xgboost algorithm
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    # 创建一个 Estimator 实例用来进行 train
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/CustomerChurn-train",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    # 设置超参：
    """
        objective="binary:logistic", 针对二分类的逻辑回归问题，输出为概率
        num_round=50, 弱评估器数量, 数量越多，模型越容易过拟合，数量过少，模型可能欠拟合
        max_depth=5, 数的最大深度
        eta=0.2, 学习率
        gamma=4,  指定了节点分裂所需的最小损失函数下降值。这个参数的值越大，算法越保守
        min_child_weight=6, 决定最小叶子节点样本权重和,这个参数可以避免过拟合
        subsample=0.7, 训练模型的子样本占整个样本集合的比例。
        silent=0,  0表示打印出运行时信息
    """
    xgb_train.set_hyperparameters(
        objective="binary:logistic",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )
    # 创建 TrainingStep 实例：
    step_train = TrainingStep(
        # 设置名字
        name="CustomerChurnTrain",
        # 设置 estimator 实例
        estimator=xgb_train,
        # 定义 inputs
        inputs={
            "train": TrainingInput(
                # 设定数据路径uri
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                # 设置 content_type
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # Processing step for evaluation
    # 创建一个 ScriptProcessor 实例
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-CustomerChurn-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    # 定义一个 PropertyFile： 用来提供一个属性文件结构。
    # 参考： https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html?highlight=PropertyFile#sagemaker.workflow.properties.PropertyFile
    evaluation_report = PropertyFile(
        name="EvaluationReport", # 用于JsonGet函数参考的属性文件的名称
        output_name="evaluation",
        path="evaluation.json",
    )
    # 定义 val的ProcessingStep
    step_eval = ProcessingStep(
        name="CustomerChurnEval",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source="/opt/ml/processing/evaluation"
            ),
        ],
        # 指定运行脚本
        code=os.path.join(BASE_DIR, "evaluate.py"),
        # 指定 evaluation_report 属性文件
        property_files=[evaluation_report],
    )

    # Register model step that will be conditionally executed
    # ModelMetrics： 接受模型指标参数转换为字典， 参考： https://sagemaker.readthedocs.io/en/stable/api/inference/model_monitor.html?highlight=ModelMetrics#sagemaker.model_metrics.ModelMetrics
    # MetricsSource： 将参数转成dict， 参考： https://sagemaker.readthedocs.io/en/stable/api/inference/model_monitor.html?highlight=ModelMetrics#sagemaker.model_metrics.MetricsSource
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                    "S3Uri"
                ]
            ),
            content_type="application/json",
        )
    )

    # Register model step that will be conditionally executed
    # RegisterModel为工作流注册模型步骤集合。
    # 参考： https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html?highlight=RegisterModel#sagemaker.workflow.step_collections.RegisterModel
    step_register = RegisterModel(
        name="CustomerChurnRegisterModel",
        estimator=xgb_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # Condition step for evaluating model quality and branching execution
    # ConditionGreaterThanOrEqualTo的构造，用于大于或等于比较。
    # 参数：在比较中使用的执行变量、参数、属性或 Python 原始值。
    # 参考： https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html?highlight=ConditionGreaterThanOrEqualTo#sagemaker.workflow.conditions.ConditionGreaterThanOrEqualTo
    cond_lte = ConditionGreaterThanOrEqualTo(  # You can change the condition here
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
        ),
        right=0.8,  # You can change the threshold here
    )
    # 定义 管道的条件步骤：
    # 参考： https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html?highlight=ConditionStep#conditionstep
    # name： 指定名称 
    # conditions： 实例的列表。
    # if_steps： 参数为StepCollection实例的列表， 如果条件列表评估为 True，则标记为准备好执行。
    # else_steps： 参数为StepCollection实例的列表，如果条件列表评估为 False，则标记为准备执行。
    step_cond = ConditionStep(
        name="CustomerChurnAccuracyCond",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # Pipeline instance
    # 创建一个 pipeline 实例对象
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline

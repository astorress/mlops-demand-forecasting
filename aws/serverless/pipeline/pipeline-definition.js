// Definición del SageMaker Pipeline (schema 2020-12-01) escrita como JSON
// plano, así cada paso se ve como código en vez de generarse con el SDK. Es
// el equivalente del DAG mlops_full_pipeline de Airflow del proyecto open
// source.
//
//   PrepareData ─┬─> Baseline ─────────────────────────────────┐
//   (Athena →    └─> Train ──> Evaluate ──> CheckMape ──┬─> RegisterModel
//    validar/split temporal)                            └─> FailHighMape
//
// Referencia (tipos de paso y sus argumentos):
//   https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html
const { imageVersion } = require('../../container/image-version')

const PROCESSING = '/opt/ml/processing'

// Std:Join → s3://<bucket>/<prefijo>/<execution id>/<...partes>
const executionUri = (base, ...parts) => ({
  'Std:Join': {
    On: '/',
    Values: [base, { Get: 'Execution.PipelineExecutionId' }, ...parts],
  },
})

const s3Input = (name, uri, localPath) => ({
  InputName: name,
  AppManaged: false,
  S3Input: {
    S3Uri: uri,
    LocalPath: localPath,
    S3DataType: 'S3Prefix',
    S3InputMode: 'File',
    S3DataDistributionType: 'FullyReplicated',
    S3CompressionType: 'None',
  },
})

const s3Output = (name, uri, localPath) => ({
  OutputName: name,
  AppManaged: false,
  S3Output: { S3Uri: uri, LocalPath: localPath, S3UploadMode: 'EndOfJob' },
})

const outputUri = (step, name) => ({
  Get: `Steps.${step}.ProcessingOutputConfig.Outputs['${name}'].S3Output.S3Uri`,
})

const cluster = (instanceType) => ({
  ClusterConfig: {
    InstanceType: instanceType,
    InstanceCount: 1,
    VolumeSizeInGB: 30,
  },
})

module.exports = async ({ resolveVariable }) => {
  const custom = (key) => resolveVariable(`self:custom.${key}`)
  const [
    accountId,
    roleName,
    artifactsBucket,
    projectPrefix,
    repositoryUri,
    database,
    curatedTable,
    athenaWorkGroup,
    modelPackageGroupName,
    maxMape,
    realtimeInstanceTypes,
    transformInstanceTypes,
  ] = await Promise.all([
    resolveVariable('aws:accountId'),
    custom('iam.sageMakerRoleName'),
    custom('s3.artifactsBucketName'),
    custom('prefixes.pipeline'),
    custom('image.repositoryUri'),
    custom('glue.databaseName'),
    custom('glue.curatedTableName'),
    custom('athena.pipelineWorkGroupName'),
    custom('registry.modelPackageGroupName'),
    custom('pipeline.maxMape'),
    custom('registry.realtimeInstanceTypes'),
    custom('registry.transformInstanceTypes'),
  ])

  const roleArn = `arn:aws:iam::${accountId}:role/${roleName}`
  // Una imagen inmutable (tag = hash del contenido) para todos los pasos.
  const imageTag = imageVersion()
  const modelImage = `${repositoryUri}:${imageTag}`
  const runBase = `s3://${artifactsBucket}/${projectPrefix}/executions`
  const evaluationUri = executionUri(runBase, 'evaluation', 'evaluation.json')
  const statisticsUri = executionUri(runBase, 'baseline', 'statistics.json')
  const constraintsUri = executionUri(runBase, 'baseline', 'constraints.json')
  const modelArtifacts = { Get: 'Steps.Train.ModelArtifacts.S3ModelArtifacts' }
  const script = (module) => [
    'python3',
    '-m',
    `src.pipelines.training.${module}`,
  ]

  const curatedQuery =
    `SELECT * FROM "${database}"."${curatedTable}" ` +
    `WHERE ingestion_date = (SELECT max(ingestion_date) ` +
    `FROM "${database}"."${curatedTable}")`

  // 1. Preparación de datos: SageMaker corre la consulta de Athena
  //    (dataset definition) y preprocess.py valida y parte en el tiempo.
  const prepareData = {
    Name: 'PrepareData',
    Type: 'Processing',
    Description: 'Athena (tabla curada) → validación → split temporal',
    Arguments: {
      ProcessingResources: cluster({
        Get: 'Parameters.ProcessingInstanceType',
      }),
      AppSpecification: {
        ImageUri: modelImage,
        ContainerEntrypoint: script('preprocess'),
        ContainerArguments: [
          '--min-rows',
          { Get: 'Parameters.MinRows' },
          '--test-days',
          { Get: 'Parameters.TestDays' },
          '--validation-days',
          { Get: 'Parameters.ValidationDays' },
        ],
      },
      RoleArn: roleArn,
      ProcessingInputs: [
        {
          InputName: 'curated',
          AppManaged: false,
          DatasetDefinition: {
            AthenaDatasetDefinition: {
              Catalog: 'AwsDataCatalog',
              Database: database,
              QueryString: curatedQuery,
              WorkGroup: athenaWorkGroup,
              OutputS3Uri: `s3://${artifactsBucket}/${projectPrefix}/athena/`,
              OutputFormat: 'JSON',
            },
            LocalPath: `${PROCESSING}/input/data`,
            DataDistributionType: 'FullyReplicated',
            InputMode: 'File',
          },
        },
      ],
      ProcessingOutputConfig: {
        Outputs: ['train', 'validation', 'test', 'baseline'].map((name) =>
          s3Output(
            name,
            executionUri(runBase, 'data', name),
            `${PROCESSING}/${name}`
          )
        ),
      },
      StoppingCondition: { MaxRuntimeInSeconds: 1800 },
    },
  }

  // 2. Baseline de drift: perfil de referencia de los datos de entrenamiento
  //    (bins y frecuencias por feature) + umbrales. Se registra como el
  //    baseline de drift del modelo; la Lambda DataDrift compara el tráfico
  //    en vivo contra él.
  const baseline = {
    Name: 'Baseline',
    Type: 'Processing',
    Description: 'Perfil de referencia para data drift (PSI)',
    Arguments: {
      ProcessingResources: cluster({
        Get: 'Parameters.ProcessingInstanceType',
      }),
      AppSpecification: {
        ImageUri: modelImage,
        ContainerEntrypoint: script('baseline'),
      },
      RoleArn: roleArn,
      ProcessingInputs: [
        s3Input(
          'baseline_dataset_input',
          outputUri('PrepareData', 'baseline'),
          `${PROCESSING}/input/baseline_dataset_input`
        ),
      ],
      ProcessingOutputConfig: {
        Outputs: [
          s3Output(
            'monitoring_output',
            executionUri(runBase, 'baseline'),
            `${PROCESSING}/output`
          ),
        ],
      },
      StoppingCondition: { MaxRuntimeInSeconds: 1800 },
    },
  }

  // 3. Entrenamiento: SageMaker corre `<imagen> train` (LightGBM con early
  //    stopping sobre el canal validation). Las métricas impresas las
  //    capturan las MetricDefinitions (Experiments / CloudWatch).
  const train = {
    Name: 'Train',
    Type: 'Training',
    Description: 'Pipeline de scikit-learn (features + LightGBM)',
    Arguments: {
      AlgorithmSpecification: {
        TrainingImage: modelImage,
        TrainingInputMode: 'File',
        MetricDefinitions: ['rmse', 'mae', 'mape', 'r2'].map((metric) => ({
          Name: `validation:${metric}`,
          Regex: `validation:${metric}=(-?[0-9\\.]+)`,
        })),
      },
      HyperParameters: {
        n_estimators: { Get: 'Parameters.NEstimators' },
        learning_rate: { Get: 'Parameters.LearningRate' },
        num_leaves: { Get: 'Parameters.NumLeaves' },
      },
      RoleArn: roleArn,
      InputDataConfig: ['train', 'validation'].map((name) => ({
        ChannelName: name,
        ContentType: 'text/csv',
        DataSource: {
          S3DataSource: {
            S3DataType: 'S3Prefix',
            S3Uri: outputUri('PrepareData', name),
            S3DataDistributionType: 'FullyReplicated',
          },
        },
      })),
      OutputDataConfig: { S3OutputPath: executionUri(runBase, 'model') },
      ResourceConfig: {
        InstanceCount: 1,
        InstanceType: { Get: 'Parameters.TrainingInstanceType' },
        VolumeSizeInGB: 30,
      },
      StoppingCondition: { MaxRuntimeInSeconds: 3600 },
    },
  }

  // 4. Evaluación offline en el split de test (los últimos meses) →
  //    evaluation.json, expuesto al pipeline como PropertyFile.
  const evaluate = {
    Name: 'Evaluate',
    Type: 'Processing',
    Description: 'Métricas del split de test (formato del Model Registry)',
    Arguments: {
      ProcessingResources: cluster({
        Get: 'Parameters.ProcessingInstanceType',
      }),
      AppSpecification: {
        ImageUri: modelImage,
        ContainerEntrypoint: script('evaluate'),
      },
      RoleArn: roleArn,
      ProcessingInputs: [
        s3Input('model', modelArtifacts, `${PROCESSING}/model`),
        s3Input('test', outputUri('PrepareData', 'test'), `${PROCESSING}/test`),
      ],
      ProcessingOutputConfig: {
        Outputs: [
          s3Output(
            'evaluation',
            executionUri(runBase, 'evaluation'),
            `${PROCESSING}/evaluation`
          ),
        ],
      },
      StoppingCondition: { MaxRuntimeInSeconds: 1800 },
    },
    PropertyFiles: [
      {
        PropertyFileName: 'EvaluationReport',
        OutputName: 'evaluation',
        FilePath: 'evaluation.json',
      },
    ],
  }

  // 6a. Registra una versión nueva del model package (Model Registry). Queda
  //     en PendingManualApproval hasta que una persona la apruebe.
  const registerModel = {
    Name: 'RegisterModel',
    Type: 'RegisterModel',
    Description: 'Versión nueva del model package (pendiente de aprobación)',
    DependsOn: ['Baseline'],
    Arguments: {
      ModelPackageGroupName: modelPackageGroupName,
      ModelPackageDescription: `image ${modelImage}`,
      ModelApprovalStatus: { Get: 'Parameters.ModelApprovalStatus' },
      InferenceSpecification: {
        Containers: [
          {
            // Endpoints y Batch Transform corren `<imagen> serve`.
            Image: modelImage,
            ModelDataUrl: modelArtifacts,
          },
        ],
        SupportedContentTypes: ['text/csv', 'application/json'],
        SupportedResponseMIMETypes: ['application/json', 'text/csv'],
        SupportedRealtimeInferenceInstanceTypes: realtimeInstanceTypes,
        SupportedTransformInstanceTypes: transformInstanceTypes,
      },
      ModelMetrics: {
        ModelQuality: {
          Statistics: { ContentType: 'application/json', S3Uri: evaluationUri },
        },
        ModelDataQuality: {
          Statistics: { ContentType: 'application/json', S3Uri: statisticsUri },
          Constraints: {
            ContentType: 'application/json',
            S3Uri: constraintsUri,
          },
        },
      },
      DriftCheckBaselines: {
        ModelDataQuality: {
          Statistics: { ContentType: 'application/json', S3Uri: statisticsUri },
          Constraints: {
            ContentType: 'application/json',
            S3Uri: constraintsUri,
          },
        },
      },
      CustomerMetadataProperties: {
        ImageTag: imageTag,
        TriggeredBy: { Get: 'Parameters.TriggeredBy' },
      },
    },
  }

  // 6b. Quality gate fallido: termina con un error explícito (sin registro).
  const failHighMape = {
    Name: 'FailHighMape',
    Type: 'Fail',
    Arguments: {
      ErrorMessage: {
        'Std:Join': {
          On: ' ',
          Values: [
            'Test MAPE above the maximum',
            { Get: 'Parameters.MaxMape' },
          ],
        },
      },
    },
  }

  // 5. Quality gate: solo se registran modelos con MAPE de test aceptable.
  const checkMape = {
    Name: 'CheckMape',
    Type: 'Condition',
    Arguments: {
      Conditions: [
        {
          Type: 'LessThanOrEqualTo',
          LeftValue: {
            'Std:JsonGet': {
              PropertyFile: {
                Get: 'Steps.Evaluate.PropertyFiles.EvaluationReport',
              },
              Path: 'regression_metrics.mape.value',
            },
          },
          RightValue: { Get: 'Parameters.MaxMape' },
        },
      ],
      IfSteps: [registerModel],
      ElseSteps: [failHighMape],
    },
  }

  const definition = {
    Version: '2020-12-01',
    Metadata: {},
    Parameters: [
      {
        Name: 'ProcessingInstanceType',
        Type: 'String',
        DefaultValue: 'ml.m5.large',
      },
      {
        Name: 'TrainingInstanceType',
        Type: 'String',
        DefaultValue: 'ml.m5.large',
      },
      { Name: 'MinRows', Type: 'String', DefaultValue: '1000' },
      { Name: 'TestDays', Type: 'String', DefaultValue: '184' },
      { Name: 'ValidationDays', Type: 'String', DefaultValue: '182' },
      { Name: 'NEstimators', Type: 'String', DefaultValue: '500' },
      { Name: 'LearningRate', Type: 'String', DefaultValue: '0.05' },
      { Name: 'NumLeaves', Type: 'String', DefaultValue: '63' },
      { Name: 'MaxMape', Type: 'Float', DefaultValue: Number(maxMape) },
      {
        Name: 'ModelApprovalStatus',
        Type: 'String',
        DefaultValue: 'PendingManualApproval',
      },
      { Name: 'TriggeredBy', Type: 'String', DefaultValue: 'manual' },
    ],
    PipelineExperimentConfig: {
      ExperimentName: { Get: 'Execution.PipelineName' },
      TrialName: { Get: 'Execution.PipelineExecutionId' },
    },
    Steps: [prepareData, baseline, train, evaluate, checkMape],
  }

  return JSON.stringify(definition)
}

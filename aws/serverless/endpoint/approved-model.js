// Guarda del contrato de despliegue: el endpoint solo se construye desde un
// model package que una persona aprobó. OnModelApproved escribe estos
// parámetros SSM al aprobar; antes valen "none", que CloudFormation rechazaría
// con un error de validación poco claro. Mejor fallar temprano.
const PREFIX = '/mlops-demand-forecasting-pipeline'

const approved =
  (name, isValid) =>
  async ({ resolveVariable }) => {
    const value = await resolveVariable(`ssm:${PREFIX}/${name}`)
    if (!isValid(String(value))) {
      throw new Error(
        `Aún no hay modelo aprobado (${PREFIX}/${name} = "${value}"). ` +
          'Aprueba una versión primero: npm run models -- --approve <version>'
      )
    }
    return value
  }

module.exports.packageArn = approved('ApprovedModelPackageArn', (value) =>
  value.startsWith('arn:')
)
module.exports.baselineConstraintsUri = approved(
  'ApprovedBaselineConstraintsUri',
  (value) => value.startsWith('s3://')
)
module.exports.baselineStatisticsUri = approved(
  'ApprovedBaselineStatisticsUri',
  (value) => value.startsWith('s3://')
)

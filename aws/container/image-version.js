// Imágenes inmutables: el tag es un hash de todo lo que va dentro de la imagen
// (Dockerfile, entrypoint, requirements fijos y los paquetes del modelo en
// src/). Cada model package apunta a la imagen exacta que lo entrenó, y el
// repositorio ECR rechaza sobrescribir un tag.
//
// Como resolver `${file()}` de Serverless además verifica que el tag exista en
// ECR, así nunca se despliega un pipeline apuntando a una imagen que no se
// publicó (`npm run image:push`). SKIP_IMAGE_CHECK=true omite esa verificación.
const crypto = require('crypto')
const fs = require('fs')
const path = require('path')
const { execFileSync } = require('child_process')

const PROJECT_DIR = path.resolve(__dirname, '..')
// Paquetes copiados a la imagen (sincronizar con container/Dockerfile).
const IMAGE_PACKAGES = [
  'features',
  'models',
  'monitoring',
  'pipelines',
  'serving',
]
// build-push.sh se incluye porque decide el formato de manifiesto que recibe
// SageMaker: cambiar cómo se construye la imagen debe producir un tag nuevo.
const IMAGE_FILES = [
  'container/Dockerfile',
  'container/entrypoint.sh',
  'container/requirements.txt',
  'container/build-push.sh',
]

const imageVersion = () => {
  const packageFiles = IMAGE_PACKAGES.flatMap((name) =>
    fs
      .readdirSync(path.join(PROJECT_DIR, 'src', name), { recursive: true })
      .filter((file) => file.endsWith('.py'))
      .map((file) => path.join('src', name, file))
  ).sort()
  const hash = crypto.createHash('sha256')
  for (const file of [...IMAGE_FILES, 'src/__init__.py', ...packageFiles]) {
    hash.update(file)
    hash.update(fs.readFileSync(path.join(PROJECT_DIR, file)))
  }
  return hash.digest('hex').slice(0, 12)
}

const assertPushed = (repositoryName, tag) => {
  try {
    execFileSync(
      'aws',
      [
        'ecr',
        'describe-images',
        '--repository-name',
        repositoryName,
        '--image-ids',
        `imageTag=${tag}`,
      ],
      { stdio: 'ignore' }
    )
  } catch {
    throw new Error(
      `La imagen ${repositoryName}:${tag} no está en ECR. ` +
        'Ejecuta `npm run image:push` antes de desplegar.'
    )
  }
}

module.exports = async ({ resolveVariable }) => {
  const tag = imageVersion()
  if (process.env.SKIP_IMAGE_CHECK !== 'true') {
    assertPushed(await resolveVariable('self:custom.image.repositoryName'), tag)
  }
  return tag
}
module.exports.imageVersion = imageVersion

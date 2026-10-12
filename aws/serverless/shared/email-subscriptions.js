// NOTIFICATION_EMAILS="a@x.co,b@y.co" → suscripciones EMAIL del tópico.
// CloudFormation no sabe convertir un string separado por comas en esta lista.
module.exports = () => {
  const emails = (process.env.NOTIFICATION_EMAILS || '')
    .split(',')
    .map((email) => email.trim())
    .filter(Boolean)

  if (emails.length === 0) {
    throw new Error(
      'NOTIFICATION_EMAILS está vacío: define los correos (separados por ' +
        'comas) que reciben las notificaciones MLOps.'
    )
  }

  return emails.map((address) => ({
    Protocol: 'email',
    Endpoint: address,
  }))
}

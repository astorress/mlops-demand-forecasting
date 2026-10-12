import js from '@eslint/js'
import eslintConfigPrettier from 'eslint-config-prettier'
import globals from 'globals'

export default [
  { ignores: ['node_modules/**', '**/.serverless/**'] },
  js.configs.recommended,
  eslintConfigPrettier,
  {
    // Serverless Framework `${file(x.js)}` resolvers are CommonJS modules.
    files: ['serverless/**/*.js', 'container/**/*.js'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'commonjs',
      globals: {
        ...globals.node,
      },
    },
    rules: {
      'max-len': ['error', { code: 80 }],
      'no-unused-vars': 'warn',
      'no-var': 'error',
      'no-duplicate-imports': 'error',
      'no-unreachable': 'error',
    },
  },
]

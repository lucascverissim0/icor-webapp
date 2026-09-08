/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ICOR_CLIENT_RELEASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

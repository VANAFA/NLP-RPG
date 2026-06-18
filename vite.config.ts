import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/NLP-RPG/', // <-- Esto arregla la pantalla en blanco
})
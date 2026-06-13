import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react' // (or vue, svelte, etc.)

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/NLP-RPG/',
})
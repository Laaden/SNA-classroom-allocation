import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://3.105.47.11:8000",
        changeOrigin: true,
      },
    },
  },
});
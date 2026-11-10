import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 前端 dev server 把 /api 反代到本地 FastAPI。
// 默认用 8765：本地 8000已被另一个无关服务占用，避免 uvicorn 启动即端口冲突。
// 生产构建产物是纯静态文件，由后端或任意静态服务器托管即可。
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
      },
    },
  },
})

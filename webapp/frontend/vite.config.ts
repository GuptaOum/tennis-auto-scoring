import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the app runs on :5173 and the API on :8000, so both API
// prefixes are proxied. In production FastAPI serves the built bundle itself
// and same-origin relative URLs work unchanged.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/jobs": "http://127.0.0.1:8000",
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});

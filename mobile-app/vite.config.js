import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    host: true, // dostęp z urządzenia w tej samej sieci (testy na telefonie)
    port: 5173,
  },
});

import { defineConfig } from "vite";

export default defineConfig({
  // Relative base so the bundle works whether deployed at /, /alpha-zero-nano/,
  // or any other subpath on GitHub Pages without rebuilding.
  base: "./",
  build: {
    target: "es2022",
    sourcemap: true,
  },
});

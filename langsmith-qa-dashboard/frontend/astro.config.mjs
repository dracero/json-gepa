import { defineConfig } from "astro/config";

import node from "@astrojs/node";

export default defineConfig({
  output: "server",

  server: {
    port: 4321,
    host: true,
  },

  adapter: node({
    mode: "standalone",
  }),

  vite: {
    server: {
      watch: {
        // Use polling to avoid hitting inotify watch limits
        usePolling: true,
        interval: 500,
      },
    },
  },
});
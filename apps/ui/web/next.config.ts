import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Turbopack config (Next.js 16 default bundler)
  turbopack: {
    root: path.resolve(__dirname, "../../.."),
    resolveAlias: {
      "@gen": path.resolve(__dirname, "../../../gen/ts"),
    },
  },
  // Allow imports from directories outside the app root
  experimental: {
    externalDir: true,
  },
  serverExternalPackages: ["nice-grpc", "@grpc/grpc-js"],
};

export default nextConfig;

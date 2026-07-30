import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // Required for the multi-stage Docker build (Dockerfile.frontend) -
  // produces a minimal, self-contained server bundle in .next/standalone
  // instead of requiring the full node_modules tree in the final image.
  output: "standalone",
  eslint: {
    // Warning: This allows production builds to successfully complete even if your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Warning: This allows production builds to successfully complete even if your project has type errors.
    ignoreBuildErrors: true,
  },
};

export default nextConfig;

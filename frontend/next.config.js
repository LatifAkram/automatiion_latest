/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Remove deprecated appDir option
  },
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8080/:path*',
      },
      {
        source: '/automation/:path*',
        destination: 'http://localhost:8080/automation/:path*',
      },
      {
        source: '/search/:path*',
        destination: 'http://localhost:8080/search/:path*',
      },
      {
        source: '/export/:path*',
        destination: 'http://localhost:8080/export/:path*',
      },
      {
        source: '/health',
        destination: 'http://localhost:8080/health',
      },
    ];
  },
};

module.exports = nextConfig;
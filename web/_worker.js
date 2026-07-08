function releaseFromEnv(env, prefix) {
  return {
    url: env[`${prefix}_URL`],
    name: env[`${prefix}_NAME`],
    size: Number(env[`${prefix}_SIZE`]),
    sha256: env[`${prefix}_SHA256`],
    build: env[`${prefix}_BUILD`],
    builtAt: env[`${prefix}_BUILT_AT`],
  };
}

function formatMiB(bytes) {
  return `${(Number(bytes) / 1024 / 1024).toFixed(2)} MiB`;
}

function download(request, release) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return new Response("Method not allowed", {
      status: 405,
      headers: { Allow: "GET, HEAD" },
    });
  }
  if (!release.url) {
    return new Response("Release artifact not configured", { status: 404 });
  }
  return Response.redirect(release.url, 302);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const cpuRelease = releaseFromEnv(env, "RELEASE");
    const gpuRelease = releaseFromEnv(env, "RELEASE_DIRECTML");

    if (url.pathname === "/download") {
      return download(request, cpuRelease);
    }

    if (url.pathname === "/download/directml") {
      return download(request, gpuRelease);
    }

    if (url.pathname === "/metadata") {
      return Response.json({
        cpu: { ...cpuRelease, sizeMiB: formatMiB(cpuRelease.size) },
        directml: gpuRelease.url
          ? { ...gpuRelease, sizeMiB: formatMiB(gpuRelease.size) }
          : null,
      }, {
        headers: { "Cache-Control": "no-store" },
      });
    }

    if (url.pathname === "/health") {
      return Response.json({ ok: true });
    }

    if (url.pathname === "/" || url.pathname === "/index.html" || url.pathname === "/docs.html") {
      const assetUrl = new URL(url.pathname === "/" ? "/index.html" : url.pathname, request.url);
      assetUrl.searchParams.set("v", "logo-20260707");
      const response = await env.ASSETS.fetch(new Request(assetUrl, request));
      const headers = new Headers(response.headers);
      headers.set("Cache-Control", "public, max-age=60");
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    }

    return env.ASSETS.fetch(request);
  },
};

function releaseFromEnv(env) {
  return {
    url: env.RELEASE_URL,
    name: env.RELEASE_NAME,
    size: Number(env.RELEASE_SIZE),
    sha256: env.RELEASE_SHA256,
    build: env.RELEASE_BUILD,
    builtAt: env.RELEASE_BUILT_AT,
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
  return Response.redirect(release.url, 302);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const release = releaseFromEnv(env);

    if (url.pathname === "/download") {
      return download(request, release);
    }

    if (url.pathname === "/metadata") {
      return Response.json({
        ...release,
        sizeMiB: formatMiB(release.size),
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

function releaseFromEnv(env) {
  return {
    key: env.RELEASE_KEY,
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

function objectHeaders(object, release, status = 200, extra = {}) {
  const headers = new Headers(extra);
  headers.set("Accept-Ranges", "bytes");
  headers.set("Cache-Control", "public, max-age=3600");
  headers.set("Content-Disposition", `attachment; filename="${release.name}"`);
  headers.set("Content-Type", "application/vnd.microsoft.portable-executable");
  headers.set("X-Content-Type-Options", "nosniff");
  headers.set("X-Release-SHA256", release.sha256);
  if (status !== 206) {
    headers.set("Content-Length", String(object?.size || release.size));
  }
  if (object?.httpEtag) {
    headers.set("ETag", object.httpEtag);
  }
  return headers;
}

function parseRange(rangeHeader, size) {
  if (!rangeHeader) return null;
  const match = rangeHeader.match(/^bytes=(\d*)-(\d*)$/);
  if (!match) return { error: true };
  let start = match[1] === "" ? null : Number(match[1]);
  let end = match[2] === "" ? null : Number(match[2]);
  if (start === null && end === null) return { error: true };
  if (start === null) {
    if (!Number.isFinite(end) || end <= 0) return { error: true };
    start = Math.max(0, size - end);
    end = size - 1;
  } else if (end === null || end >= size) {
    end = size - 1;
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start || start >= size) {
    return { error: true };
  }
  return { offset: start, length: end - start + 1, end };
}

async function download(request, env, release) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return new Response("Method not allowed", {
      status: 405,
      headers: { Allow: "GET, HEAD" },
    });
  }

  const head = await env.NVAP_DOWNLOADS.head(release.key);
  if (!head) {
    return new Response("Release artifact not found", { status: 404 });
  }

  if (request.method === "HEAD") {
    return new Response(null, {
      headers: objectHeaders(head, release),
    });
  }

  const range = parseRange(request.headers.get("Range"), head.size);
  if (range?.error) {
    return new Response("Requested range not satisfiable", {
      status: 416,
      headers: { "Content-Range": `bytes */${head.size}` },
    });
  }

  const options = range ? { range: { offset: range.offset, length: range.length } } : undefined;
  const object = await env.NVAP_DOWNLOADS.get(release.key, options);
  if (!object) {
    return new Response("Release artifact not found", { status: 404 });
  }

  if (range) {
    return new Response(object.body, {
      status: 206,
      headers: objectHeaders(object, release, 206, {
        "Content-Length": String(range.length),
        "Content-Range": `bytes ${range.offset}-${range.end}/${head.size}`,
      }),
    });
  }

  return new Response(object.body, {
    headers: objectHeaders(object, release),
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const release = releaseFromEnv(env);

    if (url.pathname === "/download") {
      return download(request, env, release);
    }

    if (url.pathname === "/metadata") {
      const head = await env.NVAP_DOWNLOADS.head(release.key);
      return Response.json({
        ...release,
        r2ObjectPresent: Boolean(head),
        r2Size: head?.size || null,
        sizeMiB: formatMiB(head?.size || release.size),
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

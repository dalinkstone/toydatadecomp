// Ultra-lightweight Bun HTTP server — zero dependencies
// Run: bun run web/server.ts

const file = Bun.file(import.meta.dir + "/index.html");

Bun.serve({
  port: 3000,
  async fetch(req) {
    const url = new URL(req.url);
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(file, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }
    return new Response("Not Found", { status: 404 });
  },
});

console.log("CVS Analytics Dashboard → http://localhost:3000");

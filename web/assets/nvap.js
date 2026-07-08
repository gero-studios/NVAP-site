// NVAP site — scroll reveals, accordion, docs TOC, and the hero neurovascular canvas.
(() => {
  'use strict';
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ── Scroll reveals ─────────────────────────────────────────
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    }
  }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
  document.querySelectorAll('.reveal').forEach((el) => io.observe(el));
  // Safety net: if the observer never delivers (e.g. background/headless tab),
  // reveal everything so content is never left invisible.
  setTimeout(() => {
    if (!document.querySelector('.reveal.in')) {
      document.querySelectorAll('.reveal').forEach((el) => el.classList.add('in'));
    }
  }, 1600);

  // ── Accordion ──────────────────────────────────────────────
  document.querySelectorAll('.acc-q').forEach((q) => {
    q.addEventListener('click', () => {
      const item = q.closest('.acc-item');
      const ans = item.querySelector('.acc-a');
      const open = item.classList.toggle('open');
      ans.style.maxHeight = open ? ans.scrollHeight + 'px' : '0px';
    });
  });

  // ── Docs TOC active highlight ──────────────────────────────
  const toc = document.querySelectorAll('.toc a');
  if (toc.length) {
    const targets = [...toc].map((a) => document.querySelector(a.getAttribute('href'))).filter(Boolean);
    const tocIO = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          toc.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
        }
      }
    }, { rootMargin: '-20% 0px -70% 0px' });
    targets.forEach((t) => tocIO.observe(t));
  }

  // ── Hero neurovascular canvas ──────────────────────────────
  const canvas = document.getElementById('neuro');
  if (canvas && !reduce) {
    const ctx = canvas.getContext('2d');
    let W, H, dpr, nodes = [], vessels = [], t = 0;
    const GREEN = 'rgba(87,217,163,', RED = 'rgba(240,87,107,';

    function resize() {
      W = canvas.clientWidth; H = canvas.clientHeight;
      if (!W || !H) { requestAnimationFrame(resize); return; }  // wait for layout
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      build();
    }
    function build() {
      // Microglia: drifting green nodes.
      const N = Math.max(18, Math.min(40, Math.round(W / 34)));
      nodes = Array.from({ length: N }, () => ({
        x: Math.random() * W, y: Math.random() * H,
        vx: (Math.random() - 0.5) * 0.18, vy: (Math.random() - 0.5) * 0.18,
        r: 1 + Math.random() * 1.8, ph: Math.random() * Math.PI * 2,
      }));
      // Vasculature: a few red bezier strands flowing left→right.
      const V = Math.max(3, Math.round(W / 320));
      vessels = Array.from({ length: V }, (_, i) => ({
        y: (H / (V + 1)) * (i + 1) + (Math.random() - 0.5) * 40,
        amp: 24 + Math.random() * 34, k: 0.006 + Math.random() * 0.004, sp: 0.2 + Math.random() * 0.3,
      }));
    }
    function frame() {
      t += 1;
      ctx.clearRect(0, 0, W, H);

      // Vessels (red flowing strands)
      for (const v of vessels) {
        ctx.beginPath();
        for (let x = -20; x <= W + 20; x += 6) {
          const y = v.y + Math.sin(x * v.k + t * 0.01 * v.sp) * v.amp + Math.sin(x * v.k * 2.3 + t * 0.006) * (v.amp * 0.3);
          x === -20 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.strokeStyle = RED + '0.16)'; ctx.lineWidth = 1.4; ctx.stroke();
        ctx.strokeStyle = RED + '0.05)'; ctx.lineWidth = 5; ctx.stroke();
      }

      // Node-to-node microglia links
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        a.x += a.vx; a.y += a.vy;
        if (a.x < 0 || a.x > W) a.vx *= -1;
        if (a.y < 0 || a.y > H) a.vy *= -1;
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j], dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy);
          if (d < 110) {
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = GREEN + (0.10 * (1 - d / 110)).toFixed(3) + ')'; ctx.lineWidth = 0.7; ctx.stroke();
          }
        }
      }
      // Node dots (pulsing green somata)
      for (const n of nodes) {
        const pulse = 0.55 + 0.45 * Math.sin(t * 0.03 + n.ph);
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = GREEN + (0.5 + pulse * 0.4).toFixed(3) + ')'; ctx.fill();
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r * 3.2, 0, Math.PI * 2);
        ctx.fillStyle = GREEN + (0.05 * pulse).toFixed(3) + ')'; ctx.fill();
      }
      raf = requestAnimationFrame(frame);
    }
    let raf; resize(); frame();
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(resize, 150); });
  }
})();

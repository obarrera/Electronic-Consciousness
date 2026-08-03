import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';
import { chromium } from 'playwright';

const data = JSON.parse(fs.readFileSync('out/scenes.json', 'utf8'));
// Remove any stale recording up front: a failed run must never leave an old
// recording in place for render.sh to silently pick up.
try { fs.unlinkSync('out/recordings/browser-recording.webm'); } catch (err) {}
try { fs.unlinkSync('out/recordings/browser-recording.mp4'); } catch (err) {}
try { fs.rmSync('out/recordings/frames', { recursive: true, force: true }); } catch (err) {}
// timing.json must die with it: a stale lead-in from a previous run would make
// render.sh trim the wrong amount and offset the whole video from the audio.
try { fs.unlinkSync('out/recordings/timing.json'); } catch (err) {}
const cfg = data.config || {};
const scenes = data.scenes || [];
const viewport = cfg.viewport || { width: 1920, height: 1080 };
const recordingDir = path.resolve('out/recordings/raw');
fs.mkdirSync(recordingDir, { recursive: true });

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function esc(text) {
  return String(text || '').replace(/[&<>'"]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[ch]));
}

async function applyZoom(page, scale, selector, text) {
  await page.evaluate(({ scale, selector, text }) => {
    let origin = '50% 35%';
    let target = null;
    if (selector) target = document.querySelector(selector);
    if (!target && text) {
      const needle = String(text).toLowerCase();
      target = [...document.querySelectorAll('h1,h2,h3,a,button,td,th,span,div')]
        .find((el) => el.childElementCount === 0 && (el.innerText || '').trim().toLowerCase().includes(needle));
    }
    if (target) {
      const r = target.getBoundingClientRect();
      origin = `${r.left + r.width / 2}px ${Math.max(0, r.top + r.height / 2)}px`;
    }
    document.body.style.transition = 'transform 1.4s cubic-bezier(.4,0,.2,1)';
    document.body.style.transformOrigin = origin;
    document.body.style.transform = `scale(${scale})`;
  }, { scale, selector: selector || null, text: text || null }).catch(() => {});
}

async function resetZoom(page) {
  await page.evaluate(() => {
    if (document.body.style.transform) {
      document.body.style.transition = 'transform 0.7s ease';
      document.body.style.transform = '';
    }
  }).catch(() => {});
}

// Smooths the hard cut a bare page.goto/navigation otherwise produces: fade the
// CURRENT page to brand purple before navigating away, and — because the init
// script below re-injects the same overlay into every new document, starting
// opaque — the incoming page fades FROM purple instead of hard-cutting in. Call
// transitionOut() right before every goto/slide navigation in the scene loop.
const TRANSITION_INIT_SCRIPT = () => {
  const install = () => {
    if (document.getElementById('socket-demo-transition')) return;
    const el = document.createElement('div');
    el.id = 'socket-demo-transition';
    el.style.cssText = 'position:fixed;inset:0;background:#1f1147;opacity:1;' +
      'pointer-events:none;z-index:2147483647;transition:opacity .35s ease;';
    (document.body || document.documentElement).appendChild(el);
    requestAnimationFrame(() => requestAnimationFrame(() => { el.style.opacity = '0'; }));
  };
  if (document.body) install();
  else document.addEventListener('DOMContentLoaded', install);
};

// Visible pointer feedback: headless/CDP-style captures don't film the OS
// cursor, so demos look like the page drives itself. A small injected cursor
// dot follows real mouse events and every mousedown emits a click ripple —
// the standard technique OSS demo recorders use. Stays offscreen until the
// first real mouse move, so slide scenes aren't decorated with a stray dot.
// Disable per-project with recording.cursor_overlay: false.
const CURSOR_INIT_SCRIPT = () => {
  const install = () => {
    if (document.getElementById('socket-demo-cursor')) return;
    const style = document.createElement('style');
    style.textContent = [
      '#socket-demo-cursor { position:fixed; width:20px; height:20px; border-radius:50%;',
      '  background:rgba(109,40,217,.85); border:2px solid #fff;',
      '  box-shadow:0 1px 6px rgba(0,0,0,.4); pointer-events:none; z-index:2147483646;',
      '  transform:translate(-50%,-50%); left:-100px; top:-100px;',
      '  transition:left .08s linear, top .08s linear; }',
      '.socket-demo-ripple { position:fixed; border-radius:50%; pointer-events:none;',
      '  border:3px solid rgba(109,40,217,.9); z-index:2147483645;',
      '  transform:translate(-50%,-50%); animation:socket-demo-ripple .6s ease-out forwards; }',
      '@keyframes socket-demo-ripple { from { opacity:.9; width:14px; height:14px; }',
      '  to { opacity:0; width:90px; height:90px; } }',
    ].join('\n');
    (document.head || document.documentElement).appendChild(style);
    const cur = document.createElement('div');
    cur.id = 'socket-demo-cursor';
    (document.body || document.documentElement).appendChild(cur);
    document.addEventListener('mousemove', (e) => {
      cur.style.left = e.clientX + 'px';
      cur.style.top = e.clientY + 'px';
    }, true);
    document.addEventListener('mousedown', (e) => {
      const r = document.createElement('div');
      r.className = 'socket-demo-ripple';
      r.style.left = e.clientX + 'px';
      r.style.top = e.clientY + 'px';
      (document.body || document.documentElement).appendChild(r);
      setTimeout(() => r.remove(), 700);
    }, true);
  };
  if (document.body) install();
  else document.addEventListener('DOMContentLoaded', install);
};

async function transitionOut(page) {
  await page.evaluate(() => {
    let el = document.getElementById('socket-demo-transition');
    if (!el) {
      el = document.createElement('div');
      el.id = 'socket-demo-transition';
      el.style.cssText = 'position:fixed;inset:0;background:#1f1147;opacity:0;' +
        'pointer-events:none;z-index:2147483647;transition:opacity .35s ease;';
      (document.body || document.documentElement).appendChild(el);
    }
    requestAnimationFrame(() => { el.style.opacity = '1'; });
  }).catch(() => {});
  await sleep(360);
}

async function dismissPopups(page) {
  // Product tours, announcement modals, dropdowns, and stray selections ruin
  // recordings. Run dismissal in rounds until a round clears nothing (tours can
  // be multi-step): press Escape, then click common dismiss buttons tolerantly
  // (Socket's dashboard tour uses "Skip all").
  const labels = ['Skip all', 'Skip', 'Dismiss', 'Got it', 'No thanks', 'Maybe later', 'Close'];
  for (let round = 0; round < 3; round++) {
    let dismissed = 0;
    try { await page.keyboard.press('Escape'); } catch (err) {}
    for (const label of labels) {
      try {
        const btn = page.getByRole('button', { name: label }).first();
        if (await btn.isVisible({ timeout: 180 })) {
          await btn.click({ timeout: 800 });
          await page.waitForTimeout(350);
          dismissed++;
        }
      } catch (err) {}
    }
    try {
      const x = page.locator('[aria-label="Close" i]').first();
      if (await x.isVisible({ timeout: 180 })) { await x.click({ timeout: 800 }); dismissed++; }
    } catch (err) {}
    if (dismissed === 0) break;
  }
}

async function injectOverlay(page) {
  await page.addStyleTag({ content: `
    #socket-demo-callout {
      position: fixed;
      left: 48px;
      bottom: 132px;
      max-width: 780px;
      z-index: 2147483647;
      display: flex;
      align-items: center;
      gap: 18px;
      padding: 14px 30px 14px 16px;
      border-radius: 16px;
      background: linear-gradient(135deg, #553c9a 0%, #6d28d9 100%);
      color: #ffffff;
      font: 600 28px/1.3 -apple-system, 'Segoe UI', Inter, Arial, sans-serif;
      letter-spacing: 0.01em;
      box-shadow: 0 14px 40px rgba(31, 17, 71, 0.45);
      border: 1px solid rgba(255, 255, 255, 0.18);
    }
    #socket-demo-callout img.socket-mark {
      width: 46px;
      height: 46px;
      flex: none;
    }
    .socket-demo-highlight {
      outline: 5px solid rgba(124, 58, 237, 0.95) !important;
      box-shadow: 0 0 0 8px rgba(124, 58, 237, 0.25) !important;
      border-radius: 8px !important;
    }
  ` });
}

const SOCKET_WORDMARK_DATA_URI = 'data:image/png;base64,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';

// Branded slide scenes: `action: 'slide'` renders scene.slide {title, subtitle,
// bullets[], stat_value, stat_label} as a Socket-branded local page and films it
// like any other scene — mix frameworks/stats with live pages in one video.
function buildSlideHtml(scene) {
  const sl = scene.slide || {};
  const bullets = (sl.bullets || [])
    .map((b) => `<li>${esc(String(b))}</li>`).join('');
  const statNum = sl.stat_value !== undefined && sl.stat_value !== null ? String(sl.stat_value) : null;
  const stat = statNum
    ? `<div class="stat" data-target="${esc(statNum)}">${/^[\d\s,\.]+$/.test(statNum) ? '0' : esc(statNum)}</div>` +
      (sl.stat_label ? `<div class="statlabel">${esc(String(sl.stat_label))}</div>` : '')
    : '';
  // Simple single-series horizontal bars: slide.bars = [{label, value}]
  const barData = Array.isArray(sl.bars) ? sl.bars : [];
  const maxVal = Math.max(1, ...barData.map((b) => Number(b.value) || 0));
  const bars = barData.map((b) => {
    const pct = Math.round(100 * (Number(b.value) || 0) / maxVal);
    return `<div class="barrow"><div class="barlabel">${esc(String(b.label))}</div>` +
      `<div class="bartrack"><div class="bar" style="--w:${pct}%"></div></div>` +
      `<div class="barval">${esc(String(b.value))}</div></div>`;
  }).join('');
  const image = sl.image
    ? `<div class="imgwrap"><img class="kenburns" src="${esc(String(sl.image))}" alt="">` +
      (sl.image_caption ? `<div class="cap">${esc(String(sl.image_caption))}</div>` : '') + `</div>`
    : '';
  // Simple branded workflow/architecture diagram: slide.flow = ['step 1', 'step 2', …]
  // renders as boxes connected by arrows. Use this instead of a sourced image when
  // there's no real diagram to pull in (docs often describe architecture in prose
  // only) — it's built from the skill's own verified narration text, not fabricated.
  const flowSteps = Array.isArray(sl.flow) ? sl.flow : [];
  const flow = flowSteps.length
    ? `<div class="flow">${flowSteps.map((s, i) =>
        (i > 0 ? '<div class="flowarrow">&#8594;</div>' : '') +
        `<div class="flowstep">${esc(String(s))}</div>`
      ).join('')}</div>`
    : '';
  // Optional subtle background video loop (opt-in via slide.background_video) —
  // stays low-opacity behind the branded gradient/content, never replaces it.
  // Only use licensed/royalty-free clips; the skill ships none by default (see
  // references/video-library.md for how to source one).
  const bgVideo = sl.background_video
    ? `<video class="bgvideo" autoplay muted loop playsinline src="${esc(String(sl.background_video))}"></video>`
    : '';
  return `<!DOCTYPE html><html><head><style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { width:1920px; height:1080px; position:relative; overflow:hidden;
           background: radial-gradient(1200px 700px at 30% 20%, #6d28d9 0%, #553c9a 38%, #2a1a5e 78%, #1f1147 100%);
           font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif; color:#fff; }
    .bgvideo { position:absolute; inset:0; width:100%; height:100%; object-fit:cover;
               opacity:0.16; z-index:0; mix-blend-mode:screen; }
    .content { position:relative; z-index:1; width:100%; height:100%; display:flex;
                flex-direction:column; align-items:center; justify-content:center;
                gap:38px; text-align:center; padding:0 140px; }
    @keyframes fadeUp { from { opacity:0; transform:translateY(26px); } to { opacity:1; transform:none; } }
    @keyframes grow { from { width:0; } to { width:var(--w); } }
    @keyframes drift { from { transform:scale(1); } to { transform:scale(1.06) translateY(-8px); } }
    img.wordmark { width:300px; animation: fadeUp .8s ease both; }
    h1 { font-size:62px; font-weight:700; letter-spacing:-0.01em; line-height:1.15; max-width:1520px;
         animation: fadeUp .8s ease .15s both; }
    .sub { font-size:31px; opacity:.9; max-width:1400px; animation: fadeUp .8s ease .3s both; }
    ul { list-style:none; font-size:37px; line-height:1.85; font-weight:600; text-align:left; max-width:1440px; }
    ul li { animation: fadeUp .7s ease both; }
    ul li::before { content:'•  '; color:#c4b5fd; }
    ${(sl.bullets || []).map((_, i) => `ul li:nth-child(${i + 1}) { animation-delay: ${0.35 + i * 0.28}s; }`).join('\n    ')}
    .stat { font-size:165px; font-weight:800; line-height:1; animation: fadeUp .8s ease .2s both; }
    .statlabel { font-size:35px; opacity:.9; animation: fadeUp .8s ease .45s both; }
    .barrow { display:grid; grid-template-columns: 430px 1fr 130px; align-items:center; gap:22px;
              width:1460px; margin:9px 0; animation: fadeUp .7s ease both; }
    ${barData.map((_, i) => `.barrow:nth-of-type(${i + 1}) { animation-delay: ${0.35 + i * 0.24}s; }`).join('\n    ')}
    .barlabel { font-size:31px; font-weight:600; text-align:right; }
    .bartrack { height:44px; background:rgba(255,255,255,.12); border-radius:10px; overflow:hidden; }
    .bar { height:100%; width:var(--w); border-radius:10px;
           background:linear-gradient(90deg,#a78bfa,#7c3aed); animation: grow 1.4s cubic-bezier(.3,0,.2,1) .5s both; }
    .barval { font-size:31px; font-weight:700; text-align:left; }
    .imgwrap { max-width:1500px; border-radius:14px; overflow:hidden;
               box-shadow:0 20px 60px rgba(0,0,0,.45); animation: fadeUp .9s ease .25s both; }
    .imgwrap img { display:block; max-width:100%; max-height:640px; animation: drift 14s ease both; }
    .cap { font-size:26px; opacity:.85; padding:14px; background:rgba(0,0,0,.35); }
    .flow { display:flex; align-items:center; justify-content:center; flex-wrap:wrap;
            gap:10px; max-width:1700px; }
    .flowstep { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.22);
                border-radius:16px; padding:26px 32px; font-size:27px; font-weight:700;
                line-height:1.25; max-width:340px; animation: fadeUp .6s ease both; }
    .flowarrow { font-size:44px; font-weight:700; color:#c4b5fd; opacity:.75;
                 animation: fadeUp .5s ease both; }
    ${flowSteps.map((_, i) => `.flow > *:nth-child(${2 * i + 1}) { animation-delay: ${0.3 + i * 0.32}s; }`).join('\n    ')}
    /* Arrows (even children) enter AFTER the step they point from — not all at
       t=0 pointing between empty space. */
    ${flowSteps.map((_, i) => i === 0 ? '' : `.flow > *:nth-child(${2 * i}) { animation-delay: ${(0.3 + (i - 1) * 0.32 + 0.18).toFixed(2)}s; }`).filter(Boolean).join('\n    ')}
  </style></head><body>
    ${bgVideo}
    <div class="content">
      <div style="font-size:30px;font-weight:700;letter-spacing:.2em;opacity:.7;text-transform:uppercase;animation:fadeUp .8s ease both;">Electronic Consciousness</div>
      ${sl.title ? `<h1>${esc(String(sl.title))}</h1>` : ''}
      ${sl.subtitle ? `<div class="sub">${esc(String(sl.subtitle))}</div>` : ''}
      ${stat}
      ${bars ? `<div class="bars">${bars}</div>` : ''}
      ${flow}
      ${bullets ? `<ul>${bullets}</ul>` : ''}
      ${image}
    </div>
    <script>
      const el = document.querySelector('.stat');
      if (el) {
        const raw = el.dataset.target || '';
        const target = Number(raw.replace(/[^\\d.]/g, ''));
        if (Number.isFinite(target) && /^[\\d\\s,\\.]+$/.test(raw)) {
          const t0 = performance.now(), dur = 1600;
          const fmt = (n) => Math.round(n).toLocaleString('en-US').replace(/,/g, ' ');
          (function tick(now) {
            const k = Math.min(1, (now - t0) / dur);
            el.textContent = fmt(target * (1 - Math.pow(1 - k, 3)));
            if (k < 1) requestAnimationFrame(tick);
          })(t0);
        }
      }
    </script>
  </body></html>`;
}

const SOCKET_MARK_DATA_URI = 'data:image/png;base64,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';

async function callout(page, text) {
  await page.evaluate(({ html, mark }) => {
    let el = document.getElementById('socket-demo-callout');
    if (!el) {
      el = document.createElement('div');
      el.id = 'socket-demo-callout';
      // Attach to <html>, not <body>: scene zooms transform <body>, and a
      // transformed ancestor would drag the fixed-position callout with it.
      document.documentElement.appendChild(el);
    }
    el.innerHTML = `<img class="socket-mark" src="${mark}" alt=""><span>${html}</span>`;
  // .catch: these run in the scene-loop preamble OUTSIDE the per-scene try — an
  // "Execution context was destroyed" during an in-flight navigation must not
  // kill the whole recording.
  }, { html: esc(text), mark: SOCKET_MARK_DATA_URI }).catch(() => {});
}

async function clearHighlights(page) {
  await page.evaluate(() => {
    document.querySelectorAll('.socket-demo-highlight').forEach(el => el.classList.remove('socket-demo-highlight'));
  }).catch(() => {});
}

async function highlightText(page, text) {
  if (!text) return false;
  return await page.evaluate((needle) => {
    const normalized = needle.toLowerCase();
    const candidates = Array.from(document.querySelectorAll('a,button,[role="button"],h1,h2,h3,p,span,div,td,th'));
    // Pick the TIGHTEST match, not the first: the first candidate is usually a
    // page-wrapper div whose innerText contains everything, and scrollIntoView
    // on the scroll container itself is a no-op (verified on dashboard alert pages).
    const match = candidates
      .filter(el => (el.innerText || el.textContent || '').toLowerCase().includes(normalized))
      .sort((a, b) => ((a.innerText || a.textContent || '').length) - ((b.innerText || b.textContent || '').length))[0];
    if (match) {
      match.classList.add('socket-demo-highlight');
      match.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
      return true;
    }
    return false;
  }, text);
}

// READ-ONLY GUARDRAILS: demos must never change org state. With read_only on
// (the default), clicks on anything that looks mutating are refused and
// navigation is limited to allowed prefixes with /settings & /billing excluded.
const MUTATING_RE = /\b(save|delete|remove|create|edit|invite|update|enable|disable|configure|add|new token|upgrade|billing|install|uninstall|rotate|revoke|transfer|archive|apply|submit|confirm|accept|approve|assign|ignore|resolve|mark as|patch|create ticket|submit pr|import)\b/i;

function isReadOnly(cfg) { return cfg.read_only !== false; }

function gotoAllowed(cfg, url) {
  if (!isReadOnly(cfg)) return true;
  if (/\/settings|\/billing|\/api-tokens/i.test(url)) return false;
  const prefixes = cfg.allowed_url_prefixes || ['https://socket.dev', 'https://docs.socket.dev', 'file://', 'about:blank'];
  // Prefix must end at a URL boundary: "https://socket.dev" must NOT admit
  // "https://socket.dev.evil.example".
  return prefixes.some((p) => {
    const u = String(url);
    if (!u.startsWith(p)) return false;
    if (p.endsWith('/') || p.endsWith(':')) return true;
    const rest = u.slice(p.length);
    return rest === '' || /^[\/?#]/.test(rest);
  });
}

async function safeClick(cfg, locator, timeout) {
  if (isReadOnly(cfg)) {
    // Links navigate — always allowed for demoing/exploring. Only buttons and
    // other controls get the mutating-text check.
    // Explicit short timeouts: locator calls otherwise inherit Playwright's 30s
    // default wait-for-element, and one missing selector would stall the scene
    // far past its narration and desync every scene after it.
    const tag = await locator.evaluate((el) => el.closest('a') ? 'a' : el.tagName.toLowerCase(), null, { timeout: 1500 }).catch(() => '');
    if (tag !== 'a') {
      const label = await locator.innerText({ timeout: 800 }).catch(() => '') ||
        await locator.getAttribute('aria-label', { timeout: 800 }).catch(() => '') || '';
      if (MUTATING_RE.test(label)) {
        console.log(`read-only: refused click on "${String(label).slice(0, 60)}"`);
        return false;
      }
    }
  }
  await locator.click({ timeout });
  return true;
}

async function clickByText(page, cfg, text, timeout) {
  if (!text || text === 'first visible alert') return false;
  // Escape regex metacharacters: scene text is a literal label. "Alerts (3)"
  // must match literally and "C++ SDK" must not throw SyntaxError.
  const pattern = new RegExp(String(text).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'i');
  const locators = [
    page.getByRole('link', { name: pattern }).first(),
    page.getByRole('button', { name: pattern }).first(),
    page.getByText(pattern).first(),
  ];
  for (const locator of locators) {
    try {
      // Fast pre-check: a locator that doesn't resolve must cost well under a
      // second, not a full stack of evaluate/innerText/click timeouts (~8s
      // each) that blows the scene's audio slot and desyncs everything after
      // it. waitFor (not isVisible, whose timeout option is a no-op) gives
      // hydrating SPAs a real, bounded chance to render the target.
      const visible = await locator.waitFor({ state: 'visible', timeout: 700 })
        .then(() => true).catch(() => false);
      if (!visible) continue;
      if (await safeClick(cfg, locator, timeout)) return true;
      // Refused as mutating — try the next locator: a different element with
      // the same accessible text (a plain link vs a button) may be safe.
    } catch (err) {
      // try next locator
    }
  }
  return false;
}

async function setupMockPage(page) {
  const feature = esc(cfg.feature_name || 'Socket Demo');
  const audience = esc(cfg.audience || 'new Socket users');
  await page.setContent(`
    <html>
      <head>
        <title>${feature}</title>
        <style>
          body { margin: 0; font-family: Inter, Arial, sans-serif; background: #f6f7fb; color: #111827; }
          header { background: linear-gradient(135deg, #111827, #6d28d9); color: white; padding: 44px 64px; }
          h1 { margin: 0 0 12px; font-size: 54px; }
          main { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 170px); }
          nav { background: white; padding: 28px; border-right: 1px solid #e5e7eb; }
          nav a { display: block; padding: 16px 18px; margin-bottom: 10px; border-radius: 12px; color: #111827; text-decoration: none; font-weight: 700; }
          nav a.active, nav a:hover { background: #ede9fe; color: #5b21b6; }
          section { padding: 42px; }
          .card { background: white; border: 1px solid #e5e7eb; border-radius: 18px; padding: 28px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(17,24,39,0.08); }
          .risk { color: #7c2d12; background: #ffedd5; padding: 8px 12px; border-radius: 999px; font-weight: 800; }
          .package { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 24px; }
        </style>
      </head>
      <body>
        <header>
          <h1>${feature}</h1>
          <p>Demo for ${audience}</p>
        </header>
        <main>
          <nav>
            <a class="active">Overview</a>
            <a href="#alerts">Alerts</a>
            <a href="#repositories">Repositories</a>
            <a href="#dependencies">Dependencies</a>
            <a href="#policies">Policies</a>
          </nav>
          <section>
            <div class="card">
              <h2>Organization Alerts</h2>
              <p>Review dependency risk across connected repositories.</p>
              <span class="risk">High Risk</span>
            </div>
            <div class="card">
              <h2>Package Risk Summary</h2>
              <p class="package">example-package@1.2.3</p>
              <p>This package shows suspicious behavior and should be reviewed before use.</p>
            </div>
            <div class="card">
              <h2>Recommended Action</h2>
              <p>Update, remove, ignore with context, or escalate to the security team.</p>
            </div>
          </section>
        </main>
      </body>
    </html>
  `, { waitUntil: 'domcontentloaded' });
}

const browserCfg = cfg.browser || {};
const profileDir = (browserCfg.profile || '').replace(/^~(?=\/)/, process.env.HOME);
const usePersistentProfile = cfg.demo_mode === 'live' && profileDir;

// Capture mode (Phase 7): 'cdp' (default) streams full-quality JPEG frames via
// Page.startScreencast — Playwright's recordVideo hardcodes VP8 at ~1 Mb/s
// (upstream won't-fix, #12056/#31424), which decides how mushy UI text looks
// before any render setting can help. 'playwright' keeps the legacy path.
const captureMode = String(cfg.recording?.capture || 'cdp').toLowerCase() === 'playwright' ? 'playwright' : 'cdp';
const captureQuality = Math.min(100, Math.max(30, Number(cfg.recording?.capture_quality) || 95));
const captureScale = Number(cfg.recording?.capture_scale) === 2 ? 2 : 1;
const framesDir = path.resolve('out/recordings/frames');
const recordVideoOption = captureMode === 'cdp' ? {} : { recordVideo: { dir: recordingDir, size: viewport } };

let browser = null;
let context;
if (usePersistentProfile) {
  // Headed persistent profile: behaves like a normal user browser, so it carries
  // Cloudflare clearance and the Socket login across runs. Prime it once with
  // `npm run prime`, then recordings are hands-free.
  fs.mkdirSync(profileDir, { recursive: true });
  context = await chromium.launchPersistentContext(profileDir, {
    headless: false,
    channel: browserCfg.channel || 'chrome',
    chromiumSandbox: true,
    slowMo: Number(cfg.recording?.slow_mo_ms || 0),
    viewport,
    ...recordVideoOption,
  });
  console.log(`Recording with persistent profile: ${profileDir}`);
} else {
  browser = await chromium.launch({ headless: true, slowMo: Number(cfg.recording?.slow_mo_ms || 0) });
  const contextOptions = {
    viewport,
    ...recordVideoOption,
  };
  const authState = cfg.auth_state;
  if (cfg.auth_mode === 'playwright_storage_state' && authState) {
    if (fs.existsSync(authState)) {
      contextOptions.storageState = authState;
    } else if (cfg.strict_auth) {
      throw new Error(`Auth storage state not found: ${authState}`);
    } else {
      console.warn(`Auth storage state not found: ${authState}. Continuing without it.`);
    }
  }
  context = await browser.newContext(contextOptions);
}
// In persistent mode reuse the initial tab so no stray about:blank tab lingers.
const page = usePersistentProfile
  ? (context.pages()[0] || await context.newPage())
  : await context.newPage();
await page.addInitScript(TRANSITION_INIT_SCRIPT);
if (cfg.recording?.cursor_overlay !== false) await page.addInitScript(CURSOR_INIT_SCRIPT);

// --- CDP screencast capture (Phase 7) ---
let cdpSession = null;
let frameIndex = 0;
let lastFrameWallMs = 0;
const frameManifest = []; // { file, wallMs }
async function startScreencast() {
  cdpSession = await context.newCDPSession(page);
  cdpSession.on('Page.screencastFrame', (ev) => {
    try {
      const file = path.join(framesDir, `f-${String(frameIndex++).padStart(6, '0')}.jpg`);
      fs.writeFileSync(file, Buffer.from(ev.data, 'base64'));
      const wallMs = Date.now();
      frameManifest.push({ file, wallMs });
      lastFrameWallMs = wallMs;
    } catch (err) {}
    // Unacked frames stall the stream — always ack, even after a write error.
    cdpSession.send('Page.screencastFrameAck', { sessionId: ev.sessionId }).catch(() => {});
  });
  await cdpSession.send('Page.startScreencast', {
    format: 'jpeg',
    quality: captureQuality,
    maxWidth: viewport.width * captureScale,
    maxHeight: viewport.height * captureScale,
    everyNthFrame: 1,
  });
}
if (captureMode === 'cdp') {
  fs.mkdirSync(framesDir, { recursive: true });
  try {
    await startScreencast();
    console.log(`Capture: CDP screencast (jpeg q${captureQuality}, scale ${captureScale}x)`);
  } catch (err) {
    // recordVideo can only be configured at context creation, so a CDP failure
    // here means there is no capture at all. Nothing has been recorded yet —
    // fail loudly with the one-flag fix rather than film into the void.
    console.error(`CDP screencast unavailable (${String(err.message || err)}).`);
    console.error('Set recording.capture: playwright in demo.yaml and re-run.');
    await context.close().catch(() => {});
    if (browser) await browser.close().catch(() => {});
    process.exit(1);
  }
  // A cross-process navigation can silently kill a screencast session. If no
  // frame arrives for 5s after a main-frame navigation, re-attach.
  page.on('framenavigated', (frame) => {
    if (frame !== page.mainFrame()) return;
    const markMs = Date.now();
    setTimeout(async () => {
      if (lastFrameWallMs < markMs) {
        try {
          await startScreencast();
          console.log('CDP screencast re-attached after navigation');
        } catch (err) {}
      }
    }, 5000);
  });
}
const video = captureMode === 'cdp' ? null : page.video();
// Wall clock anchored to the start of the recording: scene pacing below is
// absolute against this, so setup/navigation overhead can't accumulate as
// audio/visual drift.
const recordingStart = Date.now();

if (cfg.demo_mode === 'mock' || !cfg.demo_url || cfg.demo_url === 'about:blank') {
  await setupMockPage(page);
} else {
  // The initial navigation obeys the same read-only allow-list as scene gotos.
  if (!gotoAllowed(cfg, cfg.demo_url)) {
    console.error(`read-only: demo_url refused by allow-list: ${cfg.demo_url}`);
    process.exit(1);
  }
  await page.goto(cfg.demo_url, { waitUntil: 'domcontentloaded', timeout: 60000 });
}
await dismissPopups(page);
if (cfg.demo_mode === 'live') {
  // One reload after the first dismissal flushes lingering tour/announcement
  // state before any scene is on camera.
  await page.reload({ waitUntil: 'domcontentloaded' }).catch(() => {});
  await page.waitForTimeout(1500);
  await dismissPopups(page);
}
await injectOverlay(page);
await sleep(1000);

// Everything up to here (goto, bot-check, reload, dismissal) happened while the
// recording was already rolling. Measure it: render.sh trims this lead-in so the
// first scene starts exactly at audio 0.
const leadInMs = Date.now() - recordingStart;
console.log(`Lead-in before scene 1: ${(leadInMs / 1000).toFixed(1)}s (trimmed at render)`);

let audioDurations = {};
try {
  const manifest = JSON.parse(fs.readFileSync('out/audio/scene_durations.json', 'utf8'));
  for (const entry of manifest) audioDurations[entry.id] = Number(entry.seconds);
  console.log('Pacing scenes from out/audio/scene_durations.json');
  // A stale manifest (edited scenes since the last audio run) silently falls
  // back to planned durations for the missing ids while voice.wav has the OLD
  // lengths — cumulative A/V desync. Make that loud.
  const missing = scenes.filter((s) => !(s.id in audioDurations)).map((s) => s.id);
  if (missing.length) {
    console.warn(`WARNING: scene_durations.json is missing ids [${missing.join(', ')}] — ` +
      'the manifest is stale. Re-run make_audio.sh or the recording will drift from narration.');
  }
} catch (err) {
  console.warn('WARNING: no readable out/audio/scene_durations.json — pacing from planned durations only.');
}

let timelineMs = 0;
const sceneMarks = [];
for (const scene of scenes) {
  const audioSeconds = audioDurations[scene.id];
  const sceneMs = Number(audioSeconds || scene.duration_seconds || 12) * 1000;
  // Absolute pacing: this scene must END when its narration ends on the audio
  // timeline. Whatever the action/dismissal below costs comes out of the pause.
  timelineMs += sceneMs;
  const sceneEndWall = recordingStart + leadInMs + timelineMs;
  await resetZoom(page);
  const sceneAction = scene.action || 'none';
  const urlAtSceneStart = page.url();
  // Only sweep popups on scenes that create a new document. A click scene may
  // have deliberately opened a panel the next scene narrates — the Escape +
  // "Close"-button sweep would tear that state down mid-story.
  if (sceneAction === 'goto' || sceneAction === 'slide') await dismissPopups(page);
  await clearHighlights(page);
  // For navigation scenes, paint the callout AFTER the new document exists —
  // painted here it would flash over the outgoing page and be destroyed.
  if (scene.callout && sceneAction !== 'goto' && sceneAction !== 'slide') await callout(page, scene.callout);

  try {
    const action = scene.action || 'none';
    const timeout = Number(cfg.recording?.action_timeout_ms || 5000);
    // Compare against the page's CURRENT url, not cfg.demo_url: a slide scene
    // navigates to file://, so a later goto back to demo_url must not be skipped.
    if (action === 'goto' && scene.target && scene.target !== page.url()) {
      if (!gotoAllowed(cfg, scene.target)) throw new Error(`read-only: navigation refused to ${scene.target}`);
      await transitionOut(page);
      await page.goto(scene.target, { waitUntil: 'domcontentloaded', timeout: 60000 });
      // Let the SPA settle so charts/tables aren't caught half-hydrated on camera.
      await page.waitForLoadState('networkidle', { timeout: 5000 }).catch(() => {});
      await dismissPopups(page);
      await injectOverlay(page);
    } else if (action === 'search') {
      // The events/alerts search box is a one-way binding: typing updates the URL,
      // but a `?q=` param on direct navigation does NOT hydrate back into the input
      // or re-run the query — confirmed live (box stays empty, table stays unfiltered).
      // Type into the real input instead of relying on the URL.
      const box = page.getByPlaceholder(/search/i).first();
      await box.click({ timeout }).catch(() => {});
      await box.fill(scene.search_text || '', { timeout }).catch(() => {});
      await box.press('Enter', { timeout }).catch(() => {});
      await page.waitForLoadState('networkidle', { timeout: 8000 }).catch(() => {});
      await page.waitForTimeout(3000);
    } else if (action === 'slide') {
      const slidePath = path.resolve(`out/slides/${scene.id}.html`);
      fs.mkdirSync(path.dirname(slidePath), { recursive: true });
      fs.writeFileSync(slidePath, buildSlideHtml(scene));
      await transitionOut(page);
      await page.goto('file://' + slidePath, { waitUntil: 'domcontentloaded' });
      await injectOverlay(page);
    } else if (action === 'scroll') {
      const px = Number(scene.scroll_pixels || 900);
      const steps = 6;
      const budget = Math.max(1500, (sceneEndWall - Date.now()) / 3);
      // Dashboard pages scroll an inner container, not the body — wheel events
      // land wherever the pointer is, so park it over the content pane first.
      await page.mouse.move((cfg.viewport?.width || 1920) * 0.55, (cfg.viewport?.height || 1080) * 0.5).catch(() => {});
      for (let i = 0; i < steps; i++) {
        await page.mouse.wheel(0, px / steps);
        await sleep(Math.min(700, Math.max(250, budget / steps)));
      }
    } else if (action === 'click') {
      let clicked = false;
      if (scene.selector) {
        try { clicked = await safeClick(cfg, page.locator(scene.selector).first(), timeout); } catch (err) {}
      }
      if (!clicked) clicked = await clickByText(page, cfg, scene.selector_text || scene.target, timeout);
      if (!clicked) {
        if (scene.strict) throw new Error(`Could not click scene target: ${scene.selector_text || scene.target}`);
        await callout(page, `${scene.callout || scene.title} (selector not found; continuing demo)`);
      }
    } else if (action === 'highlight') {
      const ok = scene.selector ? await page.locator(scene.selector).first().evaluate(el => { el.classList.add('socket-demo-highlight'); el.scrollIntoView({block: 'center'}); return true; }, null, { timeout: 2000 }).catch(() => false) : await highlightText(page, scene.selector_text || scene.target || scene.title);
      if (!ok && scene.strict) throw new Error(`Could not highlight scene target: ${scene.selector_text || scene.target}`);
    }
  } catch (err) {
    await callout(page, `Scene issue: ${String(err.message || err).slice(0, 120)}`);
    if (scene.strict) throw err;
  }

  // Navigation scenes paint their callout now, on the document that will
  // actually be filmed (see preamble note).
  if (scene.callout && (sceneAction === 'goto' || sceneAction === 'slide')) await callout(page, scene.callout);

  // A click may have started a navigation — let it commit before the URL check
  // below, or an in-flight /settings landing would evade the goBack net. (This
  // narrows the race; a navigation that hasn't STARTED yet can still slip by.)
  if (sceneAction === 'click') {
    await page.waitForLoadState('domcontentloaded', { timeout: 3000 }).catch(() => {});
    // If the click landed on a new document, give it the same popup sweep and
    // overlay re-injection a goto scene gets — a consent/tour modal raised by
    // the destination page would otherwise stay on camera for the whole video.
    if (page.url() !== urlAtSceneStart) {
      await dismissPopups(page);
      await injectOverlay(page);
      if (scene.callout) await callout(page, scene.callout);
    }
  }

  // Safety net: if a click's navigation landed on a settings-ish page, go back.
  if (isReadOnly(cfg) && /\/settings|\/billing|\/api-tokens/i.test(page.url())) {
    console.log('read-only: navigated into a settings page — going back');
    await page.goBack({ waitUntil: 'domcontentloaded' }).catch(() => {});
    await dismissPopups(page);
  }

  const zoomScale = Number(scene.zoom || 0);
  if (zoomScale > 1) await applyZoom(page, zoomScale, scene.zoom_selector, scene.zoom_text || scene.selector_text);

  let pause = sceneEndWall - Date.now();
  if (pause < 800) {
    console.warn(`scene '${scene.id}' actions overran its audio slot by ${((800 - pause) / 1000).toFixed(1)}s`);
    pause = 800;
  }
  const half = Math.max(400, pause / 2);
  await sleep(half);
  if (scene.callout) await callout(page, scene.callout).catch(() => {});
  // Mid-scene scroll: any scene with scroll_pixels gets gentle motion during its
  // second half (dedicated 'scroll' actions already scrolled above).
  let remaining = Math.max(0, sceneEndWall - Date.now());
  if (scene.scroll_pixels && (scene.action || 'none') !== 'scroll') {
    const px = Number(scene.scroll_pixels);
    const steps = 6;
    const stepDelay = Math.min(450, remaining / (steps * 2));
    await page.mouse.move((cfg.viewport?.width || 1920) * 0.55, (cfg.viewport?.height || 1080) * 0.5).catch(() => {});
    for (let i = 0; i < steps; i++) {
      await page.mouse.wheel(0, px / steps).catch(() => {});
      await sleep(stepDelay);
    }
  }
  remaining = Math.max(0, sceneEndWall - Date.now());
  await sleep(remaining);
  sceneMarks.push({ id: scene.id, audio_end_s: +(timelineMs / 1000).toFixed(2), video_end_s: +((Date.now() - recordingStart - leadInMs) / 1000).toFixed(2) });
}
fs.mkdirSync('out/recordings', { recursive: true });
fs.writeFileSync('out/recordings/timing.json', JSON.stringify({
  lead_in_seconds: +(leadInMs / 1000).toFixed(2),
  scenes: sceneMarks,
}, null, 2));

// End clean: clear the callout and hold briefly. (The review-before-sharing
// reminder is printed by validate_output.sh, not shown on camera.)
await page.evaluate(() => { const el = document.getElementById('socket-demo-callout'); if (el) el.remove(); }).catch(() => {});
await sleep(800);
const recordEndMs = Date.now();
if (cdpSession) { try { await cdpSession.send('Page.stopScreencast'); } catch (err) {} }
await page.close();
// Resolve the raw video path BEFORE closing the context: on a persistent
// context, video.path() rejects after close and the copy silently never runs.
const rawPath = video ? await video.path() : null;
await context.close();
if (browser) await browser.close();

if (captureMode === 'cdp') {
  if (!frameManifest.length) {
    console.error('CDP capture produced no frames — nothing to assemble.');
    process.exit(1);
  }
  // Anchor the video timeline to recordingStart so render.sh's lead-in trim
  // and timing.json line up exactly: drop frames older than recordingStart
  // (keep the last of them as the t=0 frame) and stretch the first frame back
  // to recordingStart if the first paint arrived late.
  while (frameManifest.length > 1 && frameManifest[1].wallMs <= recordingStart) frameManifest.shift();
  frameManifest[0].wallMs = Math.min(frameManifest[0].wallMs, recordingStart);
  // concat demuxer: each frame holds until the next; the last holds to the end
  // (idle scenes emit no frames — their last paint simply persists on screen).
  const lines = ['ffconcat version 1.0'];
  for (let i = 0; i < frameManifest.length; i++) {
    const nextMs = i + 1 < frameManifest.length ? frameManifest[i + 1].wallMs : recordEndMs;
    lines.push(`file '${frameManifest[i].file}'`);
    lines.push(`duration ${Math.max(0.001, (nextMs - Math.max(frameManifest[i].wallMs, recordingStart)) / 1000).toFixed(3)}`);
  }
  lines.push(`file '${frameManifest[frameManifest.length - 1].file}'`);
  const concatPath = path.join(framesDir, 'concat.txt');
  fs.writeFileSync(concatPath, lines.join('\n') + '\n');
  fs.writeFileSync(path.join(framesDir, 'manifest.json'),
    JSON.stringify(frameManifest.map((f) => ({ file: path.basename(f.file), wallMs: f.wallMs })), null, 2));
  const finalPath = path.resolve('out/recordings/browser-recording.mp4');
  // Near-lossless CFR intermediate: fps=30 samples the wall-clock frame
  // timeline; render.sh's final encode remains the distribution quality gate.
  // At capture_scale 2 the lanczos downscale is where supersampling pays off.
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', concatPath,
    '-vf', `fps=30,scale=${viewport.width}:${viewport.height}:flags=lanczos`,
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '12', '-pix_fmt', 'yuv420p', finalPath],
    { stdio: 'inherit' });
  fs.rmSync(framesDir, { recursive: true, force: true });
  console.log(`Saved ${finalPath} (${frameManifest.length} captured frames)`);
} else {
  const finalPath = path.resolve('out/recordings/browser-recording.webm');
  fs.copyFileSync(rawPath, finalPath);
  console.log(`Saved ${finalPath}`);
}

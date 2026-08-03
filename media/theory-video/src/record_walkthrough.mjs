import fs from 'fs';
import { chromium } from 'playwright';

// Manual walkthrough recorder: instead of hand-writing `scenes:` in demo.yaml,
// drive the real dashboard (or docs, or any Socket property) yourself in a
// headed, persistent-profile Chrome window, click "Mark scene" before each
// beat you want in the video, then click "Finish recording". Every click,
// scroll, and navigation you make gets logged with the ACTUAL on-screen text,
// so `src/walkthrough_to_scenes.py` can turn it into a `scenes:` list that
// `record-demo.mjs` will replay exactly (same read-only guardrails apply here
// too — mutating-looking controls are not blocked during capture, but they
// will be refused on replay unless read_only is turned off).
let cfg = {};
try { cfg = JSON.parse(fs.readFileSync('out/scenes.json', 'utf8')).config || {}; } catch (err) {}
const browserCfg = cfg.browser || {};
const profileDir = (browserCfg.profile || '~/.local/share/socket-demo-video/browser-profile')
  .replace(/^~(?=\/)/, process.env.HOME);
const startUrl = process.argv[2] || cfg.demo_url || 'https://socket.dev/dashboard/org/socketdev-demo';

fs.mkdirSync(profileDir, { recursive: true });
const ctx = await chromium.launchPersistentContext(profileDir, {
  headless: false,
  channel: browserCfg.channel || 'chrome',
  chromiumSandbox: true,
  viewport: { width: 1920, height: 1080 },
});

const actions = [];
let sceneCount = 0;
let finished = false;

async function attachRecorder(page) {
  await page.exposeFunction('__socketWalkEvent', (payload) => {
    actions.push({ ...payload, t: Date.now(), url: page.url() });
    if (payload.type === 'marker') {
      sceneCount += 1;
      console.log(`\n[scene ${sceneCount} marked]`);
    } else {
      console.log(`  ${payload.type}: ${payload.text || payload.href || payload.deltaY || ''}`);
    }
  });
  await page.exposeFunction('__socketWalkFinish', () => { finished = true; });
  await page.addInitScript(() => {
    if (window.__socketWalkInstalled) return;
    window.__socketWalkInstalled = true;

    const bar = document.createElement('div');
    bar.style.cssText = 'position:fixed;top:12px;right:12px;z-index:2147483647;' +
      'background:#1f1147;color:#fff;font:13px -apple-system,sans-serif;' +
      'padding:10px 12px;border-radius:10px;box-shadow:0 4px 16px rgba(0,0,0,.35);' +
      'display:flex;gap:8px;align-items:center;';
    bar.innerHTML =
      '<span id="socket-walk-count" style="opacity:.85">0 scenes</span>' +
      '<button id="socket-walk-mark" style="cursor:pointer;border:0;border-radius:6px;' +
      'padding:6px 10px;background:#6d28d9;color:#fff;font-weight:600;">Mark scene</button>' +
      '<button id="socket-walk-finish" style="cursor:pointer;border:0;border-radius:6px;' +
      'padding:6px 10px;background:#333;color:#fff;font-weight:600;">Finish</button>';

    const install = () => {
      if (!document.body || document.getElementById('socket-walk-mark')) return;
      document.body.appendChild(bar);
      let count = 0;
      document.getElementById('socket-walk-mark').addEventListener('click', (e) => {
        e.preventDefault(); e.stopPropagation();
        count += 1;
        document.getElementById('socket-walk-count').textContent = `${count} scenes`;
        window.__socketWalkEvent({ type: 'marker' });
      });
      document.getElementById('socket-walk-finish').addEventListener('click', (e) => {
        e.preventDefault(); e.stopPropagation();
        window.__socketWalkFinish();
      });
    };
    document.addEventListener('DOMContentLoaded', install);
    install();

    let lastScrollY = window.scrollY;
    let scrollTimer = null;
    window.addEventListener('scroll', () => {
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(() => {
        const deltaY = window.scrollY - lastScrollY;
        if (Math.abs(deltaY) > 120) {
          window.__socketWalkEvent({ type: 'scroll', deltaY: Math.round(deltaY) });
          lastScrollY = window.scrollY;
        }
      }, 400);
    }, { passive: true });

    document.addEventListener('click', (e) => {
      const el = e.target.closest('a,button,[role="button"],[role="tab"],summary,input[type="submit"]');
      if (!el || bar.contains(el)) return;
      const text = (el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || '')
        .trim().replace(/\s+/g, ' ').slice(0, 80);
      const href = el.tagName === 'A' ? el.href : '';
      let selector = '';
      if (el.id) selector = `#${el.id}`;
      window.__socketWalkEvent({ type: 'click', text, href, selector, tag: el.tagName.toLowerCase() });
    }, true);
  });
}

await attachRecorder(ctx.pages()[0] || await ctx.newPage());
ctx.on('page', attachRecorder);

const page = ctx.pages()[0];
await page.goto(startUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
page.on('framenavigated', (frame) => {
  if (frame === page.mainFrame() && !finished) {
    actions.push({ type: 'goto', url: frame.url(), t: Date.now() });
  }
});

console.log('Recording started. In the Chrome window: browse the product, click');
console.log('"Mark scene" right before each beat you want in the video, then click');
console.log('"Finish" when done (or just close the window).');

const deadline = Date.now() + 30 * 60 * 1000; // 30-minute safety cap
while (!finished && Date.now() < deadline && ctx.pages().length) {
  await new Promise((r) => setTimeout(r, 500));
}

fs.mkdirSync('out/walkthrough', { recursive: true });
fs.writeFileSync('out/walkthrough/recorded.json', JSON.stringify({ startUrl, recordedAt: Date.now(), actions }, null, 2));
console.log(`\nWrote out/walkthrough/recorded.json (${actions.length} events, ${sceneCount} scenes marked)`);
console.log('Next: python src/walkthrough_to_scenes.py --merge demo.yaml');
await ctx.close().catch(() => {});

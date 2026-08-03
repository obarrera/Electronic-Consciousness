import fs from 'fs';
import { chromium } from 'playwright';

// One-time interactive session prime: opens a headed Chrome window on the demo URL
// using the persistent profile. Complete any bot check and the Socket login in the
// window; it closes itself once the dashboard loads. Recordings then run hands-free.
// Reads config from out/scenes.json when present (run generate_content.py first),
// else uses the standard demo-org defaults.
let cfg = {};
try { cfg = JSON.parse(fs.readFileSync('out/scenes.json', 'utf8')).config || {}; } catch (err) {}
const browserCfg = cfg.browser || {};
const profileDir = (browserCfg.profile || '~/.local/share/socket-demo-video/browser-profile')
  .replace(/^~(?=\/)/, process.env.HOME);
const target = cfg.demo_url && !String(cfg.demo_url).startsWith('file:')
  ? cfg.demo_url
  : 'https://socket.dev/dashboard/org/socketdev-demo';

fs.mkdirSync(profileDir, { recursive: true });
const ctx = await chromium.launchPersistentContext(profileDir, {
  headless: false,
  channel: browserCfg.channel || 'chrome',
  chromiumSandbox: true,
  viewport: { width: 1920, height: 1080 },
});
const page = ctx.pages()[0] || await ctx.newPage();
await page.goto(target, { waitUntil: 'domcontentloaded', timeout: 60000 });
console.log('Complete any verification/login in the Chrome window...');
const deadline = Date.now() + 5 * 60 * 1000;
// Dashboard targets need a real login; docs/blog/package targets just need to
// be past the Cloudflare challenge — waiting 5 minutes for "/dashboard/" on a
// docs URL would always time out.
const needsLogin = target.includes('/dashboard/');
let primed = false;
while (Date.now() < deadline) {
  const url = page.url();
  if (needsLogin) {
    if (url.includes('/dashboard/') && !url.includes('/auth/')) { primed = true; break; }
  } else {
    const title = await page.title().catch(() => '');
    if (!/just a moment|attention required/i.test(title) && url.startsWith('http')) { primed = true; break; }
  }
  await page.waitForTimeout(2000);
}
if (primed) {
  await page.waitForTimeout(3000);
  console.log('SESSION PRIMED:', page.url());
} else {
  console.log('Timed out waiting for login. Re-run: npm run prime');
}
await ctx.close();

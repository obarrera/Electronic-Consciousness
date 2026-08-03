import fs from 'fs';
import { chromium } from 'playwright';

// READ-ONLY site mapper for the Socket demo-org dashboard.
// Visits pages, reads titles and same-org links, writes out/sitemap.json + .md.
// It NEVER clicks buttons or forms — navigation is by URL only, and /settings
// URLs are excluded entirely.
// Root can be ANY Socket property: the demo-org dashboard, docs.socket.dev,
// socket.dev/blog, a package page family, etc. Pass it as argv[2] or it falls
// back to the project's demo_url, then the demo org.
// Minimal demo.yaml reader (no YAML dependency): `npm run map` is documented
// to work BEFORE generate_content.py has ever produced out/scenes.json, so the
// crawl settings must be readable straight from the project config.
function parseDemoYaml(text) {
  const out = { crawl: {} };
  const demoUrl = text.match(/^demo_url:\s*['"]?(\S+?)['"]?\s*$/m);
  if (demoUrl) out.demo_url = demoUrl[1];
  const crawlBlock = text.match(/^crawl:\s*\n((?:[ \t]+\S[^\n]*\n?|[ \t]*\n)*)/m);
  if (crawlBlock) {
    const block = crawlBlock[1];
    const num = (k) => { const m = block.match(new RegExp('^[ \\t]+' + k + ':\\s*(\\d+)', 'm')); return m ? Number(m[1]) : undefined; };
    if (num('max_pages') !== undefined) out.crawl.max_pages = num('max_pages');
    if (num('max_depth') !== undefined) out.crawl.max_depth = num('max_depth');
    const ew = block.match(/^[ \t]+events_window:\s*['"]?(\w+)/m);
    if (ew) out.crawl.events_window = ew[1];
    const seedsMatch = block.match(/^[ \t]+seeds:\s*\n((?:[ \t]+-[^\n]*\n?)*)/m);
    if (seedsMatch) {
      out.crawl.seeds = [...seedsMatch[1].matchAll(/-\s*['"]?([^'"\n]+?)['"]?\s*$/gm)].map((m) => m[1].trim());
    }
  }
  return out;
}

let cfg = {};
try { cfg = JSON.parse(fs.readFileSync('out/scenes.json', 'utf8')).config || {}; } catch (err) {}
if (!cfg.demo_url) {
  try { cfg = { ...parseDemoYaml(fs.readFileSync('demo.yaml', 'utf8')), ...cfg }; } catch (err) {}
}
let START = (process.argv[2] || String(cfg.demo_url || '')).split('#')[0].replace(/\/$/, '');
if (!START || !START.startsWith('https://')) START = 'https://socket.dev/dashboard/org/socketdev-demo';
// Crawl root: an org-dashboard crawl stays inside that org; any other property
// (a docs page, blog, package family) crawls SAME-ORIGIN, so sibling pages are
// reachable even when demo_url points at one deep page — a full-URL prefix on
// a subpath would pin the crawl to a single page and starve max_depth.
const orgMatch = START.match(/^(https:\/\/socket\.dev\/dashboard\/org\/[^\/?#]+)/);
const ORG = orgMatch ? orgMatch[1] : new URL(START).origin;
const profileDir = process.env.HOME + '/.local/share/socket-demo-video/browser-profile';

// Optional `crawl:` block in demo.yaml controls how the mapper explores a
// property, without touching this file: seeds (start URLs, absolute or
// `/path` relative to ORG), max_pages, max_depth (hops from a seed before a
// link stops getting enqueued), include/exclude (extra regexes on top of the
// built-in settings/billing block), and capture_clickables (also list
// non-link controls — buttons/tabs — per page, useful for `click` scenes).
const crawlCfg = cfg.crawl || {};
const MAX_PAGES = Number(crawlCfg.max_pages) || 22;
const MAX_DEPTH = crawlCfg.max_depth != null ? Number(crawlCfg.max_depth) : Infinity;
const includeRe = (crawlCfg.include || []).length ? new RegExp((crawlCfg.include || []).join('|'), 'i') : null;
const excludeRe = (crawlCfg.exclude || []).length ? new RegExp((crawlCfg.exclude || []).join('|'), 'i') : null;
const captureClickables = crawlCfg.capture_clickables !== false;

// Verified accepted Events time-window values (from the dashboard's own filter
// dropdown): 5m, 15m, 30m, 1h, 6h, 24h, 7d, 30d. There is no native "1 day" or
// "2 days" — other values silently redirect to the default (1h), so an invalid
// crawl.events_window would look like it worked but show the wrong window.
const EVENTS_WINDOWS = new Set(['5m', '15m', '30m', '1h', '6h', '24h', '7d', '30d']);
const eventsWindow = EVENTS_WINDOWS.has(crawlCfg.events_window) ? crawlCfg.events_window : '30d';

const isDashboardOrg = Boolean(orgMatch);
const defaultSeeds = isDashboardOrg
  ? [
      `${ORG}`,
      `${ORG}/repositories`,
      `${ORG}/dependencies`,
      `${ORG}/alerts`,
      `${ORG}/scans`,
      `${ORG}/threat-intel`,
      `${ORG}/threat-intel?tab=campaigns`,
      `${ORG}/events?tp=${eventsWindow}`,
    ]
  : [START];
const seeds = (crawlCfg.seeds || []).length
  ? crawlCfg.seeds.map((s) => (s.startsWith('/') ? `${ORG}${s}` : s))
  : defaultSeeds;

const skip = (url) => {
  // Boundary after the root prefix: /org/acme must not admit /org/acme-staging.
  if (!(url === ORG || url.startsWith(ORG + '/') || url.startsWith(ORG + '?') || url.startsWith(ORG + '#'))) return true;
  if (/\/settings|\/billing|\/api-tokens|\/members|\/integrations\/manage/i.test(url)) return true;
  if (excludeRe && excludeRe.test(url)) return true;
  if (includeRe && !includeRe.test(url)) return true;
  return false;
};

const ctx = await chromium.launchPersistentContext(profileDir, {
  headless: false,
  channel: 'chrome',
  chromiumSandbox: true,
  viewport: { width: 1920, height: 1080 },
});
const page = ctx.pages()[0] || await ctx.newPage();

const visited = new Map();
const queue = seeds.map((url) => ({ url, depth: 0 }));
while (queue.length && visited.size < MAX_PAGES) {
  const { url, depth } = queue.shift();
  const key = url.split('#')[0];
  if (visited.has(key) || skip(key)) continue;
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.waitForTimeout(3500);
    const title = (await page.title()).trim();
    const h1 = (await page.locator('h1').first().innerText({ timeout: 2000 }).catch(() => '')).trim();
    const links = await page.$$eval('a[href]', (as) =>
      as.map((a) => a.href).filter((h, i, arr) => arr.indexOf(h) === i)
    );
    const orgLinks = links.filter((l) => !skip(l.split('#')[0]));
    let clickable_controls = [];
    if (captureClickables) {
      clickable_controls = await page.$$eval(
        'button, [role="button"], [role="tab"]',
        (els) => els
          .map((el) => (el.innerText || el.getAttribute('aria-label') || el.title || '').trim().replace(/\s+/g, ' '))
          // Account/avatar controls often append the logged-in operator's
          // initial(s) to the label (e.g. "Open user menu O") — strip a
          // trailing 1-2 letter token so a crawl never bakes in who ran it.
          .map((t) => t.replace(/\s+[A-Za-z]{1,2}$/, ''))
          .filter((t) => t && t.length <= 60 && !/^(save|delete|create|edit|enable|disable|invite|rotate|revoke|resolve)\b/i.test(t))
          .filter((t, i, arr) => arr.indexOf(t) === i)
          .slice(0, 25)
      ).catch(() => []);
    }
    visited.set(key, { url: key, title, heading: h1, links: orgLinks.length, clickable_controls });
    if (depth < MAX_DEPTH) {
      // Enqueue a few interesting detail pages per section (alerts, campaigns).
      for (const l of orgLinks) {
        const lk = l.split('#')[0];
        const interesting = isDashboardOrg
          ? /\/alert\/|\/threat-intel\/campaign\/|\/repositories\/|\/events/.test(lk)
          : true; // generic roots (docs, blog, package pages): breadth-first under the root
        if (!visited.has(lk) && !queue.some((q) => q.url === lk) && interesting) {
          queue.push({ url: lk, depth: depth + 1 });
        }
      }
    }
    console.log(`mapped (depth ${depth}): ${h1 || title} — ${key}`);
  } catch (err) {
    console.log(`skip (${String(err.message).slice(0, 60)}): ${key}`);
  }
}
await ctx.close();

const pages = [...visited.values()];
fs.mkdirSync('out', { recursive: true });
fs.writeFileSync('out/sitemap.json', JSON.stringify({ org: ORG, generated_pages: pages.length, pages }, null, 2));
fs.writeFileSync('out/sitemap.md',
  `# socketdev-demo dashboard sitemap (read-only crawl)\n\n` +
  pages.map((p) => `- **${p.heading || p.title}** — ${p.url} (${p.links} org links)` +
    (p.clickable_controls && p.clickable_controls.length ? `\n  - controls: ${p.clickable_controls.join(', ')}` : '')
  ).join('\n') + '\n');
console.log(`\nWrote out/sitemap.json + out/sitemap.md (${pages.length} pages)`);

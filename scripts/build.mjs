import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const contentDir = path.join(rootDir, "content");
const publicDir = path.join(rootDir, "public");
const srcDir = path.join(rootDir, "src");
const distDir = path.join(rootDir, "dist");

const chapters = [
  {
    number: 1,
    file: "01-architecture-overview.md",
    title: "模型架构总览",
    fullTitle: "MiniMind 大模型神经网络结构与算法原理教程",
  },
  {
    number: 2,
    file: "02-basic-components.md",
    title: "基础组件",
    fullTitle: "基础组件 - Embedding & RMSNorm & RoPE",
    example: "02-basic-components-demo.py",
  },
  {
    number: 3,
    file: "03-attention.md",
    title: "注意力机制",
    fullTitle: "注意力机制 Attention",
    example: "03-attention-demo.py",
  },
  {
    number: 4,
    file: "04-mlp-moe.md",
    title: "前馈网络 MLP & MoE",
    fullTitle: "前馈网络 MLP & MoE",
    example: "04-mlp-moe-demo.py",
  },
  {
    number: 5,
    file: "05-transformer-block.md",
    title: "Transformer Block",
    fullTitle: "Transformer Block 与整体前向",
    example: "05-transformer-block-demo.py",
  },
  {
    number: 6,
    file: "06-train-pretrain-sft.md",
    title: "Pretrain & SFT",
    fullTitle: "训练算法 - Pretrain & SFT",
    example: "06-train-pretrain-sft-demo.py",
  },
  {
    number: 7,
    file: "07-train-dpo.md",
    title: "DPO 偏好优化",
    fullTitle: "训练算法 - DPO 偏好优化",
    example: "07-dpo-demo.py",
  },
  {
    number: 8,
    file: "08-train-rl-ppo-grpo.md",
    title: "PPO & GRPO",
    fullTitle: "训练算法 - PPO & GRPO 强化学习",
    example: "08-rl-demo.py",
  },
  {
    number: 9,
    file: "09-lora.md",
    title: "LoRA 低秩适配",
    fullTitle: "LoRA 低秩适配",
    example: "09-lora-demo.py",
  },
  {
    number: 10,
    file: "10-inference-generation.md",
    title: "推理生成算法",
    fullTitle: "推理生成算法",
    example: "10-generation-demo.py",
  },
].map((chapter) => ({
  ...chapter,
  slug: chapter.file.replace(/\.md$/, ""),
}));

const chapterUrlByFile = new Map(
  chapters.map((chapter) => [chapter.file, `/chapters/${chapter.slug}/`]),
);

const exampleUrlByFile = new Map(
  chapters
    .filter((chapter) => chapter.example)
    .map((chapter) => [chapter.example, `/examples/${chapter.example.replace(/\.py$/, "")}/`]),
);

marked.setOptions({
  async: false,
  breaks: false,
  gfm: true,
});

await fs.rm(distDir, { force: true, recursive: true });
await fs.mkdir(path.join(distDir, "assets"), { recursive: true });
await copyPublic();
await fs.copyFile(path.join(srcDir, "site.css"), path.join(distDir, "assets", "site.css"));

await writePage("index.html", renderHomePage());
await writePage("404.html", renderNotFoundPage());
await writeRobots();
await writeSitemap();
await writeChapterPages();
await writeExamplePages();
await enhanceCopiedStaticSites();

console.log(`Built static site into ${path.relative(rootDir, distDir)}/`);

async function copyPublic() {
  await fs.cp(publicDir, distDir, {
    recursive: true,
    filter: (source) => {
      const name = path.basename(source);
      return ![".git", ".gitignore", ".gitattributes", "README.md", "serve.py"].includes(name);
    },
  });
}

async function writeChapterPages() {
  for (const [index, chapter] of chapters.entries()) {
    const markdown = await fs.readFile(path.join(contentDir, "chapters", chapter.file), "utf8");
    const html = markdownToHtml(preprocessMarkdown(markdown));
    const { html: articleHtml, toc } = addHeadingAnchors(html);
    const previous = chapters[index - 1];
    const next = chapters[index + 1];
    const exampleLink = chapter.example
      ? `<a class="button secondary" href="${exampleUrlByFile.get(chapter.example)}">查看本章示例代码</a>`
      : "";
    const pager = `
      <nav class="pager" aria-label="章节翻页">
        ${previous ? `<a href="/chapters/${previous.slug}/">上一章：${escapeHtml(previous.title)}</a>` : "<span></span>"}
        ${next ? `<a class="next" href="/chapters/${next.slug}/">下一章：${escapeHtml(next.title)}</a>` : "<span></span>"}
      </nav>
    `;

    await writePage(
      path.join("chapters", chapter.slug, "index.html"),
      renderShell({
        title: `${chapter.number}. ${chapter.fullTitle}`,
        description: "MiniMind 大模型结构、训练和推理教程。",
        active: "tutorial",
        body: renderDocLayout({
          activeChapter: chapter.slug,
          article: `${articleHtml}<div class="button-row">${exampleLink}</div>${pager}`,
          toc,
        }),
      }),
    );
  }
}

async function writeExamplePages() {
  const downloadsDir = path.join(distDir, "downloads", "examples");
  await fs.mkdir(downloadsDir, { recursive: true });

  for (const chapter of chapters.filter((item) => item.example)) {
    const source = path.join(contentDir, "examples", chapter.example);
    const code = await fs.readFile(source, "utf8");
    await fs.copyFile(source, path.join(downloadsDir, chapter.example));

    const body = `
      <h1>${escapeHtml(chapter.fullTitle)}：示例代码</h1>
      <p>这个页面展示原始 Python 示例脚本，方便在线阅读；也可以下载后在本地使用 <code>python ${escapeHtml(chapter.example)}</code> 运行。</p>
      <div class="button-row">
        <a class="button" href="/downloads/examples/${encodeURIComponent(chapter.example)}">下载 Python 文件</a>
        <a class="button secondary" href="/chapters/${chapter.slug}/">返回对应章节</a>
      </div>
      <pre><code>${escapeHtml(code)}</code></pre>
    `;

    await writePage(
      path.join("examples", chapter.example.replace(/\.py$/, ""), "index.html"),
      renderShell({
        title: `${chapter.title}示例代码`,
        description: "MiniMind 教程配套 Python 示例代码。",
        active: "examples",
        body: renderDocLayout({
          activeChapter: chapter.slug,
          article: body,
          toc: [],
        }),
      }),
    );
  }
}

function renderHomePage() {
  const chapterCards = chapters
    .map(
      (chapter) => `
        <a class="card" href="/chapters/${chapter.slug}/">
          <span class="tag">${String(chapter.number).padStart(2, "0")}</span>
          <h3>${escapeHtml(chapter.title)}</h3>
          <p>${escapeHtml(chapter.fullTitle)}</p>
          <small>阅读章节</small>
        </a>
      `,
    )
    .join("");

  const exampleCards = chapters
    .filter((chapter) => chapter.example)
    .map(
      (chapter) => `
        <a class="card" href="${exampleUrlByFile.get(chapter.example)}">
          <span class="tag teal">Python</span>
          <h3>${escapeHtml(chapter.example)}</h3>
          <p>对应第 ${chapter.number} 章：${escapeHtml(chapter.title)}</p>
          <small>查看示例</small>
        </a>
      `,
    )
    .join("");

  return renderShell({
    title: "MiniMind 系列学习教程",
    description: "MiniMind 大模型结构、训练、推理和多模态系列教程。",
    active: "home",
    body: `
      <main class="home-layout">
        <section class="hero">
          <div>
            <div class="eyebrow">Static Tutorial Library</div>
            <h1>MiniMind 系列学习教程</h1>
            <p>这是一个纯静态教程站：主教程由 Markdown 构建为 HTML，配套 Python 示例、课程静态站、MiniMind-V 和 MiniMind-O 专题教程都可以通过同一个 Cloudflare 静态部署访问。</p>
            <div class="button-row">
              <a class="button" href="/chapters/01-architecture-overview/">开始阅读主教程</a>
              <a class="button secondary" href="/course/">打开学习课程站</a>
            </div>
          </div>
          <div class="hero-media">
            <img src="/course/images/LLM-structure.jpg" alt="MiniMind Decoder-only 结构图">
          </div>
        </section>

        <section class="section">
          <h2>专题入口</h2>
          <div class="grid cols-3">
            <a class="card" href="/course/">
              <span class="tag">Course</span>
              <h3>MiniMind 学习课程站</h3>
              <p>训练流程、数据样例、实验曲线、资源图谱和部署入口。</p>
              <small>打开课程站</small>
            </a>
            <a class="card" href="/minimind-v/">
              <span class="tag teal">Vision</span>
              <h3>MiniMind-V 视觉语言教程</h3>
              <p>VLM 架构、图文数据、训练复现、推理部署和评估实验。</p>
              <small>打开 MiniMind-V</small>
            </a>
            <a class="card" href="/minimind-o/">
              <span class="tag">Omni</span>
              <h3>MiniMind-O Omni 教程</h3>
              <p>语音视觉输入、Thinker/Talker、数据训练、流式推理和 WebUI。</p>
              <small>打开 MiniMind-O</small>
            </a>
          </div>
        </section>

        <section class="section">
          <h2>主教程章节</h2>
          <div class="grid cols-3">${chapterCards}</div>
        </section>

        <section class="section">
          <h2>示例代码</h2>
          <div class="grid cols-3">${exampleCards}</div>
        </section>
      </main>
    `,
  });
}

function renderNotFoundPage() {
  return renderShell({
    title: "页面未找到",
    description: "MiniMind 教程站 404 页面。",
    active: "",
    body: `
      <main class="home-layout">
        <section class="hero">
          <div>
            <div class="eyebrow">404</div>
            <h1>页面未找到</h1>
            <p>这个地址没有对应的静态页面。可以回到首页，或者从章节目录继续阅读。</p>
            <div class="button-row">
              <a class="button" href="/">返回首页</a>
              <a class="button secondary" href="/chapters/01-architecture-overview/">阅读主教程</a>
            </div>
          </div>
        </section>
      </main>
    `,
  });
}

function renderDocLayout({ activeChapter, article, toc }) {
  return `
    <main class="layout">
      <aside class="sidebar">
        <div class="sidebar-inner">
          <div class="nav-block">
            <p class="nav-title">主教程</p>
            ${chapters
              .map(
                (chapter) => `
                  <a class="side-link ${chapter.slug === activeChapter ? "active" : ""}" href="/chapters/${chapter.slug}/">
                    ${String(chapter.number).padStart(2, "0")} ${escapeHtml(chapter.title)}
                    <span>${escapeHtml(chapter.fullTitle)}</span>
                  </a>
                `,
              )
              .join("")}
          </div>
          <div class="nav-block">
            <p class="nav-title">专题站</p>
            <a class="site-link" href="/course/">学习课程站<span>训练流程 / 数据样例</span></a>
            <a class="site-link" href="/minimind-v/">MiniMind-V<span>视觉语言模型专题</span></a>
            <a class="site-link" href="/minimind-o/">MiniMind-O<span>Omni 交互模型专题</span></a>
          </div>
        </div>
      </aside>
      <article class="article">
        <div class="article-inner">${article}</div>
      </article>
      <aside class="toc">
        <div class="toc-inner">
          <p class="nav-title">本页目录</p>
          ${toc.length ? toc.map((item) => `<a class="level-${item.level}" href="#${item.id}">${escapeHtml(item.text)}</a>`).join("") : "<p style=\"color: var(--muted); margin: 0;\">示例代码页</p>"}
        </div>
      </aside>
    </main>
  `;
}

function renderShell({ title, description, active, body }) {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${escapeHtml(description)}">
  <title>${escapeHtml(title)} | learn-minimind</title>
  <link rel="stylesheet" href="/assets/site.css">
</head>
<body>
  <header class="site-header">
    <div class="topbar">
      <a class="brand" href="/">
        <span class="brand-mark">M</span>
        <span>MiniMind 教程<small>静态学习站</small></span>
      </a>
      <nav class="topnav" aria-label="主导航">
        <a class="${active === "home" ? "active" : ""}" href="/">首页</a>
        <a class="${active === "tutorial" ? "active" : ""}" href="/chapters/01-architecture-overview/">主教程</a>
        <a class="${active === "examples" ? "active" : ""}" href="/examples/02-basic-components-demo/">示例代码</a>
        <a href="/course/">课程站</a>
        <a href="/minimind-v/">MiniMind-V</a>
        <a href="/minimind-o/">MiniMind-O</a>
      </nav>
    </div>
  </header>
  ${body}
  <footer class="footer">
    <div class="footer-inner">learn-minimind 是纯静态教程站，可通过 Cloudflare Workers Static Assets 或 Cloudflare Pages 部署。</div>
  </footer>
</body>
</html>`;
}

function preprocessMarkdown(markdown) {
  let next = markdown;

  for (const [file, url] of chapterUrlByFile) {
    next = next.replaceAll(`(chapters/${file})`, `(${url})`);
    next = next.replaceAll(`(${file})`, `(${url})`);
  }

  for (const [file, url] of exampleUrlByFile) {
    next = next.replaceAll(`(examples/${file})`, `(${url})`);
    next = next.replaceAll(`(${file})`, `(${url})`);
  }

  next = next.replace(
    /file:\/\/\/home\/zhangxm\/model_minimind\/([^)\s#]+)(#[^) \n]+)?/g,
    (_, filePath, hash = "") => `https://github.com/jingyaogong/minimind/blob/master/${filePath}${hash}`,
  );

  next = next.replace(
    /\((?:\.\.\/){2}([^)]+)\)/g,
    (_, filePath) => `(https://github.com/jingyaogong/minimind/tree/master/${filePath.replace(/\/$/, "")})`,
  );

  return next;
}

function markdownToHtml(markdown) {
  return marked.parse(markdown);
}

function addHeadingAnchors(html) {
  const used = new Map();
  const toc = [];
  let fallback = 0;
  const withAnchors = html.replace(/<h([1-3])>([\s\S]*?)<\/h\1>/g, (match, levelText, inner) => {
    const level = Number(levelText);
    const text = stripTags(inner).trim();
    let id = slugify(text);
    if (!id) {
      fallback += 1;
      id = `section-${fallback}`;
    }
    const count = used.get(id) ?? 0;
    used.set(id, count + 1);
    if (count > 0) {
      id = `${id}-${count + 1}`;
    }
    if (level > 1) {
      toc.push({ id, level, text });
    }
    return `<h${level} id="${id}">${inner}</h${level}>`;
  });

  return { html: withAnchors, toc };
}

async function enhanceCopiedStaticSites() {
  await enhanceHtmlDirectory(path.join(distDir, "minimind-o"), (html) => {
    if (html.includes('href="/"') || !html.includes('<div class="header-actions">')) {
      return html;
    }
    return html.replace(
      '<div class="header-actions">',
      '<div class="header-actions">\n        <a class="icon-button" href="/" title="返回总教程">总教程</a>\n        <a class="icon-button" href="/minimind-v/" title="打开 MiniMind-V 教程">MiniMind-V</a>',
    );
  });

  await enhanceHtmlDirectory(path.join(distDir, "minimind-v"), (html) => {
    if (html.includes('href="/"') || !html.includes("</nav>")) {
      return html;
    }
    return html.replace(
      "</nav>",
      '        <a href="/minimind-o/">MiniMind-O</a>\n        <a href="/">总教程</a>\n      </nav>',
    );
  });
}

async function enhanceHtmlDirectory(directory, transform) {
  let entries = [];
  try {
    entries = await fs.readdir(directory, { withFileTypes: true });
  } catch {
    return;
  }

  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      await enhanceHtmlDirectory(fullPath, transform);
      continue;
    }
    if (!entry.name.endsWith(".html")) {
      continue;
    }
    const html = await fs.readFile(fullPath, "utf8");
    await fs.writeFile(fullPath, transform(html));
  }
}

async function writeRobots() {
  await writePage(
    "robots.txt",
    ["User-agent: *", "Allow: /", "Sitemap: /sitemap.xml", ""].join("\n"),
  );
}

async function writeSitemap() {
  const siteUrl = (process.env.SITE_URL || "https://learn-minimind.pages.dev").replace(/\/$/, "");
  const urls = [
    "/",
    "/course/",
    "/minimind-v/",
    "/minimind-o/",
    ...chapters.map((chapter) => `/chapters/${chapter.slug}/`),
    ...chapters.filter((chapter) => chapter.example).map((chapter) => `/examples/${chapter.example.replace(/\.py$/, "")}/`),
  ];
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((url) => `  <url><loc>${escapeHtml(`${siteUrl}${url}`)}</loc></url>`).join("\n")}
</urlset>
`;
  await writePage("sitemap.xml", xml);
}

async function writePage(relativePath, html) {
  const outputPath = path.join(distDir, relativePath);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, html);
}

function stripTags(html) {
  return html.replace(/<[^>]*>/g, "");
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/&amp;/g, "and")
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

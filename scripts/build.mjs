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

const siteOwner = {
  publicName: "微信公众号《有机系统》作者",
  email: "mssnzxm@126.com",
  updatedAt: "2026年7月10日",
};

const legalPages = [
  {
    slug: "privacy-policy",
    title: "隐私政策",
    navLabel: "隐私政策",
    description: "本站隐私政策，说明 Google 广告 Cookie、第三方广告、联系信息与日志数据的处理方式。",
  },
  {
    slug: "about",
    title: "关于我们",
    navLabel: "关于我们",
    description: "关于 MiniMind 学习教程与微信公众号《有机系统》作者的说明。",
  },
  {
    slug: "contact",
    title: "联系方式",
    navLabel: "联系方式",
    description: "联系微信公众号《有机系统》作者，反馈教程问题、版权问题或合作事项。",
  },
  {
    slug: "disclaimer",
    title: "免责声明",
    navLabel: "免责声明",
    description: "本站内容来源、教程风险、第三方链接、广告与版权免责声明。",
  },
];

const topicLinks = [
  {
    id: "home",
    href: "/",
    label: "总教程",
  },
  {
    id: "course",
    href: "/course/",
    label: "课程站",
  },
  {
    id: "minimind-v",
    href: "/minimind-v/",
    label: "MiniMind-V",
  },
  {
    id: "minimind-o",
    href: "/minimind-o/",
    label: "MiniMind-O",
  },
];

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
await writeLegalPages();
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

async function writeLegalPages() {
  for (const page of legalPages) {
    await writePage(
      path.join(page.slug, "index.html"),
      renderLegalPage(page),
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
            <div class="eyebrow">Learning Library</div>
            <h1>MiniMind 系列学习教程</h1>
            <p>围绕 MiniMind 主线教程、配套 Python 示例、课程导览、MiniMind-V 和 MiniMind-O 专题内容，整理出一条从模型结构到训练推理的学习路径。</p>
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
              <p>训练流程、数据样例、实验曲线、资源图谱和实践入口。</p>
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
    description: "MiniMind 教程页面。",
    active: "",
    body: `
      <main class="home-layout">
        <section class="hero">
          <div>
            <div class="eyebrow">404</div>
            <h1>页面未找到</h1>
            <p>这个地址没有对应的内容。可以回到首页，或者从章节目录继续阅读。</p>
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

function renderLegalPage(page) {
  return renderShell({
    title: page.title,
    description: page.description,
    active: page.slug,
    body: `
      <main class="legal-layout">
        <article class="article legal-article">
          <div class="article-inner">
            ${renderLegalContent(page.slug)}
          </div>
        </article>
      </main>
    `,
  });
}

function renderLegalContent(slug) {
  if (slug === "privacy-policy") {
    return `
      <p class="eyebrow">Site Policy</p>
      <h1>隐私政策</h1>
      <p class="legal-updated">最后更新：${siteOwner.updatedAt}</p>
      <p>本隐私政策适用于本站（MiniMind 学习教程及其相关专题页面）。本站由${siteOwner.publicName}整理维护，是个人学习笔记与教程站点，不是 MiniMind、Google 或任何第三方项目的官方网站。</p>

      <h2>我们可能收集的信息</h2>
      <ul>
        <li>访问日志：托管服务、防火墙或统计服务可能记录 IP 地址、浏览器类型、访问时间、来源页面、访问路径等基础日志，用于安全、排障和站点维护。</li>
        <li>主动提交的信息：当你通过电子邮件联系本站时，我们会收到你的邮箱地址、邮件内容以及你主动提供的其他信息。</li>
        <li>Cookie 与本地存储：本站本身不要求注册登录。若未来启用广告、统计或交互功能，相关服务可能使用 Cookie、本地存储、像素标签或类似技术。</li>
      </ul>

      <h2>Google 广告 Cookie</h2>
      <p>本站计划接入 Google AdSense。启用后，Google 作为第三方广告供应商，可能会使用 Cookie 投放广告。Google 使用广告 Cookie，可以根据用户访问本站和互联网上其他网站的情况，为用户投放个性化或非个性化广告。</p>
      <p>第三方广告供应商和广告网络也可能使用 Cookie、JavaScript、网络信标或类似技术来衡量广告效果、限制广告展示频次并展示更相关的广告。本站无法直接控制这些第三方 Cookie 的具体设置。</p>
      <p>你可以访问 <a href="https://myadcenter.google.com/" rel="noopener">Google 我的广告中心</a> 管理个性化广告设置，也可以阅读 <a href="https://policies.google.com/technologies/ads?hl=zh-CN" rel="noopener">Google 广告技术说明</a>了解 Google 如何使用 Cookie 投放广告。</p>

      <h2>Consent 与地区要求</h2>
      <p>如果本站向欧洲经济区、英国、瑞士或其他要求事先同意的地区用户展示 Google 广告，我们会根据适用规则使用 Google 认可的同意管理方式，取得必要同意后再使用 Cookie 或本地存储进行个性化广告、广告衡量或相关用途。</p>

      <h2>信息使用方式</h2>
      <ul>
        <li>维护、保护和改进本站内容与访问体验。</li>
        <li>回复你的邮件、处理勘误、版权反馈或合作咨询。</li>
        <li>在广告启用后，根据 Google AdSense 及相关广告服务规则展示广告、衡量广告效果和防范欺诈。</li>
      </ul>

      <h2>信息共享</h2>
      <p>本站不会出售你的个人信息。为托管网站、发送邮件、展示广告、统计访问或满足法律要求，相关信息可能由托管服务商、邮件服务商、Google 等第三方服务提供商按其政策处理。</p>

      <h2>第三方链接</h2>
      <p>本站教程会引用 GitHub、论文、模型平台、文档站等第三方链接。访问这些网站时，请阅读对方的隐私政策和服务条款；本站不对第三方网站的数据处理行为负责。</p>

      <h2>儿童隐私</h2>
      <p>本站面向一般技术学习者，不以 13 岁以下儿童为目标用户。如你认为本站不慎收集了儿童个人信息，请通过下方邮箱联系删除。</p>

      <h2>联系我们</h2>
      <p>隐私相关问题、数据删除请求、广告 Cookie 疑问或政策反馈，可以发送邮件至 <a href="mailto:${siteOwner.email}">${siteOwner.email}</a>。</p>
    `;
  }

  if (slug === "about") {
    return `
      <p class="eyebrow">About</p>
      <h1>关于我们</h1>
      <p class="legal-updated">最后更新：${siteOwner.updatedAt}</p>
      <p>本站是${siteOwner.publicName}整理的个人学习笔记与教程，主要围绕 MiniMind、MiniMind-V、MiniMind-O 及相关大模型训练、推理、多模态实践内容展开。</p>
      <p>本站不是 MiniMind 项目、相关模型平台、论文作者或任何第三方机构的官方网站。页面中的项目名称、仓库名称、模型名称和图片资料，仅用于学习、研究、说明与引用。</p>

      <h2>本站定位</h2>
      <ul>
        <li>把分散的 README、源码、训练脚本、数据格式和实践步骤整理成更容易阅读的中文教程。</li>
        <li>保留关键外部资料链接，方便读者继续回到原始项目、论文和文档核对细节。</li>
        <li>持续修正教程中的表达、步骤和链接，让学习路径更清楚。</li>
      </ul>

      <h2>作者信息</h2>
      <p>作者长期通过微信公众号《有机系统》记录技术学习、工程实践和系统思考。本站是这些公开学习资料的一部分，目的是帮助自己和读者更高效地复盘、运行和理解相关项目。</p>

      <h2>联系方式</h2>
      <p>勘误、版权问题、内容建议或合作沟通，请发送邮件至 <a href="mailto:${siteOwner.email}">${siteOwner.email}</a>。</p>
    `;
  }

  if (slug === "contact") {
    return `
      <p class="eyebrow">Contact</p>
      <h1>联系方式</h1>
      <p class="legal-updated">最后更新：${siteOwner.updatedAt}</p>
      <p>欢迎通过邮件反馈教程错误、链接失效、版权问题、引用建议或合作事项。</p>

      <h2>邮件联系</h2>
      <p>邮箱：<a class="contact-link" href="mailto:${siteOwner.email}">${siteOwner.email}</a></p>
      <p>为了更快定位问题，建议在邮件中写清页面地址、问题位置、期望修改方式，以及必要的截图或原始资料链接。</p>

      <h2>微信公众号</h2>
      <p>你也可以关注微信公众号《有机系统》，通过公众号文章、留言或相关入口了解作者的更多学习笔记。</p>

      <h2>处理说明</h2>
      <ul>
        <li>教程勘误和链接失效会优先处理。</li>
        <li>版权、署名、转载、删除等请求请附上权利说明和具体页面地址。</li>
        <li>本站是个人维护项目，回复时间可能受工作与学习安排影响。</li>
      </ul>
    `;
  }

  return `
    <p class="eyebrow">Disclaimer</p>
    <h1>免责声明</h1>
    <p class="legal-updated">最后更新：${siteOwner.updatedAt}</p>
    <p>本站内容由${siteOwner.publicName}基于公开资料、个人实践和学习记录整理，仅供学习、研究和技术交流参考。</p>

    <h2>非官方网站</h2>
    <p>本站不是 MiniMind、MiniMind-V、MiniMind-O、Google AdSense、GitHub、Hugging Face、ModelScope 或其他第三方项目与平台的官方网站。相关名称、商标、截图、图片和链接归其各自权利人所有。</p>

    <h2>内容准确性</h2>
    <p>本站会尽力保持内容准确、清晰和及时，但教程、代码、外部链接、模型版本、依赖版本和平台政策都可能变化。使用前请以原始仓库、官方文档和最新公开资料为准。</p>

    <h2>技术风险</h2>
    <p>运行本站示例命令、训练脚本、推理程序或第三方项目时，可能产生算力费用、数据损失、环境冲突、安全风险或结果偏差。请在理解命令含义、备份重要数据并评估成本后再执行。</p>

    <h2>广告与第三方服务</h2>
    <p>本站计划通过 Google AdSense 或其他合规广告服务展示广告。广告内容由第三方广告系统投放，不代表本站立场、推荐或担保。点击广告或访问第三方链接后产生的交易、服务、隐私和安全问题，请以第三方平台规则为准。</p>

    <h2>版权与引用</h2>
    <p>本站尊重原创与开源协议。如页面内容、图片、代码引用或署名存在问题，请通过 <a href="mailto:${siteOwner.email}">${siteOwner.email}</a> 联系，我们会在核实后及时修正、补充署名或删除相关内容。</p>

    <h2>非专业建议</h2>
    <p>本站内容不构成法律、财务、投资、医疗或其他专业意见。因使用本站内容造成的直接或间接损失，本站作者不承担相应责任，但会积极处理合理反馈并改进内容。</p>
  `;
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
            <a class="site-link" href="/">总教程<span>主线章节 / 示例代码</span></a>
            <a class="site-link" href="/course/">课程站<span>训练流程 / 数据样例</span></a>
            <a class="site-link" href="/minimind-v/">MiniMind-V<span>视觉语言模型专题</span></a>
            <a class="site-link" href="/minimind-o/">MiniMind-O<span>Omni 交互模型专题</span></a>
          </div>
          <div class="nav-block">
            <p class="nav-title">站点信息</p>
            ${legalPages.map((page) => `<a class="site-link" href="/${page.slug}/">${escapeHtml(page.navLabel)}</a>`).join("")}
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
  <title>${escapeHtml(title)} | MiniMind 学习教程</title>
  <link rel="stylesheet" href="/assets/site.css">
</head>
<body>
  <header class="site-header">
    <div class="topbar">
      <a class="brand" href="/">
        <span class="brand-mark">M</span>
        <span>MiniMind 教程<small>系列学习教程</small></span>
      </a>
    </div>
    <div class="primary-nav-row">
      ${renderTopicNav(legalPages.some((page) => page.slug === active) ? "" : "home", active)}
    </div>
    ${["home", "tutorial", "examples"].includes(active) ? `
    <div class="section-nav-row">
      <nav class="section-nav" aria-label="总教程导航">
        <a class="${active === "tutorial" ? "active" : ""}" href="/chapters/01-architecture-overview/">主教程</a>
        <a class="${active === "examples" ? "active" : ""}" href="/examples/02-basic-components-demo/">示例代码</a>
      </nav>
    </div>` : ""}
  </header>
  ${body}
  <footer class="footer">
    <div class="footer-inner">
      <div>
        <strong>MiniMind 系列学习教程</strong>
        <span>由${siteOwner.publicName}整理，覆盖模型结构、训练方法、推理生成和多模态实践；本站不是官方网站。</span>
      </div>
      <nav class="footer-links" aria-label="站点信息">
        ${legalPages.map((page) => `<a href="/${page.slug}/">${escapeHtml(page.navLabel)}</a>`).join("")}
      </nav>
    </div>
  </footer>
</body>
</html>`;
}

function renderTopicLinks(activeId = "") {
  return topicLinks
    .map((link) => `<a class="${link.id === activeId ? "active" : ""}" href="${link.href}">${escapeHtml(link.label)}</a>`)
    .join("");
}

function renderTopicNav(activeId = "", active = "") {
  return `<nav class="topic-nav" data-topic-nav aria-label="一级导航">${renderTopicLinks(activeId)}${renderSiteInfoMenu(active)}</nav>`;
}

function renderSiteInfoMenu(active = "") {
  const isActive = legalPages.some((page) => page.slug === active);
  return `
        <details class="site-info-menu ${isActive ? "active" : ""}">
          <summary>站点信息</summary>
          <div class="site-info-panel">
            ${legalPages.map((page) => `<a class="${page.slug === active ? "active" : ""}" href="/${page.slug}/">${escapeHtml(page.navLabel)}</a>`).join("")}
          </div>
        </details>`;
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
  const addLegalFooter = (html) => {
    if (html.includes("data-site-legal-links")) {
      return html;
    }
    const links = legalPages.map((page) => `<a href="/${page.slug}/">${escapeHtml(page.navLabel)}</a>`).join("");
    const legalBar = `
  <div data-site-legal-links style="border-top:1px solid rgba(120,130,150,.24);background:#fff;padding:14px 16px;text-align:center;font-size:13px;line-height:1.7;color:#667085;">
    <span>本站由${escapeHtml(siteOwner.publicName)}整理，不是官方网站。</span>
    <span style="display:inline-flex;gap:12px;flex-wrap:wrap;justify-content:center;margin-left:12px;">${links}</span>
  </div>`;
    return html.replace("</body>", `${legalBar}\n</body>`);
  };

  await enhanceHtmlDirectory(path.join(distDir, "minimind-o"), (html) => {
    let next = html;
    if (!next.includes("data-topic-nav") && next.includes('<div class="header-actions">')) {
      next = next.replace(
        '<div class="header-actions">',
        `${renderTopicNav("minimind-o")}\n      <div class="header-actions">`,
      );
    }
    return addLegalFooter(next);
  });

  await enhanceHtmlDirectory(path.join(distDir, "minimind-v"), (html) => {
    let next = html;
    const headerEnd = "</nav>\n    </div>\n  </header>";
    if (!next.includes("data-topic-nav") && next.includes(headerEnd)) {
      next = next.replace(
        headerEnd,
        `</nav>\n      ${renderTopicNav("minimind-v")}\n    </div>\n  </header>`,
      );
    }
    return addLegalFooter(next);
  });

  await enhanceHtmlDirectory(path.join(distDir, "course"), addLegalFooter);
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
  const siteUrl = (process.env.SITE_URL || "https://learn-minimind.example.com").replace(/\/$/, "");
  const urls = [
    "/",
    "/course/",
    "/minimind-v/",
    "/minimind-o/",
    ...legalPages.map((page) => `/${page.slug}/`),
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

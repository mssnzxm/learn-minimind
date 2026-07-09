(function () {
  const data = window.MiniMindLearningData;
  const views = [
    { id: "course", label: "课程", body: "按完整训练流程学习每个阶段" },
    { id: "samples", label: "数据样例", body: "整合 train_data_viewer 的数据格式与范例" },
    { id: "concepts", label: "知识点", body: "LLM 概念词典与算法对比" },
    { id: "assets", label: "资源图谱", body: "结构图、数据图和训练曲线" }
  ];

  const state = {
    route: "mini",
    view: "course",
    stageId: data.stages[0].id,
    sampleGroupId: data.sampleGroups[0].id,
    query: "",
    expandedSamples: new Set()
  };

  const els = {
    search: document.getElementById("searchInput"),
    viewTabs: document.getElementById("viewTabs"),
    progressTitle: document.getElementById("progressTitle"),
    progressText: document.getElementById("progressText"),
    progressFill: document.getElementById("progressFill"),
    sidebarLabel: document.getElementById("sidebarLabel"),
    sidebarCount: document.getElementById("sidebarCount"),
    sidebarBody: document.getElementById("sidebarBody"),
    summaryPanel: document.getElementById("summaryPanel"),
    bodyPanel: document.getElementById("bodyPanel"),
    railBody: document.getElementById("railBody"),
    assetModal: document.getElementById("assetModal"),
    assetModalTitle: document.getElementById("assetModalTitle"),
    assetModalBody: document.getElementById("assetModalBody"),
    assetModalImage: document.getElementById("assetModalImage")
  };

  const conceptByTerm = new Map(data.concepts.map((concept) => [concept.term, concept]));
  let scrollTicking = false;

  function normalize(value) {
    return String(value || "").toLowerCase();
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function escapeAttribute(value) {
    return escapeHtml(value).replaceAll("\n", " ");
  }

  function cssString(value) {
    return String(value ?? "").replaceAll("\\", "\\\\").replaceAll('"', '\\"');
  }

  function currentStage() {
    return data.stages.find((stage) => stage.id === state.stageId) || data.stages[0];
  }

  function currentSampleGroup() {
    return data.sampleGroups.find((group) => group.id === state.sampleGroupId) || data.sampleGroups[0];
  }

  function queryMatch(parts) {
    const query = normalize(state.query);
    if (!query) return true;
    return normalize(parts.filter(Boolean).join(" ")).includes(query);
  }

  function filteredStages() {
    return data.stages.filter((stage) => queryMatch([
      stage.title,
      stage.shortTitle,
      stage.tag,
      stage.summary,
      stage.problem,
      stage.scripts.join(" "),
      stage.dataFiles.join(" "),
      stage.concepts.join(" "),
      stage.sample
    ]));
  }

  function filteredSampleGroups() {
    return data.sampleGroups
      .map((group) => {
        const examples = group.examples.filter((example) => queryMatch([
          group.title,
          group.intro,
          group.files.join(" "),
          group.format,
          example.title,
          example.summary,
          example.raw
        ]));
        const groupMatches = queryMatch([group.title, group.intro, group.files.join(" "), group.format]);
        return {
          ...group,
          examples: groupMatches ? group.examples : examples
        };
      })
      .filter((group) => group.examples.length > 0 || queryMatch([group.title, group.intro, group.files.join(" ")]));
  }

  function filteredConcepts() {
    return data.concepts.filter((concept) => queryMatch([concept.term, concept.body]));
  }

  function formatCode(value, language) {
    const raw = String(value ?? "");
    const escaped = escapeHtml(raw);
    if (language === "shell") {
      return escaped
        .replace(/^([a-zA-Z0-9_./-]+)(?=\s|$)/gm, '<span class="tok-command">$1</span>')
        .replace(/(--[a-zA-Z0-9_-]+)/g, '<span class="tok-key">$1</span>');
    }

    return escaped
      .replace(/(&quot;[^&]*?&quot;)(\s*:)/g, '<span class="tok-key">$1</span>$2')
      .replace(/:\s*(&quot;.*?&quot;)/g, ': <span class="tok-string">$1</span>')
      .replace(/\b(true|false|null)\b/g, '<span class="tok-bool">$1</span>')
      .replace(/\b(-?\d+(?:\.\d+)?)\b/g, '<span class="tok-number">$1</span>');
  }

  function codeBlock(value, language = "json", id = "") {
    const idAttr = id ? ` id="${escapeAttribute(id)}"` : "";
    return `<pre class="code-block"><code${idAttr} data-language="${escapeAttribute(language)}">${formatCode(value, language)}</code></pre>`;
  }

  function copyText(value, button) {
    const finish = () => {
      const oldText = button.textContent;
      button.textContent = "已复制";
      window.setTimeout(() => {
        button.textContent = oldText;
      }, 1100);
    };

    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(value).then(finish).catch(() => fallbackCopy(value, finish));
    } else {
      fallbackCopy(value, finish);
    }
  }

  function fallbackCopy(value, finish) {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
    finish();
  }

  function renderViewTabs() {
    els.viewTabs.innerHTML = views
      .map((view) => `
        <button class="view-tab ${view.id === state.view ? "is-active" : ""}" type="button" data-view="${escapeAttribute(view.id)}">
          <strong>${escapeHtml(view.label)}</strong>
          <span>${escapeHtml(view.body)}</span>
        </button>
      `)
      .join("");
  }

  function setProgress(index, total, label) {
    const safeTotal = Math.max(total, 1);
    const percent = Math.round(((index + 1) / safeTotal) * 100);
    els.progressTitle.textContent = label;
    els.progressText.textContent = `${index + 1}/${safeTotal} · ${percent}%`;
    els.progressFill.style.width = `${percent}%`;
  }

  function renderCourseSummary() {
    const route = data.routes[state.route];
    els.summaryPanel.innerHTML = `
      <section id="overview" class="hero-band content-section" data-stage-anchor="${escapeAttribute(data.stages[0].id)}">
        <div class="hero-copy">
          <p class="eyebrow">MiniMind Course Product</p>
          <h1>把大模型训练拆成可以读懂、可以运行、可以修改的课程</h1>
          <p>
            课程按仓库真实脚本组织，从 LLM 基础、Pretrain、SFT、LoRA、DPO、PPO/GRPO/CISPO，
            一路走到 Tool Use、Agentic RL、自适应思考、模型蒸馏和部署。
          </p>
          <div class="route-summary">
            <strong>${escapeHtml(route.label)}</strong>
            <span>${escapeHtml(route.summary)}</span>
            <small>${escapeHtml(route.commandNote)}</small>
          </div>
        </div>
        <button class="hero-visual asset-trigger" type="button" data-asset-path="images/LLM-structure.jpg" data-asset-title="Dense 结构" data-asset-body="MiniMind-3 Dense Decoder-only 结构。">
          <img src="images/LLM-structure.jpg" alt="MiniMind dense model structure" />
        </button>
      </section>

      <section class="flow-panel">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Training Pipeline</p>
            <h2>阶段路线图</h2>
          </div>
          <span class="heading-note">点击任一阶段跳转课程章节</span>
        </div>
        <div class="pipeline">
          ${data.stages.map((stage) => `
            <button class="pipeline-step ${stage.id === state.stageId ? "is-active" : ""}" type="button" data-stage="${escapeAttribute(stage.id)}">
              <span>${escapeHtml(stage.order)}</span>
              <strong>${escapeHtml(stage.shortTitle)}</strong>
            </button>
          `).join("")}
        </div>
      </section>
    `;
  }

  function renderStageCard(stage) {
    const command = stage.commands[state.route] || stage.commands.mini;
    const scripts = stage.scripts.map((item) => `<code>${escapeHtml(item)}</code>`).join("");
    const files = stage.dataFiles.map((item) => `<code>${escapeHtml(item)}</code>`).join("");
    const concepts = stage.concepts.map((term) => `<span>${escapeHtml(term)}</span>`).join("");
    const checks = stage.nextCheck.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
    const notes = stage.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("");
    const commandId = `command-${stage.id}`;
    const sampleId = `sample-${stage.id}`;

    return `
      <article id="${escapeAttribute(stage.id)}" class="stage-card content-section" data-stage-anchor="${escapeAttribute(stage.id)}">
        <div class="stage-title-row">
          <div>
            <p class="eyebrow">${escapeHtml(stage.tag)}</p>
            <h2>${escapeHtml(stage.order)} ${escapeHtml(stage.title)}</h2>
          </div>
          <span class="stage-badge">${escapeHtml(stage.shortTitle)}</span>
        </div>

        <p class="stage-summary">${escapeHtml(stage.summary)}</p>

        <div class="stage-grid">
          <section>
            <h3>它解决什么问题</h3>
            <p>${escapeHtml(stage.problem)}</p>
          </section>
          <section>
            <h3>输入数据长什么样</h3>
            <p>${escapeHtml(stage.dataShape)}</p>
          </section>
          <section>
            <h3>核心损失 / 奖励信号</h3>
            <p>${escapeHtml(stage.coreSignal)}</p>
          </section>
          <section>
            <h3>训练后产物</h3>
            <p>${escapeHtml(stage.output)}</p>
          </section>
        </div>

        <div class="path-row">
          <div>
            <h3>代码入口</h3>
            <div class="code-list">${scripts}</div>
          </div>
          <div>
            <h3>数据文件</h3>
            <div class="code-list">${files}</div>
          </div>
        </div>

        <section class="command-panel">
          <div class="panel-heading">
            <h3>默认命令</h3>
            <button class="copy-button" type="button" data-copy-from="${escapeAttribute(commandId)}">复制</button>
          </div>
          ${codeBlock(command, "shell", commandId)}
        </section>

        <details class="sample-panel sample-details">
          <summary>
            <span>${escapeHtml(stage.sampleLabel)}</span>
            <button class="copy-button" type="button" data-copy-from="${escapeAttribute(sampleId)}">复制</button>
          </summary>
          ${codeBlock(stage.sample, "json", sampleId)}
        </details>

        <div class="stage-footer-grid">
          <section class="notes-panel">
            <h3>学习提示</h3>
            <ul>${notes}</ul>
          </section>
          <section class="notes-panel">
            <h3>继续下一阶段前确认</h3>
            <ul>${checks}</ul>
          </section>
        </div>

        <div class="concept-chips">${concepts}</div>

        ${stage.image ? `
          <button class="stage-image-button asset-trigger" type="button" data-asset-path="${escapeAttribute(stage.image)}" data-asset-title="${escapeAttribute(stage.title)}" data-asset-body="${escapeAttribute(stage.summary)}">
            <img class="stage-image" src="${escapeAttribute(stage.image)}" alt="${escapeAttribute(stage.title)} reference" />
          </button>
        ` : ""}
      </article>
    `;
  }

  function renderCourseBody() {
    const stages = filteredStages();
    if (!stages.some((stage) => stage.id === state.stageId) && stages.length > 0) {
      state.stageId = stages[0].id;
    }

    els.bodyPanel.innerHTML = stages.length
      ? stages.map(renderStageCard).join("")
      : `<section class="empty-state"><h2>没有匹配的课程章节</h2><p>换一个关键词试试。</p></section>`;
  }

  function renderCourseSidebar() {
    const stages = filteredStages();
    els.sidebarLabel.textContent = "课程目录";
    els.sidebarCount.textContent = String(stages.length);
    els.sidebarBody.innerHTML = `
      <nav class="stage-nav">
        ${stages.map((stage) => `
          <button class="stage-link ${stage.id === state.stageId ? "is-active" : ""}" type="button" data-stage="${escapeAttribute(stage.id)}">
            <span>${escapeHtml(stage.order)}</span>
            <strong>${escapeHtml(stage.shortTitle)}</strong>
            <small>${escapeHtml(stage.tag)}</small>
          </button>
        `).join("")}
      </nav>
    `;
  }

  function renderCourseRail() {
    const stage = currentStage();
    const concepts = stage.concepts.map((term) => {
      const concept = conceptByTerm.get(term);
      return `
        <article class="concept-card">
          <strong>${escapeHtml(term)}</strong>
          <p>${escapeHtml(concept ? concept.body : "当前阶段的关键概念。")}</p>
        </article>
      `;
    }).join("");

    els.railBody.innerHTML = `
      <section class="rail-panel">
        <div class="rail-heading"><span>当前章节</span></div>
        <div class="chapter-card">
          <strong>${escapeHtml(stage.order)} ${escapeHtml(stage.title)}</strong>
          <p>${escapeHtml(stage.summary)}</p>
        </div>
      </section>
      <section class="rail-panel">
        <div class="rail-heading"><span>本章知识点</span></div>
        <div class="concept-list">${concepts}</div>
      </section>
      <section class="rail-panel">
        <div class="rail-heading"><span>继续前检查</span></div>
        <ul class="check-list">${stage.nextCheck.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      </section>
    `;
  }

  function renderSamplesSummary() {
    els.summaryPanel.innerHTML = `
      <section class="sample-hero">
        <div>
          <p class="eyebrow">Dataset Viewer Integrated</p>
          <h1>训练数据范例中心</h1>
          <p>
            这里把 dataset/train_data_viewer.html 的内容整合到课程站里，覆盖 Pretrain、SFT、DPO、
            RLAIF、Agent RL 和 LoRA 的数据格式、样例消息、工具定义与 gt。
          </p>
        </div>
        <div class="sample-actions">
          <button class="action-button" type="button" data-expand-all="true">展开全部样例</button>
          <button class="ghost-button" type="button" data-collapse-all="true">收起全部样例</button>
        </div>
      </section>
    `;
  }

  function renderMessage(message) {
    const role = message.role || "text";
    const blocks = [];
    if (message.tools) {
      blocks.push(`<div class="tools-def-block"><strong>TOOLS 定义</strong>${codeBlock(message.tools, "json")}</div>`);
    }
    if (message.reasoning) {
      blocks.push(`<div class="reasoning-block"><strong>REASONING</strong><p>${escapeHtml(message.reasoning)}</p></div>`);
    }
    if (message.toolCall) {
      blocks.push(`<div class="tool-call-block"><strong>TOOL CALL</strong><p>${escapeHtml(message.toolCall)}</p></div>`);
    }
    if (message.pending) {
      blocks.push(`<div class="msg-content pending-gen">${escapeHtml(message.pending)}</div>`);
    } else if (message.content) {
      blocks.push(`<div class="msg-content">${escapeHtml(message.content).replaceAll("\n", "<br>")}</div>`);
    }
    return `
      <div class="msg ${escapeAttribute(role)}">
        <div class="msg-role ${escapeAttribute(role)}">${escapeHtml(role)}</div>
        ${blocks.join("")}
      </div>
    `;
  }

  function renderExample(example, group) {
    const isExpanded = state.expandedSamples.has(example.id);
    const rawId = `raw-${example.id}`;
    const visual = example.compare
      ? `<div class="dpo-compare">
          ${example.compare.map((side) => `
            <div class="dpo-side ${escapeAttribute(side.tone)}">
              <h4>${escapeHtml(side.label)}</h4>
              ${side.messages.map(renderMessage).join("")}
            </div>
          `).join("")}
        </div>`
      : `${(example.messages || []).map(renderMessage).join("")}${example.gt ? `<div class="gt-block"><strong>GROUND TRUTH</strong><p>${escapeHtml(example.gt)}</p></div>` : ""}`;

    return `
      <article class="example-card ${isExpanded ? "is-open" : ""}" data-sample-id="${escapeAttribute(example.id)}">
        <button class="example-card-header" type="button" data-toggle-sample="${escapeAttribute(example.id)}">
          <span>
            <small>${escapeHtml(group.order)} · ${escapeHtml(group.title)}</small>
            <strong>${escapeHtml(example.title)}</strong>
          </span>
          <em>${isExpanded ? "收起" : "展开"}</em>
        </button>
        <div class="example-card-body" ${isExpanded ? "" : "hidden"}>
          <p>${escapeHtml(example.summary)}</p>
          <div class="message-stack">${visual}</div>
          <details class="raw-json" open>
            <summary>
              <span>原始 JSONL</span>
              <button class="copy-button" type="button" data-copy-from="${escapeAttribute(rawId)}">复制</button>
            </summary>
            ${codeBlock(example.raw, "json", rawId)}
          </details>
        </div>
      </article>
    `;
  }

  function renderSampleGroup(group) {
    const formatId = `format-${group.id}`;
    return `
      <section id="sample-${escapeAttribute(group.id)}" class="sample-group content-section" data-sample-anchor="${escapeAttribute(group.id)}">
        <div class="sample-group-header">
          <div>
            <p class="eyebrow">${escapeHtml(group.order)} Dataset</p>
            <h2>${escapeHtml(group.title)}</h2>
            <p>${escapeHtml(group.intro)}</p>
          </div>
          <button class="stage-jump" type="button" data-view-stage="${escapeAttribute(group.stageId)}">看课程章节</button>
        </div>
        <div class="meta-row">
          ${group.files.map((file) => `<span class="meta-tag">${escapeHtml(file)}</span>`).join("")}
        </div>
        <section class="format-box">
          <div class="panel-heading">
            <h3>数据格式</h3>
            <button class="copy-button" type="button" data-copy-from="${escapeAttribute(formatId)}">复制</button>
          </div>
          ${codeBlock(group.format, "json", formatId)}
        </section>
        <div class="example-list">
          ${group.examples.map((example) => renderExample(example, group)).join("")}
        </div>
      </section>
    `;
  }

  function renderSamplesBody() {
    const groups = filteredSampleGroups();
    if (!groups.some((group) => group.id === state.sampleGroupId) && groups.length > 0) {
      state.sampleGroupId = groups[0].id;
    }
    els.bodyPanel.innerHTML = groups.length
      ? groups.map(renderSampleGroup).join("")
      : `<section class="empty-state"><h2>没有匹配的数据样例</h2><p>换一个关键词试试。</p></section>`;
  }

  function renderSamplesSidebar() {
    const groups = filteredSampleGroups();
    els.sidebarLabel.textContent = "数据目录";
    els.sidebarCount.textContent = String(groups.length);
    els.sidebarBody.innerHTML = `
      <nav class="stage-nav">
        ${groups.map((group) => `
          <button class="stage-link ${group.id === state.sampleGroupId ? "is-active" : ""}" type="button" data-sample-group="${escapeAttribute(group.id)}">
            <span>${escapeHtml(group.order)}</span>
            <strong>${escapeHtml(group.id.toUpperCase())}</strong>
            <small>${escapeHtml(group.examples.length)} 样例</small>
          </button>
        `).join("")}
      </nav>
    `;
  }

  function renderSamplesRail() {
    const group = currentSampleGroup();
    els.railBody.innerHTML = `
      <section class="rail-panel">
        <div class="rail-heading"><span>当前数据集</span></div>
        <div class="chapter-card">
          <strong>${escapeHtml(group.title)}</strong>
          <p>${escapeHtml(group.intro)}</p>
        </div>
      </section>
      <section class="rail-panel">
        <div class="rail-heading"><span>字段速记</span></div>
        <ul class="check-list">
          <li><code>conversations</code> 是多轮消息数组。</li>
          <li><code>tools</code> 是字符串化的工具 schema。</li>
          <li><code>tool_calls</code> 是 assistant 生成的结构化调用。</li>
          <li><code>gt</code> 是 Agent RL 的奖励校验目标。</li>
        </ul>
      </section>
    `;
  }

  function renderConceptsView() {
    const concepts = filteredConcepts();
    els.summaryPanel.innerHTML = `
      <section class="sample-hero">
        <div>
          <p class="eyebrow">LLM Concepts</p>
          <h1>知识点词典与算法对比</h1>
          <p>把 MiniMind 训练脚本里频繁出现的概念拆成短词条，并把 DPO / PPO / GRPO / CISPO 放在同一张表里比较。</p>
        </div>
      </section>
    `;
    els.bodyPanel.innerHTML = `
      <section class="concept-board">
        ${concepts.map((concept) => `
          <article class="concept-tile content-section" data-concept-anchor="${escapeAttribute(concept.term)}">
            <strong>${escapeHtml(concept.term)}</strong>
            <p>${escapeHtml(concept.body)}</p>
          </article>
        `).join("")}
      </section>
      <section class="stage-card algorithm-panel">
        <div class="section-heading">
          <div>
            <p class="eyebrow">RL Algorithms</p>
            <h2>DPO / PPO / GRPO / CISPO 对比</h2>
          </div>
        </div>
        ${algorithmTable()}
      </section>
    `;
    els.sidebarLabel.textContent = "知识目录";
    els.sidebarCount.textContent = String(concepts.length);
    els.sidebarBody.innerHTML = `
      <nav class="concept-nav">
        ${concepts.map((concept) => `<button type="button" data-concept="${escapeAttribute(concept.term)}">${escapeHtml(concept.term)}</button>`).join("")}
      </nav>
    `;
    els.railBody.innerHTML = `
      <section class="rail-panel">
        <div class="rail-heading"><span>推荐阅读顺序</span></div>
        <ol class="ordered-list">
          <li>Tokenizer / Causal LM / Decoder-only</li>
          <li>loss mask / chat template / SFT</li>
          <li>DPO / PPO / GRPO / CISPO</li>
          <li>Tool Calling / Agentic RL / rollout</li>
        </ol>
      </section>
    `;
    setProgress(0, 1, "知识点");
  }

  function algorithmTable() {
    return `
      <div class="algorithm-table">
        <table>
          <thead>
            <tr>
              <th>算法</th>
              <th>范式</th>
              <th>数据</th>
              <th>优势信号</th>
              <th>特点</th>
            </tr>
          </thead>
          <tbody>
            ${data.algorithms.map((item) => `
              <tr>
                <td>${escapeHtml(item.name)}</td>
                <td>${escapeHtml(item.mode)}</td>
                <td>${escapeHtml(item.data)}</td>
                <td>${escapeHtml(item.advantage)}</td>
                <td>${escapeHtml(item.extra)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function renderAssetsView() {
    els.summaryPanel.innerHTML = `
      <section class="sample-hero">
        <div>
          <p class="eyebrow">Repository Assets</p>
          <h1>项目图谱与训练曲线</h1>
          <p>所有图片都来自当前仓库 images/，点击卡片可以放大查看结构细节和 loss 曲线。</p>
        </div>
      </section>
    `;
    els.bodyPanel.innerHTML = `
      <section class="assets-band">
        <div class="asset-grid">
          ${data.assets.map((asset) => `
            <article class="asset-card asset-trigger content-section" data-asset-path="${escapeAttribute(asset.path)}" data-asset-title="${escapeAttribute(asset.title)}" data-asset-body="${escapeAttribute(asset.body)}" role="button" tabindex="0">
              <img src="${escapeAttribute(asset.path)}" alt="${escapeAttribute(asset.title)}" />
              <div>
                <strong>${escapeHtml(asset.title)}</strong>
                <p>${escapeHtml(asset.body)}</p>
              </div>
            </article>
          `).join("")}
        </div>
      </section>
    `;
    els.sidebarLabel.textContent = "资源目录";
    els.sidebarCount.textContent = String(data.assets.length);
    els.sidebarBody.innerHTML = `
      <nav class="concept-nav">
        ${data.assets.map((asset, index) => `<button type="button" data-asset-index="${index}">${escapeHtml(asset.title)}</button>`).join("")}
      </nav>
    `;
    els.railBody.innerHTML = `
      <section class="rail-panel">
        <div class="rail-heading"><span>查看提示</span></div>
        <ul class="check-list">
          <li>点击任意图片卡片可打开大图。</li>
          <li>按 Esc 或点击遮罩关闭预览。</li>
          <li>结构图适合配合课程章节阅读。</li>
        </ul>
      </section>
    `;
    setProgress(0, 1, "资源图谱");
  }

  function renderCourseView() {
    renderCourseSummary();
    renderCourseBody();
    renderCourseSidebar();
    renderCourseRail();
    updateCourseProgress();
  }

  function renderSamplesView() {
    renderSamplesSummary();
    renderSamplesBody();
    renderSamplesSidebar();
    renderSamplesRail();
    updateSampleProgress();
  }

  function renderRouteButtons() {
    document.querySelectorAll(".route-button").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.route === state.route);
    });
  }

  function render() {
    renderViewTabs();
    renderRouteButtons();
    if (state.view === "course") {
      renderCourseView();
    } else if (state.view === "samples") {
      renderSamplesView();
    } else if (state.view === "concepts") {
      renderConceptsView();
    } else {
      renderAssetsView();
    }
  }

  function updateCourseProgress() {
    const stages = filteredStages();
    const index = Math.max(0, stages.findIndex((stage) => stage.id === state.stageId));
    setProgress(index, stages.length || 1, "课程进度");
    document.querySelectorAll("[data-stage]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.stage === state.stageId);
    });
  }

  function updateSampleProgress() {
    const groups = filteredSampleGroups();
    const index = Math.max(0, groups.findIndex((group) => group.id === state.sampleGroupId));
    setProgress(index, groups.length || 1, "数据样例");
    document.querySelectorAll("[data-sample-group]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.sampleGroup === state.sampleGroupId);
    });
  }

  function setActiveFromScroll() {
    if (state.view !== "course" && state.view !== "samples") return;
    const selector = state.view === "course" ? "[data-stage-anchor]" : "[data-sample-anchor]";
    const attr = state.view === "course" ? "stageAnchor" : "sampleAnchor";
    const sections = Array.from(document.querySelectorAll(selector));
    if (!sections.length) return;

    const offset = 150;
    let active = sections[0];
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= offset) active = section;
    }

    if (state.view === "course") {
      const id = active.dataset[attr];
      if (id && state.stageId !== id) {
        state.stageId = id;
        renderCourseSidebar();
        renderCourseRail();
        updateCourseProgress();
      }
    } else {
      const id = active.dataset[attr];
      if (id && state.sampleGroupId !== id) {
        state.sampleGroupId = id;
        renderSamplesSidebar();
        renderSamplesRail();
        updateSampleProgress();
      }
    }
  }

  function scheduleScrollUpdate() {
    if (scrollTicking) return;
    scrollTicking = true;
    window.requestAnimationFrame(() => {
      setActiveFromScroll();
      scrollTicking = false;
    });
  }

  function scrollToId(id) {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function openAssetModal(asset) {
    if (!asset) return;
    els.assetModalTitle.textContent = asset.title;
    els.assetModalBody.textContent = asset.body;
    els.assetModalImage.src = asset.path;
    els.assetModalImage.alt = asset.title;
    els.assetModal.classList.add("is-open");
    els.assetModal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    els.assetModal.querySelector(".asset-modal-close")?.focus();
  }

  function closeAssetModal() {
    els.assetModal.classList.remove("is-open");
    els.assetModal.setAttribute("aria-hidden", "true");
    els.assetModalImage.removeAttribute("src");
    els.assetModalImage.alt = "";
    document.body.style.overflow = "";
  }

  function handleSampleToggle(id) {
    if (state.expandedSamples.has(id)) {
      state.expandedSamples.delete(id);
    } else {
      state.expandedSamples.add(id);
    }
    renderSamplesView();
    window.setTimeout(() => {
      document.querySelector(`[data-sample-id="${cssString(id)}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 0);
  }

  function expandAllSamples(expand) {
    state.expandedSamples.clear();
    if (expand) {
      data.sampleGroups.forEach((group) => {
        group.examples.forEach((example) => state.expandedSamples.add(example.id));
      });
    }
    renderSamplesView();
  }

  function bindEvents() {
    document.addEventListener("click", (event) => {
      const copyButton = event.target.closest("[data-copy-from]");
      if (copyButton) {
        event.preventDefault();
        event.stopPropagation();
        const target = document.getElementById(copyButton.dataset.copyFrom);
        if (target) copyText(target.textContent, copyButton);
        return;
      }

      const viewButton = event.target.closest("[data-view]");
      if (viewButton) {
        state.view = viewButton.dataset.view;
        render();
        window.scrollTo({ top: 0, behavior: "smooth" });
        return;
      }

      const routeButton = event.target.closest("[data-route]");
      if (routeButton) {
        state.route = routeButton.dataset.route;
        render();
        return;
      }

      const stageButton = event.target.closest("[data-stage]");
      if (stageButton) {
        state.view = "course";
        state.stageId = stageButton.dataset.stage;
        render();
        window.setTimeout(() => scrollToId(state.stageId), 0);
        return;
      }

      const viewStageButton = event.target.closest("[data-view-stage]");
      if (viewStageButton) {
        state.view = "course";
        state.stageId = viewStageButton.dataset.viewStage;
        render();
        window.setTimeout(() => scrollToId(state.stageId), 0);
        return;
      }

      const sampleGroupButton = event.target.closest("[data-sample-group]");
      if (sampleGroupButton) {
        state.sampleGroupId = sampleGroupButton.dataset.sampleGroup;
        scrollToId(`sample-${state.sampleGroupId}`);
        updateSampleProgress();
        renderSamplesRail();
        return;
      }

      const toggleSampleButton = event.target.closest("[data-toggle-sample]");
      if (toggleSampleButton) {
        handleSampleToggle(toggleSampleButton.dataset.toggleSample);
        return;
      }

      if (event.target.closest("[data-expand-all]")) {
        expandAllSamples(true);
        return;
      }

      if (event.target.closest("[data-collapse-all]")) {
        expandAllSamples(false);
        return;
      }

      const conceptButton = event.target.closest("[data-concept]");
      if (conceptButton) {
        const target = Array.from(document.querySelectorAll("[data-concept-anchor]"))
          .find((item) => item.dataset.conceptAnchor === conceptButton.dataset.concept);
        target?.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }

      const assetIndexButton = event.target.closest("[data-asset-index]");
      if (assetIndexButton) {
        const asset = data.assets[Number(assetIndexButton.dataset.assetIndex)];
        openAssetModal(asset);
        return;
      }

      const assetTrigger = event.target.closest(".asset-trigger");
      if (assetTrigger) {
        openAssetModal({
          path: assetTrigger.dataset.assetPath,
          title: assetTrigger.dataset.assetTitle,
          body: assetTrigger.dataset.assetBody
        });
      }
    });

    document.addEventListener("keydown", (event) => {
      if ((event.key === "Enter" || event.key === " ") && event.target.matches(".asset-card")) {
        event.preventDefault();
        event.target.click();
      }

      if (event.key === "Escape" && els.assetModal.classList.contains("is-open")) {
        closeAssetModal();
      }
    });

    els.assetModal.addEventListener("click", (event) => {
      if (event.target.closest("[data-close-modal]")) {
        closeAssetModal();
      }
    });

    els.search.addEventListener("input", (event) => {
      state.query = event.target.value;
      render();
    });

    window.addEventListener("scroll", scheduleScrollUpdate, { passive: true });
  }

  function initializeFromHash() {
    const id = location.hash.replace("#", "");
    if (data.stages.some((stage) => stage.id === id)) {
      state.view = "course";
      state.stageId = id;
    }
    const sampleId = id.replace("sample-", "");
    if (data.sampleGroups.some((group) => group.id === sampleId)) {
      state.view = "samples";
      state.sampleGroupId = sampleId;
    }
  }

  initializeFromHash();
  bindEvents();
  render();
})();

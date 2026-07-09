# learn-minimind

MiniMind 系列学习教程的纯静态版本。这个仓库不包含原教程站里的微信公众号同步、在线编辑器、API 路由或 React 客户端状态；所有页面都在构建时生成到 `dist/`，可以直接部署到 Cloudflare。

## 内容

- `content/chapters/`：MiniMind 主教程 Markdown。
- `content/examples/`：配套 Python 示例代码。
- `public/course/`：MiniMind 学习课程静态站。
- `public/minimind-v/`：MiniMind-V 静态专题教程。
- `public/minimind-o/`：MiniMind-O 静态专题教程。
- `scripts/build.mjs`：把 Markdown 和静态资源构建为 `dist/`。

## 本地构建

```bash
npm install
npm run build
```

构建结果在 `dist/`：

- `/`：系列教程首页。
- `/chapters/01-architecture-overview/`：主教程章节。
- `/examples/02-basic-components-demo/`：示例代码页。
- `/course/`：课程静态站。
- `/minimind-v/`：MiniMind-V 教程。
- `/minimind-o/`：MiniMind-O 教程。

## Cloudflare Workers Static Assets

仓库已经提供 `wrangler.jsonc`，指向 `dist/`：

```bash
npm install
npm run build
npx wrangler deploy
```

也可以直接使用脚本：

```bash
npm run deploy
```

首次部署时需要按 Wrangler 提示登录 Cloudflare，并确认创建名为 `learn-minimind` 的 Worker。

## Cloudflare Pages

如果使用 Cloudflare Pages 连接 GitHub 仓库，推荐配置：

```text
Build command: npm ci && npm run build
Build output directory: dist
Root directory: /
```

也可以手动部署：

```bash
npm run pages:deploy
```

## 更新内容

修改 `content/chapters/*.md` 或 `content/examples/*.py` 后重新运行：

```bash
npm run build
```

如需调整站点样式，编辑 `src/site.css`；如需调整生成规则，编辑 `scripts/build.mjs`。

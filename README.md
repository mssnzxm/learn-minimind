# MiniMind 系列学习教程

MiniMind 系列学习教程，内容覆盖主线模型结构、训练方法、推理生成、课程导览、MiniMind-V 和 MiniMind-O 专题。

## 内容

- `content/chapters/`：MiniMind 主教程 Markdown。
- `content/examples/`：配套 Python 示例代码。
- `public/course/`：MiniMind 学习课程静态站。
- `public/minimind-v/`：MiniMind-V 静态专题教程。
- `public/minimind-o/`：MiniMind-O 静态专题教程。
- `scripts/build.mjs`：生成可发布的网站文件。

## 本地使用

```bash
npm install
npm run build
```

生成结果包含以下入口：

- `/`：系列教程首页。
- `/chapters/01-architecture-overview/`：主教程章节。
- `/examples/02-basic-components-demo/`：示例代码页。
- `/course/`：课程导览。
- `/minimind-v/`：MiniMind-V 教程。
- `/minimind-o/`：MiniMind-O 教程。

## 发布配置

```text
Build command: npm ci && npm run build
Build output directory: dist
Root directory: /
```

## 更新内容

修改 `content/chapters/*.md` 或 `content/examples/*.py` 后重新运行：

```bash
npm run build
```

如需调整站点样式，编辑 `src/site.css`；如需调整生成规则，编辑 `scripts/build.mjs`。

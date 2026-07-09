# Learn MiniMind-O

MiniMind-O 学习教程静态网站。内容面向想从源码、训练数据、推理脚本和 WebUI 入手学习 MiniMind-O 的读者。

## 本地预览

```bash
python3 serve.py --port 8765
```

然后访问：

```text
http://127.0.0.1:8765/
```

也可以直接用任意静态服务器托管当前目录。

## Cloudflare Pages 部署

本项目通过 Cloudflare Pages 的 Git 集成部署（连接 GitHub 仓库后，推送即自动部署），无需 wrangler CLI，也无需构建步骤。

仓库内所有页面和静态资源都已经准备好，目录结构如下：

```text
.
├── index.html
├── foundations.html
├── setup.html
├── architecture.html
├── data-training.html
├── inference-webui.html
├── experiments.html
├── references.html
├── serve.py
├── _headers
├── .nojekyll
├── .gitattributes
└── assets/
    ├── app.js
    ├── styles.css
    └── images/
```

在 Cloudflare Pages 控制台连接本仓库后，按以下设置即可：

- Framework preset: `None`
- Build command: 留空
- Build output directory: `/`
- Root directory: 仓库根目录

推送 `main` 分支后 Cloudflare 会自动拉取并部署。

## 内容来源

教程内容基于本地 MiniMind-O 代码阅读和公开资料整理，主要参考：

- https://github.com/jingyaogong/minimind-o
- http://arxiv.org/abs/2605.03937
- https://huggingface.co/collections/jingyaogong/minimind-o
- https://modelscope.cn/studios/gongjy/MiniMind-O

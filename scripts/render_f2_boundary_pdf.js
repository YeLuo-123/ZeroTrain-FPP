#!/usr/bin/env node
"use strict";

// Pre-render the report's 27 display formulas with KaTeX.  WeasyPrint does
// not lay out MathML fractions consistently in this environment, while KaTeX
// HTML is deterministic and retains vector text in the generated PDF.
const fs = require("fs");
const katex = require("/usr/share/code/resources/app/node_modules/katex");

const [sourcePath, outputPath] = process.argv.slice(2);
if (!sourcePath || !outputPath) {
  throw new Error("usage: render_f2_boundary_pdf.js SOURCE_HTML OUTPUT_HTML");
}

const formulas = [
  String.raw`I=A+B\cos\Phi+B_5\cos(5\Phi)`,
  String.raw`\delta_m=\frac{2\pi m}{12},\qquad m=0,\ldots,11`,
  String.raw`S_m^{\mathrm{ideal}}=c(1+\cos\delta_m)`,
  String.raw`J_k=\sum_{m=0}^{11}S_m\!\left[A+B\cos(\Phi+\alpha_k+\delta_m)+B_5\cos\!\left(5(\Phi+\alpha_k+\delta_m)\right)\right]`,
  String.raw`u_p(x,y)\in[0,1)`,
  String.raw`\begin{aligned}\Phi_2&=2\pi\cdot2u_p=4\pi u_p,\\ \Phi_h&=2\pi f_hu_p\end{aligned}`,
  String.raw`\Phi_h=\frac{48}{2}\Phi_2=24\Phi_2`,
  String.raw`\phi_2=\operatorname{mod}(\Phi_2,2\pi)`,
  String.raw`K_2=\left\lfloor\frac{\Phi_2}{2\pi}\right\rfloor\in\{0,1\}`,
  String.raw`\Phi_2=\phi_2+2\pi K_2`,
  String.raw`\Gamma=\{(x,y)\mid\Phi_2(x,y)=2\pi\}`,
  String.raw`\begin{aligned}C&=\sum_{k=0}^{2}I_k\cos\frac{2\pi k}{3},\\ S&=\sum_{k=0}^{2}I_k\sin\frac{2\pi k}{3}\end{aligned}`,
  String.raw`\phi=\operatorname{mod}\!\left[-\operatorname{atan2}(S,C),\,2\pi\right]`,
  String.raw`\begin{aligned}D&=c_2C+c_3S,\\ N&=-c_3C+c_2S\end{aligned}`,
  String.raw`\phi=\operatorname{mod}\!\left[-\operatorname{atan2}(N,D),\,2\pi\right]`,
  String.raw`\begin{aligned}A&=\frac{I_0+I_1+I_2}{3},\\ B&=\frac{2}{3}\sqrt{C^2+S^2},\\ Q&=\frac{B}{|A|+\varepsilon}\end{aligned}`,
  String.raw`d(x,y)=\phi_2(x+1,y)-\phi_2(x,y)`,
  String.raw`d(x,y)<-\pi\qquad\text{且}\qquad\min(Q_x,Q_{x+1})>T_Q`,
  String.raw`s(x,y)=-d(x,y)\min(Q_x,Q_{x+1})`,
  String.raw`\begin{aligned}m&=\operatorname{median}(x_{\mathrm{obs}}),\\ \operatorname{MAD}&=\operatorname{median}(|x_{\mathrm{obs}}-m|)\end{aligned}`,
  String.raw`\hat{x}_{\Gamma}(y)=\operatorname{PCHIP}\{(y_i,x_i)\}`,
  String.raw`\hat K_2(x,y)=\begin{cases}0,&x<\hat{x}_{\Gamma}(y),\\1,&x\geq\hat{x}_{\Gamma}(y),\end{cases}`,
  String.raw`\hat{\Phi}_2=\phi_2+2\pi\hat K_2`,
  String.raw`\hat K_h=\operatorname{round}\!\left[\frac{(f_h/2)\hat{\Phi}_2-\phi_h}{2\pi}\right]`,
  String.raw`\hat{\Phi}_h=\phi_h+2\pi\hat K_h`,
  String.raw`E=E_{\mathrm{continuous}}+E_{\mathrm{topology}}`,
  String.raw`\{P_\Gamma,P(K_2=0),P(K_2=1),C\}=F_\theta(I_{2,0:2},\phi_2,Q,M_{\mathrm{seed}})`,
];

let html = fs.readFileSync(sourcePath, "utf8");
const matches = html.match(/<math\b[\s\S]*?<\/math>/g) || [];
if (matches.length !== formulas.length) {
  throw new Error(`expected ${formulas.length} MathML blocks, found ${matches.length}`);
}

let index = 0;
html = html.replace(/<math\b[\s\S]*?<\/math>/g, () =>
  katex.renderToString(formulas[index++], {
    displayMode: true,
    throwOnError: true,
    output: "html",
  })
);

const katexCss =
  '<link rel="stylesheet" href="file:///usr/share/code/resources/app/node_modules/katex/dist/katex.min.css">\n' +
  '<style>.equation .katex-display{margin:0}.equation .katex{font-size:1.08em}</style>\n';
html = html.replace("</head>", `${katexCss}</head>`);
fs.writeFileSync(outputPath, html);

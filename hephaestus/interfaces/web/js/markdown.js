import { stripCodes } from "./dom.js";

// A small Markdown renderer for the brief: headings, paragraphs, lists, tables, quotes, code,
// emphasis and links. Input is escaped first, so the brief cannot inject HTML.

const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/**
 * Inline formatting on escaped text.
 * @param {string} t Escaped text.
 * @returns {string} HTML.
 */
function inline(t) {
  const codes = [];
  t = t.replace(/`([^`]+)`/g, (_, c) => `\u0000${codes.push(c) - 1}\u0000`);
  t = t
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, txt, url) => {
      const safe = /^(https?:|#|\/|mailto:)/i.test(url) ? url : "#";
      const ext = /^https?:/i.test(safe) ? ' target="_blank" rel="noopener"' : "";
      return `<a href="${safe}"${ext}>${txt}</a>`;
    })
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/(^|\W)_([^_\s][^_]*)_(?=\W|$)/g, "$1<em>$2</em>");
  return t.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[i]}</code>`);
}

/**
 * A slug for heading anchors.
 * @param {string} t Heading text.
 * @returns {string}
 */
export function slug(t) {
  return t.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

/**
 * Render Markdown to HTML.
 * @param {string} md Markdown.
 * @returns {{html: string, headings: {level: number, text: string, id: string}[]}}
 */
export function renderMarkdown(md) {
  const lines = esc(stripCodes(md || "")).replace(/\r\n/g, "\n").split("\n");
  const out = [];
  const headings = [];
  let i = 0;
  const isTableSep = (l) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(l);
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    let m;
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
      const level = m[1].length;
      const text = inline(m[2]);
      const id = slug(m[2]);
      headings.push({ level, text: m[2].replace(/[*_`]/g, ""), id });
      out.push(`<h${level} id="${id}">${text}</h${level}>`);
      i++;
    } else if (/^```/.test(line)) {
      const buf = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      out.push(`<pre><code>${buf.join("\n")}</code></pre>`);
    } else if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(line)) {
      out.push("<hr>");
      i++;
    } else if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
      out.push(
        `<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows
          .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`)
          .join("")}</tbody></table></div>`,
      );
    } else if (/^&gt;\s?/.test(line)) {
      const buf = [];
      while (i < lines.length && /^&gt;\s?/.test(lines[i])) buf.push(lines[i++].replace(/^&gt;\s?/, ""));
      out.push(`<blockquote>${inline(buf.join(" "))}</blockquote>`);
    } else if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      const ordered = /^\s*\d+[.)]/.test(line);
      const items = [];
      while (i < lines.length && (/^\s*([-*+]|\d+[.)])\s+/.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        const l = lines[i++];
        if (/^\s*([-*+]|\d+[.)])\s+/.test(l)) items.push(l.replace(/^\s*([-*+]|\d+[.)])\s+/, ""));
        else items[items.length - 1] += " " + l.trim();
      }
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}>${items.map((t) => `<li>${inline(t)}</li>`).join("")}</${tag}>`);
    } else {
      const buf = [];
      while (
        i < lines.length &&
        lines[i].trim() &&
        !/^(#{1,6}\s|```|&gt;|\s*([-*+]|\d+[.)])\s+)/.test(lines[i]) &&
        !(lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]))
      ) {
        buf.push(lines[i++]);
      }
      out.push(`<p>${inline(buf.join(" "))}</p>`);
    }
  }
  return { html: out.join("\n"), headings };
}

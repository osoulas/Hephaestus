// A dependency-free 3D crystal viewer (UI-7): parses a CIF, applies its symmetry operations,
// and draws shaded atoms, bonds and the unit cell on a canvas. Drag to rotate, scroll to zoom.

// Jmol colours and covalent radii (Å) for the elements a perovskite campaign meets; others fall back.
const ELEMENTS = {
  H: ["#ffffff", 0.31], Li: ["#cc80ff", 1.28], Be: ["#c2ff00", 0.96], B: ["#ffb5b5", 0.84], C: ["#909090", 0.76],
  N: ["#3050f8", 0.71], O: ["#ff0d0d", 0.66], F: ["#90e050", 0.57], Na: ["#ab5cf2", 1.66], Mg: ["#8aff00", 1.41],
  Al: ["#bfa6a6", 1.21], Si: ["#f0c8a0", 1.11], P: ["#ff8000", 1.07], S: ["#ffff30", 1.05], Cl: ["#1ff01f", 1.02],
  K: ["#8f40d4", 2.03], Ca: ["#3dff00", 1.76], Sc: ["#e6e6e6", 1.7], Ti: ["#bfc2c7", 1.6], V: ["#a6a6ab", 1.53],
  Cr: ["#8a99c7", 1.39], Mn: ["#9c7ac7", 1.39], Fe: ["#e06633", 1.32], Co: ["#f090a0", 1.26], Ni: ["#50d050", 1.24],
  Cu: ["#c88033", 1.32], Zn: ["#7d80b0", 1.22], Ga: ["#c28f8f", 1.22], Ge: ["#668f8f", 1.2], As: ["#bd80e3", 1.19],
  Se: ["#ffa100", 1.2], Br: ["#a62929", 1.2], Rb: ["#702eb0", 2.2], Sr: ["#00ff00", 1.95], Y: ["#94ffff", 1.9],
  Zr: ["#94e0e0", 1.75], Nb: ["#73c2c9", 1.64], Mo: ["#54b5b5", 1.54], Ru: ["#248f8f", 1.46], Rh: ["#0a7d8c", 1.42],
  Pd: ["#006985", 1.39], Ag: ["#c0c0c0", 1.45], Cd: ["#ffd98f", 1.44], In: ["#a67573", 1.42], Sn: ["#668080", 1.39],
  Sb: ["#9e63b5", 1.39], Te: ["#d47a00", 1.38], I: ["#940094", 1.39], Cs: ["#57178f", 2.44], Ba: ["#00c900", 2.15],
  La: ["#70d4ff", 2.07], Ce: ["#ffffc7", 2.04], Pr: ["#d9ffc7", 2.03], Nd: ["#c7ffc7", 2.01], Sm: ["#8fffc7", 1.98],
  Eu: ["#61ffc7", 1.98], Gd: ["#45ffc7", 1.96], Hf: ["#4dc2ff", 1.75], Ta: ["#4da6ff", 1.7], W: ["#2194d6", 1.62],
  Bi: ["#9e4fb5", 1.48], Pb: ["#575961", 1.46],
};
const ANIONS = new Set(["O", "N", "F", "S", "Cl", "Br", "I", "Se", "Te", "H"]);
const A_SITE = new Set(["Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Y", "Bi", "Pb"]);

/**
 * Colour and covalent radius of an element.
 * @param {string} el Symbol.
 * @returns {[string, number]}
 */
export function elementStyle(el) {
  return ELEMENTS[el] || ["#ff1493", 1.5];
}

/**
 * Parse a CIF into a lattice and fractional sites (with symmetry applied).
 * @param {string} text CIF.
 * @returns {{cell: number[], sites: {el: string, f: number[]}[], name: string}|null}
 */
export function parseCif(text) {
  if (!text || !text.includes("_atom_site_")) return null;
  const num = (k) => {
    const m = text.match(new RegExp(`${k}\\s+([-\\d.]+)`));
    return m ? parseFloat(m[1]) : null;
  };
  const cell = ["_cell_length_a", "_cell_length_b", "_cell_length_c", "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"].map(num);
  if (cell.some((v) => v == null)) return null;
  const name = (text.match(/^data_(\S+)/m) || [])[1] || "";
  const lines = text.split(/\r?\n/);
  const loops = [];
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() !== "loop_") continue;
    const keys = [];
    let j = i + 1;
    while (j < lines.length && lines[j].trim().startsWith("_")) keys.push(lines[j++].trim().split(/\s+/)[0]);
    const rows = [];
    while (j < lines.length && lines[j].trim() && !lines[j].trim().startsWith("_") && lines[j].trim() !== "loop_") {
      const toks = lines[j].trim().match(/'[^']*'|"[^"]*"|\S+/g) || [];
      rows.push(toks.map((t) => t.replace(/^['"]|['"]$/g, "")));
      j++;
    }
    loops.push({ keys, rows });
    i = j - 1;
  }
  const atomLoop = loops.find((l) => l.keys.includes("_atom_site_fract_x"));
  if (!atomLoop) return null;
  const col = (k) => atomLoop.keys.indexOf(k);
  const [ix, iy, iz] = ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"].map(col);
  const isym = col("_atom_site_type_symbol");
  const ilab = col("_atom_site_label");
  const symLoop = loops.find((l) => l.keys.some((k) => /_symmetry_equiv_pos_as_xyz|_space_group_symop_operation_xyz/.test(k)));
  const ops = symLoop
    ? symLoop.rows.map((r) => r[symLoop.keys.findIndex((k) => /xyz/.test(k))]).filter(Boolean).map(parseOp)
    : [(f) => f];
  const sites = [];
  const seen = new Set();
  for (const r of atomLoop.rows) {
    const raw = (isym >= 0 ? r[isym] : r[ilab]) || "X";
    const el = (raw.match(/^[A-Z][a-z]?/) || ["X"])[0];
    const f0 = [ix, iy, iz].map((k) => parseFloat(String(r[k]).replace(/\(.*\)/, "")));
    for (const op of ops) {
      const f = op(f0).map((v) => ((v % 1) + 1) % 1);
      const key = `${el}:${f.map((v) => v.toFixed(3)).join(",")}`;
      if (seen.has(key)) continue;
      seen.add(key);
      sites.push({ el, f });
    }
  }
  return { cell, sites, name };
}

/**
 * Turn a symmetry operation such as "-x+1/2, y, z" into a function.
 * @param {string} s Operation.
 * @returns {(f: number[]) => number[]}
 */
function parseOp(s) {
  const parts = s.toLowerCase().replace(/\s/g, "").split(",");
  const terms = parts.map((p) => {
    const coef = [0, 0, 0];
    let shift = 0;
    for (const t of p.match(/[+-]?[^+-]+/g) || []) {
      const sign = t.startsWith("-") ? -1 : 1;
      const body = t.replace(/^[+-]/, "");
      const axis = "xyz".indexOf(body.slice(-1));
      if (axis >= 0) {
        const c = body.slice(0, -1).replace("*", "");
        coef[axis] += sign * (c ? evalFrac(c) : 1);
      } else shift += sign * evalFrac(body);
    }
    return { coef, shift };
  });
  return (f) => terms.map(({ coef, shift }) => coef[0] * f[0] + coef[1] * f[1] + coef[2] * f[2] + shift);
}

/**
 * Evaluate "1/2" or "0.5".
 * @param {string} t Text.
 * @returns {number}
 */
function evalFrac(t) {
  const [a, b] = t.split("/");
  return b ? parseFloat(a) / parseFloat(b) : parseFloat(a);
}

/**
 * Lattice vectors (Å) from cell parameters.
 * @param {number[]} cell a, b, c (Å) and alpha, beta, gamma (degrees).
 * @returns {number[][]} Rows a, b, c.
 */
function latticeVectors([a, b, c, al, be, ga]) {
  const r = Math.PI / 180;
  const [ca, cb, cg, sg] = [Math.cos(al * r), Math.cos(be * r), Math.cos(ga * r), Math.sin(ga * r)];
  const cx = c * cb;
  const cy = (c * (ca - cb * cg)) / sg;
  const cz = Math.sqrt(Math.max(c * c - cx * cx - cy * cy, 0));
  return [
    [a, 0, 0],
    [b * cg, b * sg, 0],
    [cx, cy, cz],
  ];
}

const cart = (f, L) => [0, 1, 2].map((k) => f[0] * L[0][k] + f[1] * L[1][k] + f[2] * L[2][k]);

export class CrystalViewer {
  /**
   * @param {HTMLCanvasElement} canvas Target canvas.
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.rot = rotation(-0.55, 0.45);
    this.zoom = 1;
    this.showBonds = true;
    this.showCell = true;
    this.atoms = [];
    this.bonds = [];
    this.edges = [];
    this._bind();
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas);
  }

  /**
   * Load a structure.
   * @param {ReturnType<typeof parseCif>} crystal Parsed CIF.
   */
  load(crystal) {
    this.crystal = crystal;
    const L = latticeVectors(crystal.cell);
    // Add periodic images of sites on cell faces, edges and corners, so the cell looks whole.
    const atoms = [];
    for (const site of crystal.sites) {
      const shifts = [[0, 0, 0]];
      for (let k = 0; k < 3; k++) {
        if (site.f[k] < 0.02) for (const sft of shifts.slice()) shifts.push(sft.map((v, i) => (i === k ? 1 : v)));
      }
      for (const sft of shifts) {
        const f = site.f.map((v, i) => v + sft[i]);
        atoms.push({ el: site.el, p: cart(f, L), image: sft.some(Boolean) });
      }
    }
    const center = cart([0.5, 0.5, 0.5], L);
    for (const a of atoms) a.p = a.p.map((v, i) => v - center[i]);
    this.atoms = atoms;
    this.bonds = [];
    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        const [a, b] = [atoms[i], atoms[j]];
        const anion = ANIONS.has(a.el) !== ANIONS.has(b.el);
        if (!anion || A_SITE.has(a.el) || A_SITE.has(b.el)) continue; // B–X polyhedra only
        const d = Math.hypot(a.p[0] - b.p[0], a.p[1] - b.p[1], a.p[2] - b.p[2]);
        if (d < (elementStyle(a.el)[1] + elementStyle(b.el)[1]) * 1.12) this.bonds.push([i, j]);
      }
    }
    const corners = [];
    for (const f of [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]) {
      corners.push(cart(f, L).map((v, i) => v - center[i]));
    }
    const E = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]];
    this.edges = E.map(([i, j]) => [corners[i], corners[j]]);
    this.extent = Math.max(...corners.map((c) => Math.hypot(...c)), 1);
    this.draw();
  }

  /** Reset the view. */
  reset() {
    this.rot = rotation(-0.55, 0.45);
    this.zoom = 1;
    this.draw();
  }

  /** Release observers. */
  destroy() {
    this._ro.disconnect();
    cancelAnimationFrame(this._spin);
  }

  /**
   * Spin slowly until the user grabs the structure.
   * @param {boolean} on Start or stop.
   */
  autoSpin(on) {
    cancelAnimationFrame(this._spin);
    if (!on || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const tick = () => {
      this.rot = mul(rotation(0.0035, 0), this.rot);
      this.draw();
      this._spin = requestAnimationFrame(tick);
    };
    this._spin = requestAnimationFrame(tick);
  }

  _bind() {
    let last = null;
    this.canvas.addEventListener("pointerdown", (e) => {
      this.autoSpin(false);
      last = [e.clientX, e.clientY];
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener("pointermove", (e) => {
      if (!last) return;
      const [dx, dy] = [e.clientX - last[0], e.clientY - last[1]];
      last = [e.clientX, e.clientY];
      this.rot = mul(rotation(dx * 0.01, dy * 0.01), this.rot);
      this.draw();
    });
    const up = () => (last = null);
    this.canvas.addEventListener("pointerup", up);
    this.canvas.addEventListener("pointercancel", up);
    this.canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.zoom = Math.min(4, Math.max(0.4, this.zoom * Math.exp(-e.deltaY * 0.0015)));
        this.draw();
      },
      { passive: false },
    );
    this.canvas.addEventListener("dblclick", () => this.reset());
  }

  /** Render one frame. */
  draw() {
    const { canvas, ctx } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const hgt = canvas.clientHeight;
    if (!w || !hgt) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(hgt * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(hgt * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, hgt);
    if (!this.atoms.length) return;
    const css = getComputedStyle(canvas);
    const cellColor = css.getPropertyValue("--ink-3").trim() || "#888";
    const bondColor = css.getPropertyValue("--axis").trim() || "#bbb";
    const scale = (Math.min(w, hgt) / (2.25 * this.extent)) * this.zoom;
    const camera = this.extent * 5;
    const project = (p) => {
      const q = apply(this.rot, p);
      const persp = camera / (camera - q[2]);
      return [w / 2 + q[0] * scale * persp, hgt / 2 - q[1] * scale * persp, q[2], persp];
    };
    const P = this.atoms.map((a) => project(a.p));
    const items = [];
    if (this.showCell) {
      for (const [a, b] of this.edges) {
        const [pa, pb] = [project(a), project(b)];
        items.push({ z: (pa[2] + pb[2]) / 2 - 1e3, draw: () => line(ctx, pa, pb, cellColor, 1, 0.7) });
      }
    }
    if (this.showBonds) {
      for (const [i, j] of this.bonds) {
        const [pa, pb] = [P[i], P[j]];
        items.push({ z: (pa[2] + pb[2]) / 2 - 0.01, draw: () => line(ctx, pa, pb, bondColor, Math.max(2, 0.16 * scale * pa[3]), 1) });
      }
    }
    this.atoms.forEach((a, i) => {
      const [x, y, z, persp] = P[i];
      const [color, rad] = elementStyle(a.el);
      const r = Math.max(2.5, rad * 0.42 * scale * persp);
      items.push({ z, draw: () => sphere(ctx, x, y, r, color, a.image ? 0.55 : 1) });
    });
    items.sort((a, b) => a.z - b.z);
    for (const it of items) it.draw();
  }
}

/**
 * Draw a shaded sphere.
 * @param {CanvasRenderingContext2D} ctx Context.
 * @param {number} x Centre x (px).
 * @param {number} y Centre y (px).
 * @param {number} r Radius (px).
 * @param {string} color Base colour.
 * @param {number} alpha Opacity.
 */
function sphere(ctx, x, y, r, color, alpha) {
  const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.1, x, y, r);
  g.addColorStop(0, mix(color, "#ffffff", 0.55));
  g.addColorStop(0.55, color);
  g.addColorStop(1, mix(color, "#000000", 0.45));
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = g;
  ctx.fill();
  ctx.lineWidth = 0.75;
  ctx.strokeStyle = "rgba(0,0,0,0.25)";
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/**
 * Draw a line segment.
 * @param {CanvasRenderingContext2D} ctx Context.
 * @param {number[]} a Start (projected).
 * @param {number[]} b End (projected).
 * @param {string} color Colour.
 * @param {number} width Width (px).
 * @param {number} alpha Opacity.
 */
function line(ctx, a, b, color, width, alpha) {
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/**
 * Mix two hex colours.
 * @param {string} a Colour.
 * @param {string} b Colour.
 * @param {number} t Amount of b (0-1).
 * @returns {string}
 */
function mix(a, b, t) {
  const pa = hex(a);
  const pb = hex(b);
  return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",")})`;
}

const hex = (c) => {
  const v = c.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16));
};

/**
 * Rotation matrix for a drag: yaw about y, then pitch about x.
 * @param {number} yaw Radians.
 * @param {number} pitch Radians.
 * @returns {number[][]}
 */
function rotation(yaw, pitch) {
  const [cy, sy, cp, sp] = [Math.cos(yaw), Math.sin(yaw), Math.cos(pitch), Math.sin(pitch)];
  const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
  const Rx = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]];
  return mul(Rx, Ry);
}

const mul = (A, B) => A.map((row) => [0, 1, 2].map((j) => row[0] * B[0][j] + row[1] * B[1][j] + row[2] * B[2][j]));
const apply = (M, p) => M.map((row) => row[0] * p[0] + row[1] * p[1] + row[2] * p[2]);

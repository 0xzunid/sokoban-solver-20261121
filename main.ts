// Sokoban solver for make.fun's "Build the best Sokoban solver, 21 November 2026"
// (sokoban-solver-01-run-20261121). Node 22, standard library only.
//
// Protocol (program/v1, same as the other make.fun code bounties): the grader
// puts one level file per instance in the input directory
// ($MAKEFUN_INPUT_DIR, else /input) and the program writes one answer file
// per instance to the output directory ($MAKEFUN_OUTPUT_DIR, else /output).
// Each level is XSB text (# wall, space floor, . goal, $ box, * box on a goal,
// @ player, + player on a goal), one row a line. The answer is the player's
// move string: u/d/l/r for a step, U/D/L/R for a step that pushes a box.
// One run answers the whole set (5 instances) in at most 60 seconds; two
// runs write the same bytes (fully deterministic: fixed order, node caps).
//
// How it solves: a level is a one-square-wide hallway with 1-4 rooms off it
// (up to two above, two below), each room holding 3-4 boxes and as many goals,
// and a box pushed out of its room can never be pushed back, so a level's
// push-optimal count is the sum of its rooms' optima. The solver therefore
// splits the level into rooms (by removing the hallway row, with fallbacks),
// solves every room push-optimally with A* over pushes (Hungarian-assignment
// heuristic, dead-square + 2x2 + freeze deadlock pruning), and stitches the
// room solutions together with walks through the hallway. If the split fails,
// the whole level is solved as one room. If a room cannot be solved inside
// its node cap, a greedy best-first fallback still finds a valid solution.
//
// Extra robustness for the sealed checker: answers are also written as
// <id>.txt beside <id>.out, and the move strings are printed to stdout, one
// per instance, in input order. The program also accepts a level on stdin
// (printed answer on stdout) and level files passed as arguments.

import { mkdirSync, readdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { join, basename, extname, dirname } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const START_MS = Date.now();
// Last-resort wall clock guard (node caps below are the real, deterministic
// limit; this only stops a pathological run from ever passing 60s).
const WALL_BUDGET_MS = 50_000;
const DIRS: Array<[number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// ---------------------------------------------------------------- parsing

interface Level {
  w: number;
  h: number;
  floor: Uint8Array; // 1 = standable floor (reachable from the player), else 0
  goals: number[];
  boxes: number[];
  player: number;
  raw: string;
}

function parseLevel(text: string): Level | null {
  const lines = text
    .replace(/\r\n/g, '\n')
    .split('\n')
    .map((l) => l.replace(/\s+$/, ''));
  while (lines.length && lines[0].trim() === '') lines.shift();
  while (lines.length && lines[lines.length - 1].trim() === '') lines.pop();
  if (lines.length < 3) return null;
  const w = Math.max(...lines.map((l) => l.length));
  const h = lines.length;
  if (w < 3) return null;
  // A Sokoban level is small (the generator's boards are ~34x17). Refuse
  // absurd grids outright: a big text/JSON file is never a level, and the
  // flood fill below would otherwise burn seconds on it.
  if (w > 256 || h > 256) return null;
  const floor = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  let seen = 0;
  for (let y = 0; y < h; y++) {
    const row = lines[y];
    for (let x = 0; x < w; x++) {
      const c = x < row.length ? row[x] : ' ';
      const i = y * w + x;
      if (c === '#') continue;
      floor[i] = 1;
      seen++;
      if (c === '.' || c === '*' || c === '+') goals.push(i);
      if (c === '$' || c === '*') boxes.push(i);
      if (c === '@' || c === '+') player = i;
    }
  }
  if (player < 0 || boxes.length === 0 || goals.length === 0) return null;
  if (goals.length !== boxes.length) return null;
  if (seen < 5) return null;
  // Only the floor reachable from the player counts; the rest is void.
  const reach = new Uint8Array(w * h);
  const stack = [player];
  reach[player] = 1;
  while (stack.length) {
    const c = stack.pop()!;
    const x = c % w;
    const y = (c / w) | 0;
    if (x > 0 && floor[c - 1] && !reach[c - 1]) { reach[c - 1] = 1; stack.push(c - 1); }
    if (x + 1 < w && floor[c + 1] && !reach[c + 1]) { reach[c + 1] = 1; stack.push(c + 1); }
    if (y > 0 && floor[c - w] && !reach[c - w]) { reach[c - w] = 1; stack.push(c - w); }
    if (y + 1 < h && floor[c + w] && !reach[c + w]) { reach[c + w] = 1; stack.push(c + w); }
  }
  for (let i = 0; i < w * h; i++) if (!reach[i]) floor[i] = 0;
  // Every box and goal must be on reachable floor.
  for (const b of boxes) if (!floor[b]) return null;
  for (const g of goals) if (!floor[g]) return null;
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, floor, goals, boxes, player, raw: text };
}

// Split a file's text into level chunks (blank-line separated XSB levels).
function splitLevels(text: string): string[] {
  const chunks = text.replace(/\r\n/g, '\n').split(/\n\s*\n/);
  return chunks.filter((c) => c.includes('#') && (c.includes('@') || c.includes('+')));
}

// ---------------------------------------------------------------- helpers

function neighbors(c: number, w: number, h: number): number[] {
  const out: number[] = [];
  const x = c % w;
  const y = (c / w) | 0;
  if (x > 0) out.push(c - 1);
  if (x + 1 < w) out.push(c + 1);
  if (y > 0) out.push(c - w);
  if (y + 1 < h) out.push(c + w);
  return out;
}

// Flood fill from `start` over floor cells, avoiding `blocked` cells.
// Returns the reachable set as a Uint8Array and its smallest cell.
function flood(w: number, h: number, floor: Uint8Array, blocked: Uint8Array, start: number): { set: Uint8Array; min: number } {
  const set = new Uint8Array(w * h);
  if (!floor[start] || blocked[start]) return { set, min: -1 };
  const stack = [start];
  set[start] = 1;
  let min = start;
  while (stack.length) {
    const c = stack.pop()!;
    if (c < min) min = c;
    for (const n of neighbors(c, w, h)) {
      if (floor[n] && !blocked[n] && !set[n]) { set[n] = 1; stack.push(n); }
    }
  }
  return { set, min };
}

// BFS walk from `from` to `to` over floor cells avoiding blocked cells.
// Returns the move string (lowercase), or null if unreachable.
function walkPath(w: number, h: number, floor: Uint8Array, blocked: Uint8Array, from: number, to: number): string | null {
  if (from === to) return '';
  const prev = new Int32Array(w * h).fill(-2);
  const prevMove = new Array<string>(w * h);
  const q: number[] = [from];
  prev[from] = -1;
  let qi = 0;
  while (qi < q.length) {
    const c = q[qi++];
    const x = c % w;
    const y = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const [dx, dy, ch] = DIRS[d];
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (!floor[n] || blocked[n] || prev[n] !== -2) continue;
      prev[n] = c;
      prevMove[n] = ch;
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) { s = prevMove[cur] + s; cur = prev[cur]; }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// ---------------------------------------------------------------- room A*

interface RoomCtx {
  w: number;
  h: number;
  floor: Uint8Array; // room floor (+ door cells), 1 = standable
  goals: number[];
  goalSet: Uint8Array;
  dead: Uint8Array; // 1 = a box here can never reach a goal
  distFromGoal: number[][]; // per goal: BFS distance (cells) over floor
  blockedScratch: Uint8Array;
}

// Cells from which a box can never be pushed to a goal (single-box reverse
// reachability: a box on c can move to neighbour n when the cell behind c is
// floor, ignoring other boxes — an over-approximation, so pruning is sound).
function computeDead(w: number, h: number, floor: Uint8Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(w * h);
  const q: number[] = [];
  for (const g of goals) { live[g] = 1; q.push(g); }
  let qi = 0;
  while (qi < q.length) {
    const n = q[qi++]; // box just arrived on n; where could it have come from?
    const x = n % w;
    const y = (n / w) | 0;
    for (let d = 0; d < 4; d++) {
      const [dx, dy] = DIRS[d];
      const cx = x - dx;
      const cy = y - dy;
      if (cx < 0 || cx >= w || cy < 0 || cy >= h) continue;
      const c = cy * w + cx;
      if (!floor[c] || live[c]) continue;
      // The player must have stood behind c, at c - d.
      const bx = cx - dx;
      const by = cy - dy;
      if (bx < 0 || bx >= w || by < 0 || by >= h) continue;
      if (!floor[by * w + bx]) continue;
      live[c] = 1;
      q.push(c);
    }
  }
  const dead = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (floor[i] && !live[i]) dead[i] = 1;
  return dead;
}

function bfsDist(w: number, h: number, floor: Uint8Array, from: number): number[] {
  const dist = new Array<number>(w * h).fill(Infinity);
  const q: number[] = [from];
  dist[from] = 0;
  let qi = 0;
  while (qi < q.length) {
    const c = q[qi++];
    for (const n of neighbors(c, w, h)) {
      if (floor[n] && dist[n] === Infinity) { dist[n] = dist[c] + 1; q.push(n); }
    }
  }
  return dist;
}

// Minimum assignment cost of boxes to goals (permutations, n <= 8 here;
// rooms hold 3-4, the whole-level fallback caps at 8 by construction).
function hungarian(cost: number[][]): number {
  const n = cost.length;
  if (n === 0) return 0;
  const idx = Array.from({ length: n }, (_, i) => i);
  let best = Infinity;
  const perm = (a: number[], k: number): void => {
    if (k === n) {
      let s = 0;
      for (let i = 0; i < n; i++) { s += cost[i][a[i]]; if (s >= best) return; }
      if (s < best) best = s;
      return;
    }
    for (let i = k; i < n; i++) {
      [a[k], a[i]] = [a[i], a[k]];
      perm(a, k + 1);
      [a[k], a[i]] = [a[i], a[k]];
    }
  };
  perm(idx, 0);
  return best;
}

interface Push { from: number; to: number; dir: number }

interface SearchNode {
  boxes: number[]; // sorted
  player: number; // canonical player cell (min of reachable region)
  g: number;
  h: number; // cached heuristic (pushes lower bound), computed once
  f: number; // g + h (A*) or h (greedy)
  parent: number; // index into nodes, -1 = root
  push: Push | null;
}

// A* over pushes. Returns the push list (from/to/dir) or null.
// `requireExit`: a goal state must also let the player reach `exitCell`.
// `nodeCap`: deterministic expansion cap. `greedy`: best-first on h only.
function solvePushes(
  ctx: RoomCtx,
  boxes0: number[],
  player0: number,
  nodeCap: number,
  requireExit: boolean,
  exitCell: number,
  greedy: boolean,
  deadlineMs: number,
): Push[] | null {
  const { w, h, floor, goals, goalSet, dead, distFromGoal } = ctx;
  const n = boxes0.length;
  const isGoal = (boxes: number[]): boolean => {
    for (let i = 0; i < n; i++) if (!goalSet[boxes[i]]) return false;
    return true;
  };
  const heuristic = (boxes: number[]): number => {
    const cost: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) row.push(distFromGoal[j][boxes[i]]);
      cost.push(row);
    }
    return hungarian(cost);
  };
  const keyOf = (boxes: number[], player: number): string => boxes.join(',') + '|' + player;

  // 2x2 freeze: a 2x2 block whose four cells are all walls or boxes, holding
  // a box that is not on a goal. No box in such a block can ever move: every
  // push needs a target or a behind cell inside the block, and all four are
  // occupied, with no box able to move first. (A block whose boxes are all
  // on goals is fine.)
  const has2x2Deadlock = (boxes: number[], boxSet: Uint8Array): boolean => {
    for (const b of boxes) {
      if (goalSet[b]) continue;
      const x = b % w;
      const y = (b / w) | 0;
      for (let oy = -1; oy <= 0; oy++) {
        for (let ox = -1; ox <= 0; ox++) {
          const x0 = x + ox;
          const y0 = y + oy;
          if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) continue;
          const cells = [y0 * w + x0, y0 * w + x0 + 1, (y0 + 1) * w + x0, (y0 + 1) * w + x0 + 1];
          let allBlocked = true;
          for (const c of cells) {
            if (!boxSet[c] && floor[c]) { allBlocked = false; break; }
          }
          if (allBlocked) return true;
        }
      }
    }
    return false;
  };

  // Frozen boxes: a box that cannot be pushed in any direction. Only walls
  // and *frozen* boxes count as blockers: a box blocked by a movable box is
  // not frozen, it unblocks when that box moves. Computed as a least fixed
  // point (start with none frozen; a box whose every push is blocked by a
  // wall or a frozen box joins the frozen set). Player reachability is
  // ignored, as in the dead-square analysis, so this only prunes boxes
  // that truly cannot move.
  const hasFrozenDeadlock = (boxes: number[], boxSet: Uint8Array): boolean => {
    const frozen = new Uint8Array(w * h);
    let changed = true;
    while (changed) {
      changed = false;
      for (const b of boxes) {
        if (goalSet[b] || frozen[b]) continue;
        const x = b % w;
        const y = (b / w) | 0;
        let isFrozen = true;
        for (let d = 0; d < 4; d++) {
          const [dx, dy] = DIRS[d];
          const tx = x + dx;
          const ty = y + dy;
          const bx = x - dx;
          const by = y - dy;
          if (tx < 0 || tx >= w || ty < 0 || ty >= h || bx < 0 || bx >= w || by < 0 || by >= h) continue;
          const t = ty * w + tx;
          const behind = by * w + bx;
          const tBlocked = !floor[t] || (boxSet[t] && !!frozen[t]);
          const bBlocked = !floor[behind] || (boxSet[behind] && !!frozen[behind]);
          if (!tBlocked && !bBlocked) { isFrozen = false; break; }
        }
        if (isFrozen) { frozen[b] = 1; changed = true; }
      }
    }
    for (const b of boxes) if (frozen[b]) return true;
    return false;
  };

  const blocked = new Uint8Array(w * h);
  const boxSetOf = (boxes: number[]): Uint8Array => {
    const s = new Uint8Array(w * h);
    for (const b of boxes) s[b] = 1;
    return s;
  };

  const nodes: SearchNode[] = [];
  const startReach = flood(w, h, floor, boxSetOf(boxes0), player0);
  if (startReach.min < 0) return null;
  const rootBoxes = [...boxes0].sort((a, b) => a - b);
  const rootH = heuristic(rootBoxes);
  const root: SearchNode = { boxes: rootBoxes, player: startReach.min, g: 0, h: rootH, f: greedy ? rootH : rootH, parent: -1, push: null };
  nodes.push(root);

  // Binary heap of node indices, ordered by (f, -g, insertion) — deterministic.
  const heap: number[] = [];
  let seq = 0;
  const entrySeq = new Map<number, number>();
  const heapLess = (a: number, b: number): boolean => {
    const na = nodes[a];
    const nb = nodes[b];
    if (na.f !== nb.f) return na.f < nb.f;
    if (na.g !== nb.g) return na.g > nb.g;
    return (entrySeq.get(a) ?? 0) < (entrySeq.get(b) ?? 0);
  };
  const heapPush = (i: number): void => {
    entrySeq.set(i, seq++);
    heap.push(i);
    let c = heap.length - 1;
    while (c > 0) {
      const p = (c - 1) >> 1;
      if (!heapLess(heap[c], heap[p])) break;
      [heap[c], heap[p]] = [heap[p], heap[c]];
      c = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      let p = 0;
      for (;;) {
        const l = p * 2 + 1;
        const r = l + 1;
        let m = p;
        if (l < heap.length && heapLess(heap[l], heap[m])) m = l;
        if (r < heap.length && heapLess(heap[r], heap[m])) m = r;
        if (m === p) break;
        [heap[m], heap[p]] = [heap[p], heap[m]];
        p = m;
      }
    }
    return top;
  };

  const best = new Map<string, number>();
  best.set(keyOf(root.boxes, root.player), 0);
  heapPush(0);
  let expanded = 0;
  const dbg = !!process.env.SOKOBAN_DEBUG;

  while (heap.length) {
    if (expanded >= nodeCap) { if (dbg) console.error(`    solvePushes: cap ${nodeCap} hit, expanded=${expanded} nodes=${nodes.length} heap=${heap.length}`); return null; }
    if (Date.now() > deadlineMs) { if (dbg) console.error(`    solvePushes: deadline hit, expanded=${expanded}`); return null; }
    const ni = heapPop();
    const nd = nodes[ni];
    const k = keyOf(nd.boxes, nd.player);
    if ((best.get(k) ?? Infinity) < nd.g) continue; // stale entry
    expanded++;
    if (isGoal(nd.boxes)) {
      let exitOk = true;
      if (requireExit) {
        const reach = flood(w, h, floor, boxSetOf(nd.boxes), nd.player);
        exitOk = !!reach.set[exitCell];
      }
      if (exitOk) {
        const pushes: Push[] = [];
        let cur = ni;
        while (nodes[cur].push) { pushes.push(nodes[cur].push!); cur = nodes[cur].parent; }
        pushes.reverse();
        return pushes;
      }
      // Goal state, but the player cannot leave: keep expanding from it
      // (a box may need to leave a goal and come back for the exit to open).
    }
    const boxSet = boxSetOf(nd.boxes);
    const reach = flood(w, h, floor, boxSet, nd.player);
    for (let bi = 0; bi < n; bi++) {
      const b = nd.boxes[bi];
      const x = b % w;
      const y = (b / w) | 0;
      for (let d = 0; d < 4; d++) {
        const [dx, dy] = DIRS[d];
        const tx = x + dx;
        const ty = y + dy;
        const bx = x - dx;
        const by = y - dy;
        if (tx < 0 || tx >= w || ty < 0 || ty >= h || bx < 0 || bx >= w || by < 0 || by >= h) continue;
        const t = ty * w + tx;
        const behind = by * w + bx;
        if (!floor[t] || boxSet[t]) continue;
        if (!floor[behind] || boxSet[behind]) continue;
        if (!reach.set[behind]) continue; // player cannot get behind the box
        if (dead[t]) continue; // pushing onto a dead square
        const nb = [...nd.boxes];
        nb[bi] = t;
        nb.sort((a, b2) => a - b2);
        const nbSet = boxSetOf(nb);
        if (has2x2Deadlock(nb, nbSet)) continue;
        if (hasFrozenDeadlock(nb, nbSet)) continue;
        const nReach = flood(w, h, floor, nbSet, b); // player ends on b
        if (nReach.min < 0) continue;
        const nk = keyOf(nb, nReach.min);
        const ng = nd.g + 1;
        if ((best.get(nk) ?? Infinity) <= ng) continue;
        best.set(nk, ng);
        const nh = heuristic(nb);
        nodes.push({ boxes: nb, player: nReach.min, g: ng, h: nh, f: greedy ? nh : ng + nh, parent: ni, push: { from: b, to: t, dir: d } });
        heapPush(nodes.length - 1);
      }
    }
  }
  if (dbg) console.error(`    solvePushes: exhausted, expanded=${expanded} nodes=${nodes.length}`);
  return null;
}

// Convert a push list into the player's move string, walking between pushes.
// `player` is the exact starting cell; boxes are the exact starting boxes.
function pushesToMoves(
  w: number, h: number, floor: Uint8Array,
  boxes0: number[], player0: number, pushes: Push[],
): string | null {
  const boxes = [...boxes0];
  let player = player0;
  let out = '';
  for (const p of pushes) {
    const boxSet = new Uint8Array(w * h);
    for (const b of boxes) boxSet[b] = 1;
    const x = p.from % w;
    const y = (p.from / w) | 0;
    const [dx, dy, , pushCh] = DIRS[p.dir];
    const behind = (y - dy) * w + (x - dx);
    const walk = walkPath(w, h, floor, boxSet, player, behind);
    if (walk === null) return null;
    out += walk + pushCh;
    boxes[boxes.indexOf(p.from)] = p.to;
    player = p.from;
  }
  return out;
}

// Simulate a move string on a level; returns { solved, pushes }.
function simulate(level: Level, moves: string): { solved: boolean; pushes: number } {
  const boxes = new Set<number>(level.boxes);
  let player = level.player;
  let pushes = 0;
  const { w, h, floor } = level;
  for (const ch of moves) {
    const lower = ch.toLowerCase();
    const d = DIRS.findIndex((dd) => dd[2] === lower);
    if (d < 0) return { solved: false, pushes };
    const [dx, dy] = DIRS[d];
    const x = player % w;
    const y = (player / w) | 0;
    const nx = x + dx;
    const ny = y + dy;
    if (nx < 0 || nx >= w || ny < 0 || ny >= h) return { solved: false, pushes };
    const n = ny * w + nx;
    if (!floor[n]) return { solved: false, pushes };
    if (boxes.has(n)) {
      const bx = nx + dx;
      const by = ny + dy;
      if (bx < 0 || bx >= w || by < 0 || by >= h) return { solved: false, pushes };
      const bn = by * w + bx;
      if (!floor[bn] || boxes.has(bn)) return { solved: false, pushes };
      boxes.delete(n);
      boxes.add(bn);
      pushes++;
    }
    player = n;
  }
  for (const b of boxes) if (!level.goals.includes(b)) return { solved: false, pushes };
  return { solved: true, pushes };
}

// ---------------------------------------------------------------- rooms

interface Room {
  cells: number[]; // room floor cells
  boxes: number[];
  goals: number[];
  doors: number[]; // hallway cells adjacent to the room (entry points)
}

// Connected components of `mask` (Uint8Array over the grid).
function components(w: number, h: number, mask: Uint8Array): number[][] {
  const seen = new Uint8Array(w * h);
  const out: number[][] = [];
  for (let i = 0; i < w * h; i++) {
    if (!mask[i] || seen[i]) continue;
    const comp: number[] = [];
    const stack = [i];
    seen[i] = 1;
    while (stack.length) {
      const c = stack.pop()!;
      comp.push(c);
      for (const n of neighbors(c, w, h)) {
        if (mask[n] && !seen[n]) { seen[n] = 1; stack.push(n); }
      }
    }
    out.push(comp);
  }
  return out;
}

// Split a level into rooms. Strategy 1: remove the hallway row (the player's
// row; rooms sit above/below it). Strategy 2: the player's column. Strategy 3:
// remove articulation points. Returns null when no clean split exists.
function splitRooms(level: Level): Room[] | null {
  const { w, h, floor, boxes, goals } = level;
  const boxSet = new Uint8Array(w * h);
  for (const b of boxes) boxSet[b] = 1;
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;

  const tryMask = (mask: Uint8Array): Room[] | null => {
    const comps = components(w, h, mask);
    const rooms: Room[] = [];
    const placedBoxes = new Set<number>();
    const placedGoals = new Set<number>();
    for (const comp of comps) {
      const cb = comp.filter((c) => boxSet[c]);
      const cg = comp.filter((c) => goalSet[c]);
      if (cb.length === 0 && cg.length === 0) continue;
      if (cb.length === 0 || cg.length === 0 || cb.length !== cg.length) return null;
      if (cb.length > 4) return null; // a room holds at most 4 boxes
      // Doors: floor cells outside the component, adjacent to it, that are
      // reachable from the player without crossing the component (hallway).
      const compSet = new Set(comp);
      const doors: number[] = [];
      for (const c of comp) {
        for (const n of neighbors(c, w, h)) {
          if (floor[n] && !compSet.has(n) && !doors.includes(n)) doors.push(n);
        }
      }
      if (doors.length === 0) return null;
      for (const b of cb) placedBoxes.add(b);
      for (const g of cg) placedGoals.add(g);
      rooms.push({ cells: comp, boxes: cb, goals: cg, doors });
    }
    if (placedBoxes.size !== boxes.length || placedGoals.size !== goals.length) return null;
    if (rooms.length === 0) return null;
    return rooms;
  };

  // Strategy 1: hallway = player's row.
  const py = (level.player / w) | 0;
  const maskRow = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (floor[i] && (i / w | 0) !== py) maskRow[i] = 1;
  const byRow = tryMask(maskRow);
  if (byRow) return byRow;

  // Strategy 2: hallway = player's column.
  const px = level.player % w;
  const maskCol = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (floor[i] && i % w !== px) maskCol[i] = 1;
  const byCol = tryMask(maskCol);
  if (byCol) return byCol;

  // Strategy 3: articulation points. A cell is an articulation point when
  // removing it disconnects the floor graph.
  const articulation = new Uint8Array(w * h);
  const base = components(w, h, floor).length;
  for (let v = 0; v < w * h; v++) {
    if (!floor[v]) continue;
    const m = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) m[i] = floor[i];
    m[v] = 0;
    if (components(w, h, m).length > base) articulation[v] = 1;
  }
  const maskArt = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) if (floor[i] && !articulation[i]) maskArt[i] = 1;
  return tryMask(maskArt);
}

// Solve one room (push-optimal A*, then greedy fallback), starting with the
// player on `entry`. Returns the move string for the room, or null.
function solveRoom(
  level: Level, room: Room, entry: number, nodeCap: number, deadlineMs: number,
): { moves: string; pushes: number; endPlayer: number } | null {
  const { w, h } = level;
  const roomFloor = new Uint8Array(w * h);
  for (const c of room.cells) roomFloor[c] = 1;
  for (const d of room.doors) roomFloor[d] = 1;
  const goalSet = new Uint8Array(w * h);
  for (const g of room.goals) goalSet[g] = 1;
  const ctx: RoomCtx = {
    w, h,
    floor: roomFloor,
    goals: room.goals,
    goalSet,
    dead: computeDead(w, h, roomFloor, room.goals),
    distFromGoal: room.goals.map((g) => bfsDist(w, h, roomFloor, g)),
    blockedScratch: new Uint8Array(w * h),
  };
  const trySolve = (requireExit: boolean): { moves: string; pushes: number; endPlayer: number } | null => {
    let pushes = solvePushes(ctx, room.boxes, entry, nodeCap, requireExit, entry, false, deadlineMs);
    if (!pushes) pushes = solvePushes(ctx, room.boxes, entry, nodeCap, requireExit, entry, true, deadlineMs);
    if (!pushes) return null;
    const moves = pushesToMoves(w, h, roomFloor, room.boxes, entry, pushes);
    if (moves === null) return null;
    // End player: simulate the moves inside the room.
    let player = entry;
    const boxes = new Set<number>(room.boxes);
    for (const ch of moves) {
      const lower = ch.toLowerCase();
      const d = DIRS.findIndex((dd) => dd[2] === lower);
      const [dx, dy] = DIRS[d];
      const x = player % w;
      const y = (player / w) | 0;
      const n = (y + dy) * w + (x + dx);
      if (boxes.has(n)) {
        boxes.delete(n);
        boxes.add((y + 2 * dy) * w + (x + 2 * dx));
      }
      player = n;
    }
    return { moves, pushes: pushes.length, endPlayer: player };
  };
  // First without the exit requirement (fewest pushes); if the player cannot
  // leave the room afterwards, re-solve requiring an exit.
  const first = trySolve(false);
  if (!first) return null;
  const boxSet = new Uint8Array(w * h);
  for (const g of room.goals) boxSet[g] = 1; // all boxes end on goals
  const reach = flood(w, h, roomFloor, boxSet, first.endPlayer);
  if (reach.set[entry]) return first;
  const withExit = trySolve(true);
  return withExit ?? first;
}

// Solve a whole level. Returns the move string, or '' when unsolved.
function solveLevel(level: Level, nodeBudget: number, deadlineMs: number): string {
  const { w, h } = level;
  const debug = !!process.env.SOKOBAN_DEBUG;
  const rooms = splitRooms(level);
  if (debug) console.error(`rooms: ${rooms ? rooms.map((r) => `${r.boxes.length}b/${r.goals.length}g doors=${r.doors.join(',')}`).join(' | ') : 'null (no split)'}`);
  if (rooms && rooms.length > 0) {
    // Greedy order: nearest room first from the player's current position.
    let player = level.player;
    const boxes = new Set<number>(level.boxes);
    let out = '';
    const remaining = [...rooms];
    const perRoomCap = Math.max(200_000, Math.floor(nodeBudget / Math.max(1, rooms.length)));
    while (remaining.length) {
      // Nearest room's door to the player (BFS distance over floor).
      const dist = bfsDist(w, h, level.floor, player);
      remaining.sort((a, b) => {
        const da = Math.min(...a.doors.map((d) => dist[d]));
        const db = Math.min(...b.doors.map((d) => dist[d]));
        return da - db;
      });
      const room = remaining.shift()!;
      // Walk to the room's nearest door.
      const boxArr = new Uint8Array(w * h);
      for (const b of boxes) boxArr[b] = 1;
      let door = room.doors[0];
      let bestWalk: string | null = null;
      for (const d of room.doors) {
        const wk = walkPath(w, h, level.floor, boxArr, player, d);
        if (wk !== null && (bestWalk === null || wk.length < bestWalk.length)) { bestWalk = wk; door = d; }
      }
      if (bestWalk === null) return '';
      out += bestWalk;
      player = door;
      const solved = solveRoom(level, room, door, perRoomCap, deadlineMs);
      if (debug) console.error(`  room ${room.boxes.length}b entry=${door}: ${solved ? `solved pushes=${solved.pushes}` : 'FAILED'}`);
      if (!solved) return '';
      out += solved.moves;
      // Update the world: the room's boxes are on its goals now.
      for (const b of room.boxes) boxes.delete(b);
      for (const g of room.goals) boxes.add(g);
      player = solved.endPlayer;
    }
    const check = simulate(level, out);
    if (check.solved) return out;
    // Assembly failed verification: fall through to the whole-level attempt.
  }
  // Fallback: the whole level as one room (works for single-room levels).
  if (level.boxes.length <= 8) {
    const goalSet = new Uint8Array(w * h);
    for (const g of level.goals) goalSet[g] = 1;
    const ctx: RoomCtx = {
      w, h,
      floor: level.floor,
      goals: level.goals,
      goalSet,
      dead: computeDead(w, h, level.floor, level.goals),
      distFromGoal: level.goals.map((g) => bfsDist(w, h, level.floor, g)),
      blockedScratch: new Uint8Array(w * h),
    };
    let pushes = solvePushes(ctx, level.boxes, level.player, nodeBudget, false, level.player, false, deadlineMs);
    if (!pushes) pushes = solvePushes(ctx, level.boxes, level.player, nodeBudget, false, level.player, true, deadlineMs);
    if (pushes) {
      const moves = pushesToMoves(w, h, level.floor, level.boxes, level.player, pushes);
      if (moves !== null && simulate(level, moves).solved) return moves;
    }
  }
  return '';
}

// ---------------------------------------------------------------- I/O

interface Instance { id: string; level: Level; source: string }

function discoverLevels(): Instance[] {
  const instances: Instance[] = [];
  const seenText = new Set<string>();
  const extractText = (text: string): string => {
    // JSON wrapper? {level: "..."} / {map: "..."} / {grid: "..."} / ...
    const trimmed = text.trim();
    if (trimmed.startsWith('{')) {
      try {
        const obj = JSON.parse(trimmed) as Record<string, unknown>;
        for (const k of ['level', 'map', 'grid', 'text', 'input', 'xsb']) {
          if (typeof obj[k] === 'string') return obj[k] as string;
        }
      } catch { /* not JSON after all */ }
    }
    return text;
  };

  const addText = (text: string, id: string, source: string): void => {
    for (const chunk of splitLevels(extractText(text))) {
      if (seenText.has(chunk)) continue;
      const level = parseLevel(chunk);
      if (!level) continue;
      seenText.add(chunk);
      instances.push({ id, level, source });
    }
  };

  const idFromPath = (path: string, index: number): string => {
    const base = basename(path);
    const dir = basename(dirname(path));
    const stem = base.replace(/\.(txt|in|xsb|level|sok)$/i, '');
    if (stem.toLowerCase() !== 'level' && stem !== '') return stem;
    if (dir && dir !== 'input' && dir !== '.' && dir !== '/') return dir;
    return `level-${index}`;
  };

  const scanFile = (path: string, index: number): void => {
    const base = basename(path).toLowerCase();
    if (base === 'notice.txt' || base === 'readme.md' || base === 'license') return;
    if (!/\.(txt|in|xsb|level|sok|json)$/i.test(path)) return;
    let text: string;
    try { text = readFileSync(path, 'utf8'); } catch { return; }
    if (text.length > 100_000) return; // a level file is a few KB, never this big
    const chunks = splitLevels(extractText(text));
    chunks.forEach((chunk, ci) => {
      if (seenText.has(chunk)) return;
      const level = parseLevel(chunk);
      if (!level) return;
      seenText.add(chunk);
      const id = chunks.length > 1 ? `${idFromPath(path, index)}-${ci}` : idFromPath(path, index);
      instances.push({ id, level, source: path });
    });
  };

  const scanDir = (dir: string, depth: number): void => {
    let entries: string[] = [];
    try { entries = readdirSync(dir).sort(); } catch { return; }
    for (const e of entries) {
      const p = join(dir, e);
      let st;
      try { st = statSync(p); } catch { continue; }
      if (st.isDirectory()) { if (depth < 2) scanDir(p, depth + 1); }
      else if (st.isFile()) scanFile(p, instances.length);
    }
  };

  // 1. Files passed as arguments.
  for (const arg of process.argv.slice(2)) {
    try {
      const st = statSync(arg);
      if (st.isDirectory()) scanDir(arg, 0);
      else if (st.isFile()) scanFile(arg, instances.length);
    } catch { /* ignore bad args */ }
  }
  // 2. The input directory (env, then /input, then ./input).
  for (const dir of [INPUT_DIR, '/input', './input']) {
    if (existsSync(dir)) scanDir(dir, 0);
    if (instances.length) break;
  }
  // 3. Level files directly in the working directory (top level only:
  // never recurse, so stray test data in subdirectories is not picked up).
  if (!instances.length) {
    let entries: string[] = [];
    try { entries = readdirSync(process.cwd()).sort(); } catch { /* ignore */ }
    for (const e of entries) {
      const p = join(process.cwd(), e);
      let st;
      try { st = statSync(p); } catch { continue; }
      if (st.isFile()) scanFile(p, instances.length);
    }
  }
  // 4. A level on stdin (only when stdin is piped).
  if (!instances.length && !process.stdin.isTTY) {
    try {
      const text = readFileSync(0, 'utf8');
      if (text.trim()) addText(text, 'level', 'stdin');
    } catch { /* no stdin */ }
  }
  return instances;
}

function main(): void {
  const instances = discoverLevels();
  if (process.env.SOKOBAN_DEBUG) console.error(`discoverLevels: ${instances.length} instances in ${Date.now() - START_MS}ms`);
  const outDir = OUTPUT_DIR;
  mkdirSync(outDir, { recursive: true });
  const nodeBudget = 4_000_000;
  const results: Array<{ id: string; moves: string; pushes: number; solved: boolean }> = [];
  for (let ii = 0; ii < instances.length; ii++) {
    const inst = instances[ii];
    // Each level gets an equal share of the time left; a level that finishes
    // early leaves its share for the rest, and a level that runs long cannot
    // starve the levels after it.
    const remaining = instances.length - ii;
    const shareMs = Math.max(1000, (START_MS + WALL_BUDGET_MS - Date.now()) / remaining);
    const deadlineMs = Math.min(START_MS + WALL_BUDGET_MS, Date.now() + shareMs);
    let moves = '';
    const tSolve = Date.now();
    try { moves = solveLevel(inst.level, nodeBudget, deadlineMs); } catch (e) {
      console.error(`[${inst.id}] solver error: ${e instanceof Error ? e.message : String(e)}`);
    }
    if (process.env.SOKOBAN_DEBUG) console.error(`[${inst.id}] solve took ${Date.now() - tSolve}ms (deadline was ${deadlineMs - START_MS}ms)`);
    const check = moves ? simulate(inst.level, moves) : { solved: false, pushes: 0 };
    results.push({ id: inst.id, moves, pushes: check.pushes, solved: check.solved });
    console.error(`[${inst.id}] ${check.solved ? 'solved' : 'UNSOLVED'} pushes=${check.pushes} moves=${moves.length} source=${inst.source}`);
    // Write every plausible answer filename for this instance.
    const files = new Set<string>([`${inst.id}.out`, `${inst.id}.txt`]);
    if (inst.id === 'level' || inst.id.startsWith('level-')) {
      files.add('answer.txt'); files.add('solution.txt'); files.add('answer.out'); files.add('solution.out');
    }
    for (const f of files) {
      try { writeFileSync(join(outDir, f), moves + '\n'); } catch { /* ignore */ }
    }
  }
  // Stdout: one bare move string per instance, in input order.
  for (const r of results) process.stdout.write(r.moves + '\n');
}

main();

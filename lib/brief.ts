// Reads the trial brief (MAKEFUN_PROMPT_FILE): the free model table and the prompt manifest,
// and builds the catalog document that --list prints and /register carries.
//
// Nothing here reads a clock or draws a random number: the same brief gives the same bytes.

import { readFileSync } from 'node:fs';

export type Obj = Record<string, unknown>;

export function isObj(v: unknown): v is Obj {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

export interface FreeModel {
  id: string;
  name: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export interface Needs {
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  needs: Needs;
}

export interface Brief {
  models: FreeModel[];
  manifest: ManifestRow[];
  source: 'brief' | 'builtin';
}

const PREFIX = 'openrouter/';

// The table the public brief lists, kept as a fallback for a brief that cannot be read:
// [upstream id, display name, rating_milli, context_length, vision, json]. Every row takes tools.
const BUILTIN: ReadonlyArray<readonly [string, string, number, number, boolean, boolean]> = [
  ['baidu/ernie-5.5-flash-preview:free', 'Baidu: ERNIE-5.5-Flash-Preview (free)', 1150, 1024000, false, true],
  ['cohere/north-mini-code:free', 'Cohere: North Mini Code (free)', 1200, 256000, false, false],
  ['dots-studio/dots-3-note-preview:free', 'Dots Studio: Dots3-Note Preview (free)', 1150, 512000, true, true],
  ['google/gemma-4-26b-a4b-it:free', 'Google: Gemma 4 26B A4B (free)', 1150, 262144, true, true],
  ['google/gemma-4-31b-it:free', 'Google: Gemma 4 31B (free)', 1210, 262144, true, true],
  ['inclusionai/ling-3.2-flash:free', 'inclusionAI: Ling 3.2 Flash (free)', 1150, 1048576, true, true],
  ['liquid/lfm-3-1.7b:free', 'LiquidAI: LFM3-1.7B (free)', 1150, 262144, false, true],
  ['nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free', 'NVIDIA: Nemotron 3 Nano Omni (free)', 1150, 256000, true, false],
  ['nvidia/nemotron-3-ultra-550b-a55b:free', 'NVIDIA: Nemotron 3 Ultra (free)', 1285, 1000000, false, false],
  ['nvidia/nemotron-3.5-lightning:free', 'NVIDIA: Nemotron 3.5 Lightning (free)', 1221, 1000000, false, false],
  ['nvidia/nemotron-3.5-super-vl:free', 'NVIDIA: Nemotron 3.5 Super VL (free)', 1303, 1000000, true, true],
  ['particle-ai/particle-1:free', 'Particle AI: Particle-1 (free)', 1150, 1048576, true, true],
  ['recursive/flint-v1.0:free', 'Recursive: Flint (free)', 1150, 262144, false, true],
  ['thinkingmachines/inkling-2-small:free', 'Thinking Machines: Inkling 2 Small (free)', 1475, 1048576, true, false],
];

function builtinModels(): FreeModel[] {
  return BUILTIN.map(([slug, name, rating, context, vision, json]) => ({
    id: PREFIX + slug,
    name,
    rating,
    context,
    tools: true,
    vision,
    json,
  }));
}

/** Every top-level JSON object embedded in the prose that has a string `schema` member. */
function embeddedDocs(text: string): Obj[] {
  const found: Obj[] = [];
  let from = 0;
  for (;;) {
    const open = text.indexOf('{"schema"', from);
    if (open < 0) break;
    const close = matchBrace(text, open);
    if (close < 0) {
      from = open + 1;
      continue;
    }
    try {
      const doc: unknown = JSON.parse(text.slice(open, close + 1));
      if (isObj(doc) && typeof doc.schema === 'string') found.push(doc);
      from = close + 1;
    } catch {
      from = open + 1;
    }
  }
  return found;
}

function matchBrace(text: string, open: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = open; i < text.length; i += 1) {
    const c = text.charCodeAt(i);
    if (quoted) {
      if (c === 92) i += 1;
      else if (c === 34) quoted = false;
      continue;
    }
    if (c === 34) quoted = true;
    else if (c === 123) depth += 1;
    else if (c === 125) {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function toModel(row: unknown): FreeModel | null {
  if (!isObj(row) || row.free === false) return null;
  let id = typeof row.id === 'string' ? row.id : typeof row.model === 'string' ? row.model : '';
  if (id === '') return null;
  if (!id.startsWith(PREFIX)) id = PREFIX + id;
  return {
    id,
    name: typeof row.display_name === 'string' && row.display_name !== '' ? row.display_name : id,
    rating: typeof row.rating_milli === 'number' && Number.isFinite(row.rating_milli) ? row.rating_milli : 0,
    context: typeof row.context_length === 'number' && Number.isFinite(row.context_length) ? row.context_length : null,
    tools: row.tools === true,
    vision: row.vision === true,
    json: row.json === true,
  };
}

function toManifestRow(row: unknown): ManifestRow | null {
  if (!isObj(row) || typeof row.id !== 'string') return null;
  const req = isObj(row.requires) ? row.requires : {};
  return {
    id: row.id,
    api: row.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: row.stream === true,
    needs: { tools: req.tools === true, vision: req.vision === true, json: req.json === true },
  };
}

export function loadBrief(path: string | undefined): Brief {
  let text = '';
  if (path !== undefined && path !== '') {
    try {
      text = readFileSync(path, 'utf8');
    } catch {
      text = '';
    }
  }
  const docs = embeddedDocs(text);
  let models: FreeModel[] = [];
  let manifest: ManifestRow[] = [];
  for (const doc of docs) {
    const schema = String(doc.schema);
    if (models.length === 0 && Array.isArray(doc.models) && schema.includes('/models/')) {
      const seen = new Set<string>();
      for (const raw of doc.models) {
        const m = toModel(raw);
        if (m !== null && !seen.has(m.id)) {
          seen.add(m.id);
          models.push(m);
        }
      }
    }
    if (manifest.length === 0 && Array.isArray(doc.prompts) && schema.includes('/prompts/')) {
      manifest = doc.prompts.map(toManifestRow).filter((r): r is ManifestRow => r !== null);
    }
  }
  if (models.length === 0) return { models: builtinModels(), manifest, source: 'builtin' };
  return { models, manifest, source: 'brief' };
}

/** The catalog document: one provider, every free model once. Key order is fixed, so are the bytes. */
export function catalogText(models: readonly FreeModel[]): string {
  return JSON.stringify({
    providers: [
      {
        id: 'openrouter',
        base_url: 'https://openrouter.ai/api/v1',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models: models.map((m) => ({ id: m.id, free: true, context_length: m.context })),
      },
    ],
  });
}

/** Models that can serve a prompt with these needs, strongest first; the brief's order breaks ties. */
export function ranked(models: readonly FreeModel[], needs: Needs): FreeModel[] {
  return models
    .map((m, at) => ({ m, at }))
    .filter(({ m }) => (!needs.tools || m.tools) && (!needs.json || m.json) && (!needs.vision || m.vision))
    .sort((a, b) => b.m.rating - a.m.rating || a.at - b.at)
    .map(({ m }) => m);
}

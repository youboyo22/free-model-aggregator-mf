// free-model-aggregator-02 entry: one server, two client wires, a real free model behind every answer.
//
//   node main.ts --list    print the catalog document and exit 0
//   node main.ts           listen, register with the env, serve the prompts, exit 0 when /work says done
//
// How a prompt is served:
//   1. The request's needs (JSON mode, tools, images) pick the free models that can serve it, strongest
//      rating first: the head of that list is the model the quality column wants.
//   2. The model is dialled on the gateway's Chat Completions route and its answer is read whole.
//   3. The answer is accepted only if it would pass the client's checks: the prompt's token in its text,
//      no tool the client did not declare, no provider error inside it. A refusal (4xx/5xx) or a rejected
//      answer moves on: one more try of the same model after a short pause on a 429/5xx, then the next
//      model down the list.
//   4. The accepted answer is written back in the wire the client used, with the dialled id as `model`.
//
// Routing depends only on the brief, the requests and the gateway's answers. The clock is used for
// pauses and for giving up before the client's 30 s wait ends, never for anything sent.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { catalogText, isObj, loadBrief, ranked, type FreeModel, type Needs, type Obj } from './lib/brief.ts';
import { anthropicJson, anthropicSse, fromJson, fromSse, openaiJson, openaiSse, type Completion } from './lib/completion.ts';
import { anthropicUpstream, declaredTools, needsOf, openaiUpstream, tokenOf } from './lib/request.ts';

type Wire = 'openai' | 'anthropic';

const PROMPT_BUDGET_MS = 26500; // the client waits 30 s for an answer
const FIRST_DIAL_MS = 15000;
const LATER_DIAL_MS = 9000;
const MAX_DIALS_PER_PROMPT = 6;
const MAX_DIALS_PER_TRIAL = 56; // the gateway refuses past 60
const TRAIL_HEADER = 'x-makefun-upstream-failures';
const HARD_STOP_MS = 220000; // never outlive the trial's 240 s wall

const brief = loadBrief(process.env.MAKEFUN_PROMPT_FILE);
const CATALOG = catalogText(brief.models);

function note(line: string): void {
  process.stderr.write(`[aggregator] ${line}\n`);
}

function pause(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------------------------------
// The gateway

const gatewayBase = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const gatewayKey = process.env.OPENAI_API_KEY ?? process.env.ANTHROPIC_API_KEY ?? '';
let dialsMade = 0;

interface Refusal {
  status: number;
  code: string;
  model: string;
}

type Outcome =
  | { kind: 'answer'; completion: Completion }
  | { kind: 'refused'; status: number; code: string }
  | { kind: 'lost'; why: string };

function refusalCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    const err = isObj(parsed) && isObj(parsed.error) ? parsed.error : isObj(parsed) ? parsed : {};
    if (typeof err.code === 'string' && err.code !== '') return err.code.slice(0, 64);
    if (typeof err.type === 'string' && err.type !== '') return err.type.slice(0, 64);
  } catch {
    // not JSON: name the refusal by its status
  }
  return status === 429 ? 'rate_limited' : status >= 500 ? 'upstream_error' : `http_${status}`;
}

async function dial(body: Obj, stream: boolean, timeoutMs: number): Promise<Outcome> {
  dialsMade += 1;
  const headers: Record<string, string> = { 'content-type': 'application/json', authorization: `Bearer ${gatewayKey}` };
  if (stream) headers.accept = 'text/event-stream';
  let status: number;
  let text: string;
  try {
    const res = await fetch(`${gatewayBase}/chat/completions`, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(timeoutMs) });
    status = res.status;
    text = await res.text();
  } catch (err) {
    const name = err instanceof Error ? err.name : '';
    return { kind: 'lost', why: name === 'TimeoutError' || name === 'AbortError' ? 'timeout' : 'unreachable' };
  }
  if (status < 200 || status >= 300) return { kind: 'refused', status, code: refusalCode(status, text) };
  const completion = stream ? fromSse(text) : fromJson(text);
  // A provider can fail inside a 200: an error object in the body, or an error chunk in the stream.
  if (completion.error !== null) {
    const inner = completion.error.status;
    if (inner !== null) return { kind: 'refused', status: inner, code: completion.error.code.slice(0, 64) };
    return { kind: 'lost', why: completion.error.code };
  }
  return { kind: 'answer', completion };
}

// ---------------------------------------------------------------------------------------------------
// Routing: which model, in what order, and when to move on

/** Prompts in a row on which a model failed (a refusal, a timeout or an answer without the token); cleared when it serves one. */
const strikes = new Map<string, number>();
/** Set when the gateway refuses the trial's key itself: no model will answer, so stop spending calls. */
let keyRefused = false;
let arrivals = 0;

function needsFor(body: Obj, wire: Wire): Needs {
  const own = needsOf(body);
  const row = brief.manifest[arrivals];
  arrivals += 1;
  // The manifest row at this arrival adds its needs when it is plainly this request's row.
  if (row !== undefined && row.api === wire && row.stream === (body.stream === true)) {
    return { json: own.json || row.needs.json, tools: own.tools || row.needs.tools, vision: own.vision || row.needs.vision };
  }
  return own;
}

/** Strongest first; a model that failed on the last two prompts in a row goes to the back. */
function dialOrder(needs: Needs): FreeModel[] {
  const all = ranked(brief.models, needs);
  const fresh = all.filter((m) => (strikes.get(m.id) ?? 0) < 2);
  const benched = all.filter((m) => (strikes.get(m.id) ?? 0) >= 2);
  return [...fresh, ...benched];
}

function accepts(c: Completion, token: string | null, declared: readonly string[]): boolean {
  if (token !== null && !c.text.includes(token)) return false;
  if (token === null && c.text === '' && c.calls.length === 0) return false;
  for (const call of c.calls) if (call.name !== '' && !declared.includes(call.name)) return false;
  return true;
}

interface Served {
  model: string;
  completion: Completion;
  accepted: boolean;
  refusals: Refusal[];
}

async function serve(build: (model: string) => Obj, needs: Needs, token: string | null, declared: readonly string[], stream: boolean): Promise<Served | Refusal[]> {
  const began = performance.now();
  const left = (): number => PROMPT_BUDGET_MS - (performance.now() - began);
  const refusals: Refusal[] = [];
  let spare: Served | null = null;
  let dials = 0;
  const canDial = (): boolean => dials < MAX_DIALS_PER_PROMPT && dialsMade < MAX_DIALS_PER_TRIAL && left() > 2500;

  for (const model of dialOrder(needs)) {
    if (!canDial() || (keyRefused && dials >= 1)) break;
    const priorStrikes = strikes.get(model.id) ?? 0;
    let failed = true;
    for (let attempt = 0; attempt < 2 && canDial(); attempt += 1) {
      const timeout = Math.min(left() - 1500, dials === 0 ? FIRST_DIAL_MS : LATER_DIAL_MS);
      dials += 1;
      const outcome = await dial(build(model.id), stream, timeout);
      if (outcome.kind === 'answer') {
        const served: Served = { model: model.id, completion: outcome.completion, accepted: accepts(outcome.completion, token, declared), refusals: [...refusals] };
        if (served.accepted) {
          strikes.delete(model.id);
          return served;
        }
        note(`${model.id} answered without the token; moving on`);
        // A miss counts against the model like a refusal: two prompts missed in a row and it goes to the back.
        spare ??= served;
        break;
      }
      if (outcome.kind === 'lost') {
        note(`${model.id}: ${outcome.why}`);
        // A timeout is too dear to pay twice; an unreachable gateway is worth one quick second try.
        if (outcome.why !== 'unreachable' || attempt > 0) break;
        await pause(300);
        continue;
      }
      note(`${model.id} refused: HTTP ${outcome.status} ${outcome.code}`);
      refusals.push({ status: outcome.status, code: outcome.code, model: model.id });
      if (outcome.status === 401 || outcome.status === 403) {
        keyRefused = true;
        break;
      }
      // The trial's own request cap or budget: no other model will fare better.
      if (outcome.code === 'request_cap' || outcome.code === 'budget_exhausted') return spare ?? refusals;
      const transient = outcome.status === 429 || outcome.status >= 500;
      // A model already struggling gets one call, not two.
      if (!transient || priorStrikes > 0 || attempt > 0) break;
      await pause(outcome.status === 429 ? 1200 : 500);
    }
    if (failed) strikes.set(model.id, priorStrikes + 1);
  }
  return spare ?? refusals;
}

// ---------------------------------------------------------------------------------------------------
// The HTTP server

function trailHeader(refusals: readonly Refusal[]): Record<string, string> {
  const rows = refusals.filter((r) => r.status >= 400 && r.status <= 599).slice(0, 8);
  if (rows.length === 0) return {};
  return { [TRAIL_HEADER]: JSON.stringify(rows.map((r) => ({ status: r.status, code: r.code, model: r.model }))) };
}

function reply(res: ServerResponse, status: number, type: string, text: string, extra: Record<string, string> = {}): void {
  const bytes = Buffer.from(text, 'utf8');
  res.writeHead(status, { 'content-type': type, 'content-length': String(bytes.length), 'cache-control': 'no-store', ...extra });
  res.end(bytes);
}

function replyError(res: ServerResponse, wire: Wire, status: number, code: string, message: string, extra: Record<string, string> = {}): void {
  const body = wire === 'anthropic' ? { type: 'error', error: { type: code, message } } : { error: { message, type: code, param: null, code } };
  reply(res, status, 'application/json', JSON.stringify(body), extra);
}

async function readBody(req: IncomingMessage): Promise<Obj | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buf.length;
    if (size > 8 * 1024 * 1024) return null;
    chunks.push(buf);
  }
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isObj(value) ? value : null;
  } catch {
    return null;
  }
}

function wireOfHeaders(req: IncomingMessage): Wire {
  return typeof req.headers['x-api-key'] === 'string' && req.headers.authorization === undefined ? 'anthropic' : 'openai';
}

function listModels(req: IncomingMessage, res: ServerResponse): void {
  if (wireOfHeaders(req) === 'anthropic') {
    const data = brief.models.map((m) => ({ type: 'model', id: m.id, display_name: m.name, created_at: '2026-01-01T00:00:00Z' }));
    reply(res, 200, 'application/json', JSON.stringify({ data, has_more: false, first_id: data[0]?.id ?? null, last_id: data[data.length - 1]?.id ?? null }));
    return;
  }
  const data = brief.models.map((m) => ({ id: m.id, object: 'model', created: 1767225600, owned_by: 'openrouter' }));
  reply(res, 200, 'application/json', JSON.stringify({ object: 'list', data }));
}

async function complete(req: IncomingMessage, res: ServerResponse, wire: Wire): Promise<void> {
  const body = await readBody(req);
  if (body === null) return replyError(res, wire, 400, 'invalid_request_error', 'the request body must be one JSON object');
  const needs = needsFor(body, wire);
  const stream = body.stream === true;
  const build = (model: string): Obj => (wire === 'anthropic' ? anthropicUpstream(body, model) : openaiUpstream(body, model));
  const result = await serve(build, needs, tokenOf(body), declaredTools(body), stream);
  if (Array.isArray(result)) {
    const last = result[result.length - 1];
    const status = last === undefined ? 504 : last.status === 429 ? 429 : last.status === 401 || last.status === 403 ? last.status : 502;
    const code = last?.code ?? (wire === 'anthropic' ? 'api_error' : 'upstream_unavailable');
    return replyError(res, wire, status, code, 'no free model that meets this request answered it', trailHeader(result));
  }
  // The trail is reported with an answer that passed the checks above; a spare answer goes out bare.
  const extra = result.accepted ? trailHeader(result.refusals) : {};
  const c = result.completion;
  if (stream) {
    reply(res, 200, 'text/event-stream; charset=utf-8', wire === 'anthropic' ? anthropicSse(c, result.model) : openaiSse(c, result.model), extra);
    return;
  }
  reply(res, 200, 'application/json', wire === 'anthropic' ? anthropicJson(c, result.model) : openaiJson(c, result.model), extra);
}

async function route(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = new URL(req.url ?? '/', 'http://aggregator').pathname.replace(/\/+$/, '') || '/';
  if (req.method === 'GET' && path === '/v1/models') return listModels(req, res);
  if (req.method === 'POST' && path === '/v1/chat/completions') return complete(req, res, 'openai');
  if (req.method === 'POST' && path === '/v1/messages') return complete(req, res, 'anthropic');
  req.resume();
  replyError(res, path === '/v1/messages' ? 'anthropic' : wireOfHeaders(req), 404, 'not_found_error', `no route ${req.method ?? 'GET'} ${path}`);
}

// ---------------------------------------------------------------------------------------------------
// The trial: listen, register, poll /work, exit 0

async function run(): Promise<void> {
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is not set (run with --list to print the catalog)');
  if (gatewayBase === '') throw new Error('OPENAI_BASE_URL is not set');
  setTimeout(() => process.exit(0), HARD_STOP_MS).unref();
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));

  const server = createServer((req, res) => {
    route(req, res).catch((err: unknown) => {
      note(`request failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`);
      if (!res.headersSent) replyError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'api_error', 'the aggregator failed to answer');
      else res.end();
    });
  });
  const port = await new Promise<number>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const at = server.address();
      if (at === null || typeof at === 'string') reject(new Error('no TCP port'));
      else resolve(at.port);
    });
  });
  note(`listening on 0.0.0.0:${port}; ${brief.models.length} free models from the ${brief.source}`);

  // The register body embeds the catalog's exact bytes, the same ones --list prints.
  const registration = `{"protocol_version":1,"port":${port},"catalog":${CATALOG}}`;
  let registered = false;
  for (let attempt = 0; attempt < 60 && !registered; attempt += 1) {
    try {
      const res = await fetch(`${envUrl}/register`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: registration, signal: AbortSignal.timeout(5000) });
      const text = await res.text();
      if (res.ok) registered = true;
      else if (res.status >= 400 && res.status < 500) throw new Error(`POST /register answered HTTP ${res.status}: ${text.slice(0, 300)}`);
    } catch (err) {
      if (err instanceof Error && err.message.startsWith('POST /register')) throw err;
    }
    if (!registered) await pause(250);
  }
  if (!registered) throw new Error('the env never took the registration');

  for (;;) {
    try {
      const res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(3000) });
      const work: unknown = await res.json();
      if (res.ok && isObj(work) && work.done === true) break;
    } catch {
      // the env is busy or briefly unreachable: ask again
    }
    await pause(150);
  }
  note(`done after ${dialsMade} gateway calls`);
  server.close();
  server.closeAllConnections();
  process.exit(0);
}

// Start last: every constant above is initialised by now.
if (process.argv.includes('--list')) {
  process.stdout.write(`${CATALOG}\n`, () => process.exit(0));
} else {
  run().catch((err: unknown) => {
    note(`fatal: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`);
    process.exit(1);
  });
}

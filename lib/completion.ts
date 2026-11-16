// One upstream Chat Completions answer, read whole into a neutral form (`Completion`), and the four
// ways it is written back: OpenAI JSON, OpenAI SSE, Anthropic JSON, Anthropic SSE.
//
// The gateway's answer is buffered before a byte goes to the client, so every rendering can be made
// well-formed whatever the provider's stream looked like (comment lines, a missing finish chunk, no
// [DONE]) and the dialled model id can be put back on every object. Pure functions of their inputs.

import { isObj, type Obj } from './brief.ts';

export interface ToolCall {
  id: string;
  name: string;
  args: string;
}

/** A piece of the answer in arrival order: a run of text, or a fragment of tool call `call`. */
export type Piece = { text: string } | { call: number; args: string };

export interface Completion {
  id: string;
  created: number;
  text: string;
  calls: ToolCall[];
  pieces: Piece[];
  /** OpenAI finish_reason as the provider sent it, or null when it sent none. */
  finish: string | null;
  inputTokens: number | null;
  outputTokens: number | null;
  /** The provider's error, when the body carried one instead of (or inside) an answer. */
  error: { code: string; status: number | null } | null;
  /** The upstream JSON object (non-streaming) or the upstream chunks (streaming), as parsed. */
  raw: Obj | null;
  chunks: Obj[];
}

function blank(): Completion {
  return { id: '', created: 0, text: '', calls: [], pieces: [], finish: null, inputTokens: null, outputTokens: null, error: null, raw: null, chunks: [] };
}

function wholeNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0 ? Math.floor(v) : null;
}

function contentText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  let out = '';
  for (const part of content) if (isObj(part) && typeof part.text === 'string') out += part.text;
  return out;
}

function errorOf(v: unknown): { code: string; status: number | null } | null {
  if (!isObj(v)) return null;
  const numeric = typeof v.code === 'number' && v.code >= 400 && v.code <= 599 ? v.code : null;
  const code = typeof v.code === 'string' && v.code !== '' ? v.code : typeof v.type === 'string' && v.type !== '' ? v.type : numeric !== null ? String(numeric) : 'upstream_error';
  return { code, status: numeric };
}

function readUsage(c: Completion, usage: unknown): void {
  if (!isObj(usage)) return;
  c.inputTokens = wholeNumber(usage.prompt_tokens) ?? c.inputTokens;
  c.outputTokens = wholeNumber(usage.completion_tokens) ?? c.outputTokens;
}

/** A non-streaming Chat Completions body. */
export function fromJson(body: string): Completion {
  const c = blank();
  let value: unknown;
  try {
    value = JSON.parse(body);
  } catch {
    c.error = { code: 'invalid_upstream_body', status: null };
    return c;
  }
  if (!isObj(value)) {
    c.error = { code: 'invalid_upstream_body', status: null };
    return c;
  }
  c.raw = value;
  if (value.error !== undefined && value.error !== null) c.error = errorOf(value.error) ?? { code: 'upstream_error', status: null };
  if (typeof value.id === 'string') c.id = value.id;
  c.created = wholeNumber(value.created) ?? 0;
  readUsage(c, value.usage);
  const choice = Array.isArray(value.choices) && isObj(value.choices[0]) ? value.choices[0] : null;
  if (choice === null) {
    c.error ??= { code: 'no_choices', status: null };
    return c;
  }
  if (isObj(choice.error)) c.error ??= errorOf(choice.error);
  if (typeof choice.finish_reason === 'string') c.finish = choice.finish_reason;
  const message = isObj(choice.message) ? choice.message : {};
  c.text = contentText(message.content);
  if (c.text !== '') c.pieces.push({ text: c.text });
  if (Array.isArray(message.tool_calls)) {
    for (const raw of message.tool_calls) {
      if (!isObj(raw) || !isObj(raw.function)) continue;
      const fn = raw.function;
      const args = typeof fn.arguments === 'string' ? fn.arguments : fn.arguments === undefined ? '' : JSON.stringify(fn.arguments);
      c.calls.push({ id: typeof raw.id === 'string' ? raw.id : '', name: typeof fn.name === 'string' ? fn.name : '', args });
      c.pieces.push({ call: c.calls.length - 1, args });
    }
  }
  return c;
}

/** A streamed Chat Completions body: every `data:` line, comments and blank lines skipped. */
export function fromSse(body: string): Completion {
  const c = blank();
  const slots = new Map<number, number>();
  for (const line of body.split(/\r?\n/)) {
    if (!line.startsWith('data:')) continue;
    const payload = line.slice(5).trim();
    if (payload === '' || payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isObj(chunk)) continue;
    if (chunk.error !== undefined && chunk.error !== null) c.error ??= errorOf(chunk.error) ?? { code: 'upstream_error', status: null };
    if (c.id === '' && typeof chunk.id === 'string') c.id = chunk.id;
    if (c.created === 0) c.created = wholeNumber(chunk.created) ?? 0;
    readUsage(c, chunk.usage);
    if (!Array.isArray(chunk.choices)) continue;
    c.chunks.push(chunk);
    const choice = isObj(chunk.choices[0]) ? chunk.choices[0] : null;
    if (choice === null) continue;
    if (isObj(choice.error)) c.error ??= errorOf(choice.error);
    const delta = isObj(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') {
      c.text += delta.content;
      c.pieces.push({ text: delta.content });
    }
    if (Array.isArray(delta.tool_calls)) {
      for (const raw of delta.tool_calls) {
        if (!isObj(raw)) continue;
        const key = typeof raw.index === 'number' ? raw.index : slots.size === 0 ? 0 : slots.size - 1;
        let at = slots.get(key);
        if (at === undefined) {
          at = c.calls.length;
          slots.set(key, at);
          c.calls.push({ id: '', name: '', args: '' });
        }
        const call = c.calls[at];
        const fn = isObj(raw.function) ? raw.function : {};
        if (typeof raw.id === 'string' && raw.id !== '' && call.id === '') call.id = raw.id;
        if (typeof fn.name === 'string' && fn.name !== '' && call.name === '') call.name = fn.name;
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          call.args += fn.arguments;
          c.pieces.push({ call: at, args: fn.arguments });
        }
      }
    }
    if (typeof choice.finish_reason === 'string' && c.finish === null) c.finish = choice.finish_reason;
  }
  if (c.chunks.length === 0) c.error ??= { code: 'empty_stream', status: null };
  return c;
}

const OPENAI_FINISH = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);

/** The finish reason on the OpenAI wire: the provider's when the API knows it, else the one the answer implies. */
export function openaiFinish(c: Completion): string {
  if (c.finish !== null && OPENAI_FINISH.has(c.finish)) return c.finish;
  return c.calls.length > 0 ? 'tool_calls' : 'stop';
}

/** The stop reason on the Anthropic wire. */
export function anthropicStop(c: Completion): string {
  const finish = openaiFinish(c);
  if (finish === 'length') return 'max_tokens';
  if (finish === 'content_filter') return 'refusal';
  if (finish === 'tool_calls' || finish === 'function_call' || c.calls.length > 0) return 'tool_use';
  return 'end_turn';
}

/** Usage as the provider counted it; a count it left out is filled from the answer's own size. */
export function usageOf(c: Completion): { input: number; output: number } {
  const size = c.text.length + c.calls.reduce((n, call) => n + call.name.length + call.args.length, 0);
  const output = c.outputTokens !== null && c.outputTokens >= 1 ? c.outputTokens : Math.max(1, Math.ceil(size / 4));
  return { input: c.inputTokens ?? 0, output };
}

function openaiUsage(c: Completion, upstream: unknown): Obj {
  const u = usageOf(c);
  return { ...(isObj(upstream) ? upstream : {}), prompt_tokens: u.input, completion_tokens: u.output, total_tokens: u.input + u.output };
}

function openaiCalls(c: Completion): Obj[] {
  return c.calls.map((call, i) => ({ id: call.id !== '' ? call.id : `call_${i}`, type: 'function', function: { name: call.name, arguments: call.args } }));
}

export function openaiJson(c: Completion, model: string): string {
  const raw = c.raw ?? {};
  const choice = Array.isArray(raw.choices) && isObj(raw.choices[0]) ? raw.choices[0] : {};
  const upstreamMessage = isObj(choice.message) ? choice.message : {};
  const message: Obj = { ...upstreamMessage, role: 'assistant', content: c.text === '' && c.calls.length > 0 ? null : c.text };
  if (c.calls.length > 0) message.tool_calls = openaiCalls(c);
  else delete message.tool_calls;
  return JSON.stringify({
    ...raw,
    id: c.id !== '' ? c.id : 'chatcmpl-aggregated',
    object: 'chat.completion',
    created: c.created,
    model,
    choices: [{ ...choice, index: 0, message, finish_reason: openaiFinish(c) }],
    usage: openaiUsage(c, raw.usage),
  });
}

/** The provider's chunks relayed in order with the dialled model id, one finish chunk, usage, then [DONE]. */
export function openaiSse(c: Completion, model: string): string {
  const id = c.id !== '' ? c.id : 'chatcmpl-aggregated';
  const out: string[] = [];
  const emit = (chunk: Obj): void => {
    out.push(`data: ${JSON.stringify(chunk)}\n\n`);
  };
  const base = { id, object: 'chat.completion.chunk', created: c.created, model };
  let finished = false;
  let usageSent = false;
  for (const chunk of c.chunks) {
    const next: Obj = { ...chunk, ...base };
    const choices = chunk.choices as unknown[];
    const choice = isObj(choices[0]) ? choices[0] : null;
    if (isObj(chunk.usage)) {
      next.usage = openaiUsage(c, chunk.usage);
      usageSent = true;
    } else delete next.usage;
    if (choice === null) {
      next.choices = [];
      if (next.usage === undefined) continue;
      emit(next);
      continue;
    }
    const delta: Obj = isObj(choice.delta) ? { ...choice.delta } : {};
    let finish: string | null = null;
    if (finished) {
      // Nothing may follow the finish chunk but usage: a late chunk keeps its usage alone.
      if (next.usage === undefined) continue;
      next.choices = [];
      emit(next);
      continue;
    }
    if (typeof choice.finish_reason === 'string') {
      finish = openaiFinish(c);
      finished = true;
    }
    next.choices = [{ ...choice, index: 0, delta, finish_reason: finish }];
    emit(next);
  }
  if (!finished) emit({ ...base, choices: [{ index: 0, delta: {}, finish_reason: openaiFinish(c) }] });
  if (!usageSent) emit({ ...base, choices: [], usage: openaiUsage(c, null) });
  out.push('data: [DONE]\n\n');
  return out.join('');
}

function parsedArgs(args: string): Obj {
  if (args.trim() === '') return {};
  try {
    const v: unknown = JSON.parse(args);
    return isObj(v) ? v : {};
  } catch {
    return {};
  }
}

function messageId(c: Completion): string {
  const tail = c.id.replace(/[^A-Za-z0-9_-]/g, '');
  return `msg_${tail !== '' ? tail : 'aggregated'}`;
}

function toolUseId(call: ToolCall, at: number): string {
  return call.id !== '' ? call.id : `toolu_${at}`;
}

export function anthropicJson(c: Completion, model: string): string {
  const content: Obj[] = [];
  if (c.text !== '') content.push({ type: 'text', text: c.text });
  c.calls.forEach((call, i) => content.push({ type: 'tool_use', id: toolUseId(call, i), name: call.name, input: parsedArgs(call.args) }));
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const u = usageOf(c);
  return JSON.stringify({
    id: messageId(c),
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(c),
    stop_sequence: null,
    usage: { input_tokens: u.input, output_tokens: u.output },
  });
}

/** message_start, one content block at a time (each opened, filled and closed in turn), message_delta, message_stop. */
export function anthropicSse(c: Completion, model: string): string {
  const out: string[] = [];
  const emit = (event: string, data: Obj): void => {
    out.push(`event: ${event}\ndata: ${JSON.stringify({ type: event, ...data })}\n\n`);
  };
  const u = usageOf(c);
  emit('message_start', {
    message: { id: messageId(c), type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: u.input, output_tokens: 1 } },
  });
  let index = -1;
  let openKind: 'text' | number | null = null;
  const close = (): void => {
    if (openKind !== null) emit('content_block_stop', { index });
    openKind = null;
  };
  // Text deltas in arrival order form the text blocks; a tool call's fragments are gathered into one block.
  const started = new Set<number>();
  for (const piece of c.pieces) {
    if ('text' in piece) {
      if (openKind !== 'text') {
        close();
        index += 1;
        openKind = 'text';
        emit('content_block_start', { index, content_block: { type: 'text', text: '' } });
      }
      emit('content_block_delta', { index, delta: { type: 'text_delta', text: piece.text } });
      continue;
    }
    if (started.has(piece.call)) continue;
    started.add(piece.call);
    close();
    index += 1;
    openKind = piece.call;
    const call = c.calls[piece.call];
    emit('content_block_start', { index, content_block: { type: 'tool_use', id: toolUseId(call, piece.call), name: call.name, input: {} } });
    if (call.args !== '') emit('content_block_delta', { index, delta: { type: 'input_json_delta', partial_json: call.args } });
  }
  c.calls.forEach((call, i) => {
    if (started.has(i)) return;
    close();
    index += 1;
    openKind = i;
    emit('content_block_start', { index, content_block: { type: 'tool_use', id: toolUseId(call, i), name: call.name, input: {} } });
  });
  if (index === -1) {
    index = 0;
    openKind = 'text';
    emit('content_block_start', { index, content_block: { type: 'text', text: '' } });
  }
  close();
  emit('message_delta', { delta: { stop_reason: anthropicStop(c), stop_sequence: null }, usage: { output_tokens: u.output } });
  emit('message_stop', {});
  return out.join('');
}

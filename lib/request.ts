// The client's request: what it needs of a model, the token it asks back, and the one Chat Completions
// body that goes upstream (an Anthropic Messages request is rewritten into it).

import { isObj, type Needs, type Obj } from './brief.ts';

const DEFAULT_MAX_TOKENS = 1024;

function blocksText(content: unknown, joiner: string): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isObj(block) && typeof block.text === 'string') parts.push(block.text);
  return parts.join(joiner);
}

/** What the request itself asks of a model: JSON mode, tools, images. */
export function needsOf(body: Obj): Needs {
  const format = isObj(body.response_format) ? body.response_format.type : undefined;
  let vision = false;
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!isObj(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) if (isObj(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  return { json: format === 'json_object' || format === 'json_schema', tools: Array.isArray(body.tools) && body.tools.length > 0, vision };
}

/** The tool names the client declared, on either wire. */
export function declaredTools(body: Obj): string[] {
  const names: string[] = [];
  if (!Array.isArray(body.tools)) return names;
  for (const tool of body.tools) {
    if (!isObj(tool)) continue;
    if (isObj(tool.function) && typeof tool.function.name === 'string') names.push(tool.function.name);
    else if (typeof tool.name === 'string') names.push(tool.name);
  }
  return names;
}

/** The single WORD-1234 token of the last user turn, or null when it holds none or several. */
export function tokenOf(body: Obj): string | null {
  if (!Array.isArray(body.messages)) return null;
  for (let i = body.messages.length - 1; i >= 0; i -= 1) {
    const message = body.messages[i];
    if (!isObj(message) || message.role !== 'user') continue;
    const text = blocksText(message.content, '\n');
    const seen = new Set<string>();
    for (const m of text.matchAll(/(?<![A-Z0-9])[A-Z]{3,12}-[0-9]{4}(?![0-9A-Z])/g)) seen.add(m[0]);
    return seen.size === 1 ? [...seen][0] : null;
  }
  return null;
}

/** An OpenAI-wire body as it goes upstream: unchanged but for the model, an output cap and stream usage. */
export function openaiUpstream(body: Obj, model: string): Obj {
  const out: Obj = { ...body, model };
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') out.max_tokens = DEFAULT_MAX_TOKENS;
  if (out.stream === true) out.stream_options = { ...(isObj(out.stream_options) ? out.stream_options : {}), include_usage: true };
  return out;
}

function imagePart(block: Obj): Obj | null {
  const source = isObj(block.source) ? block.source : null;
  if (source === null) return null;
  if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (content === undefined || content === null) return '';
  const text = blocksText(content, '\n');
  return text !== '' ? text : JSON.stringify(content);
}

function userTurn(content: unknown[], out: Obj[]): void {
  const parts: Obj[] = [];
  const flush = (): void => {
    if (parts.length === 0) return;
    const textOnly = parts.every((p) => p.type === 'text');
    out.push({ role: 'user', content: textOnly ? parts.map((p) => p.text as string).join('\n') : parts.splice(0) });
    parts.length = 0;
  };
  for (const block of content) {
    if (!isObj(block)) continue;
    if (block.type === 'tool_result') {
      flush();
      out.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
    } else if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
    else if (block.type === 'image') {
      const part = imagePart(block);
      if (part !== null) parts.push(part);
    }
  }
  flush();
}

function assistantTurn(content: unknown[], out: Obj[]): void {
  const calls: Obj[] = [];
  const texts: string[] = [];
  for (const block of content) {
    if (!isObj(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    else if (block.type === 'tool_use') {
      calls.push({
        id: typeof block.id === 'string' ? block.id : `call_${calls.length}`,
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isObj(block.input) ? block.input : {}) },
      });
    }
  }
  const turn: Obj = { role: 'assistant', content: texts.length > 0 ? texts.join('') : null };
  if (calls.length > 0) turn.tool_calls = calls;
  out.push(turn);
}

/** An Anthropic Messages request as the Chat Completions body the gateway takes. */
export function anthropicUpstream(body: Obj, model: string): Obj {
  const messages: Obj[] = [];
  const system = blocksText(body.system, '\n\n');
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isObj(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    if (typeof raw.content === 'string') messages.push({ role, content: raw.content });
    else if (Array.isArray(raw.content)) {
      if (role === 'assistant') assistantTurn(raw.content, messages);
      else userTurn(raw.content, messages);
    }
  }
  const out: Obj = { model, messages, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : DEFAULT_MAX_TOKENS };
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools: Obj[] = [];
    for (const tool of body.tools) {
      if (!isObj(tool) || typeof tool.name !== 'string') continue;
      const fn: Obj = { name: tool.name, parameters: isObj(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} } };
      if (typeof tool.description === 'string') fn.description = tool.description;
      tools.push({ type: 'function', function: fn });
    }
    if (tools.length > 0) out.tools = tools;
  }
  const choice = isObj(body.tool_choice) ? body.tool_choice : null;
  if (choice !== null && out.tools !== undefined) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

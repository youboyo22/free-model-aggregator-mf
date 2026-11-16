# free-model-aggregator-02 entry

One Node 22 server (`main.ts`, standard library only) that fronts OpenRouter's free models for the
make.fun `free-model-aggregator-02` competition. It answers the OpenAI Chat Completions wire and the
Anthropic Messages wire, in JSON and in server-sent events, and every answer is a real completion
dialled through the trial's gateway.

## Layout

| file | what it holds |
| --- | --- |
| `main.ts` | the HTTP server, model routing and failover, registration and the `/work` poll |
| `lib/brief.ts` | reads the brief's model table and prompt manifest; builds the catalog |
| `lib/request.ts` | reads a client request (needs, token, declared tools); builds the upstream body |
| `lib/completion.ts` | reads an upstream answer whole and writes it back in either wire |

## How a prompt is served

1. The request's needs (JSON mode, tools, images) select the free models able to serve it, strongest
   rating first.
2. The best one is dialled on the gateway's Chat Completions route (an Anthropic request is translated
   first) and its answer is buffered.
3. The answer is accepted only if it carries the prompt's token in its text, names no undeclared tool
   and holds no provider error. A 429/5xx gets one more try on the same model after a short pause;
   otherwise the next model down the list is dialled. A model that fails two prompts in a row is tried
   last from then on.
4. The accepted answer is written back in the client's wire with the dialled id as `model`, its usage
   and stop reason, and, when refusals were recovered from, the `x-makefun-upstream-failures` trail.

Routing is a function of the brief, the requests and the gateway's answers only, so a replay of a
recorded trial leaves the same state.

## Run

```
node main.ts --list     # print the catalog document
node main.ts            # inside a trial: needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, OPENAI_BASE_URL, OPENAI_API_KEY
```

MIT licensed.

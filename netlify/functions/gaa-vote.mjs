import { getStore } from "@netlify/blobs";

const STORE_NAME = "gaa-sideline-verdict";
const STATE_KEY = "state-v1";

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  };
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders() },
  });
}

async function readState(store) {
  const data = await store.get(STATE_KEY, { type: "json" });
  if (!data || typeof data !== "object") {
    return { d: 0, m: 0, submissions: [] };
  }
  const d = Number(data.d) || 0;
  const m = Number(data.m) || 0;
  const submissions = Array.isArray(data.submissions) ? data.submissions : [];
  return { d, m, submissions };
}

export default async (req) => {
  const headers = corsHeaders();

  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers });
  }

  const store = getStore(STORE_NAME);

  try {
    if (req.method === "GET") {
      const state = await readState(store);
      const total = state.d + state.m;
      return json({
        d: state.d,
        m: state.m,
        total,
        comments: state.submissions.filter((s) => s && s.opinion && String(s.opinion).trim()).length,
      });
    }

    if (req.method === "POST") {
      let body;
      try {
        body = await req.json();
      } catch {
        return json({ error: "invalid_json" }, 400);
      }

      const vote = body.vote === "d" || body.vote === "m" ? body.vote : null;
      if (!vote) {
        return json({ error: "vote_required", hint: "vote must be \"d\" or \"m\"" }, 400);
      }

      const name = typeof body.name === "string" ? body.name.trim().slice(0, 120) : "";
      const opinion = typeof body.opinion === "string" ? body.opinion.trim().slice(0, 2000) : "";

      const state = await readState(store);
      state[vote]++;
      state.submissions.push({
        vote,
        name,
        opinion,
        ts: new Date().toISOString(),
      });
      if (state.submissions.length > 2500) {
        state.submissions = state.submissions.slice(-2500);
      }

      await store.setJSON(STATE_KEY, state);
      const total = state.d + state.m;
      return json({
        ok: true,
        d: state.d,
        m: state.m,
        total,
        comments: state.submissions.filter((s) => s && s.opinion && String(s.opinion).trim()).length,
      });
    }
  } catch (e) {
    return json({ error: "storage_error", message: String(e && e.message ? e.message : e) }, 503);
  }

  return new Response("Method Not Allowed", { status: 405, headers });
};

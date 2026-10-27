(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const CHIP = { cannot_corroborate: "Can't corroborate", unclear: "Unclear", may_corroborate: "Suspect said it first", not_adopted: "Not adopted" };
  const ORDER = ["interrogator_first", "planted_adopted", "public", "prior_contact", "suspect_first", "never_adopted", "not_mentioned"];
  const KIND = { asks: "asks", asserts: "says", denies: "denies", assents: "agrees" };
  // Reviewer marks live only in this browser (nothing is sent anywhere).
  const reviewKey = (rep) => `provenance_review_${rep.case.id}`;
  function reviews(rep) { try { return JSON.parse(localStorage.getItem(reviewKey(rep)) || "{}"); } catch (_) { return {}; } }
  function saveReview(rep, id, val) {
    const r = reviews(rep);
    if (val) r[id] = val; else delete r[id];
    try { localStorage.setItem(reviewKey(rep), JSON.stringify(r)); } catch (_) { /* private mode: marks last for this page view only */ }
    memReview[id] = val;
  }
  const memReview = {};
  const reviewOf = (rep, id) => (id in memReview ? memReview[id] : reviews(rep)[id]);
  const TACTIC = { we_already_know: "\"We already know\"", minimization: "Minimizing (\"it's OK\", \"not your fault\")", promises: "Promises or offers of help", demand_truth: "Demands to \"be honest\"" };
  let current = null; // {report, lines(id)->Promise}

  // ------------------------------------------------------------ routing
  async function boot() {
    let cases = [];
    try { cases = await (await fetch("/api/cases")).json(); } catch (e) { /* offline */ }
    const nav = $("#caseNav");
    nav.innerHTML = cases.map((c) => `<a href="#${esc(c.id)}" data-id="${esc(c.id)}">${esc(shortTitle(c))}</a>`).join("") + `<a href="#try" data-id="try">Try a transcript</a>`;
    window.addEventListener("hashchange", () => route(cases));
    route(cases);
    $("#sampleBtn").onclick = fillSample;
    $("#tryForm").onsubmit = runTry;
  }
  function shortTitle(c) { return c.suspect ? `${c.suspect.split(" ").slice(-1)[0]} case` : c.title; }
  function route(cases) {
    const id = location.hash.slice(1) || (cases[0] && cases[0].id) || "try";
    document.querySelectorAll("#caseNav a").forEach((a) => (a.dataset.id === id ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
    if (id === "try") { $("#try").scrollIntoView({ block: "start" }); return; }
    loadCase(id);
  }

  async function loadCase(id) {
    const r = await fetch(`/api/case/${encodeURIComponent(id)}`);
    if (!r.ok) { $("#headline").textContent = "Couldn't load this case."; return; }
    const rep = await r.json();
    current = { report: rep, lines: (uid) => fetch(`/api/case/${encodeURIComponent(id)}/lines?around=${encodeURIComponent(uid)}`).then((x) => x.json()) };
    render(rep);
  }

  // ------------------------------------------------------------ rendering
  function render(rep) {
    $("#caseTitle").textContent = rep.case.title;
    $("#caseSub").textContent = `${rep.case.suspect}: ${rep.sessions.filter((s) => s.recorded).length} interviews with public transcripts, ${rep.utterance_count.toLocaleString()} turns of speech, ${rep.facts.length} key details checked`;
    $("#headline").textContent = rep.headline;
    $("#draftNote").textContent = "Draft for counsel, computed from the record with machine-checked lines: investigator lines are matched by the details' key words in code; the suspect's lines are labeled by Nemotron 3 Ultra and checked against the same key words. Every line a verdict rests on is in the review queue, deciding lines first.";
    renderCounts(rep);
    renderMap(rep);
    renderSessions(rep);
    renderMemo(rep.memo);
    renderHow(rep);
    $("#detail").hidden = true;
    $("#csvBtn").onclick = () => exportCsv(rep);
    renderGuess(rep);
    renderReviewCount(rep);
  }

  function renderGuess(rep) {
    // A few details he stated or agreed to: some investigators said first, some he did. Deterministic picks.
    const inv = rep.facts.filter((f) => f.origin === "interrogator_first").sort((a, b) => b.interrogator_mentions_before_suspect - a.interrogator_mentions_before_suspect);
    const sus = rep.facts.filter((f) => f.origin === "suspect_first");
    const picks = [inv[0], sus[0], inv[1], inv[2], sus[1]].filter(Boolean);
    $("#guessSec").hidden = picks.length < 3;
    if (picks.length < 3) return;
    const who = rep.case.suspect.split(" ")[0];
    let answered = 0, right = 0;
    $("#guessScore").textContent = "";
    $("#guessList").innerHTML = picks.map((f, i) => `<li data-i="${i}"><span class="gq">${esc(f.fact.label)}</span>
      <span class="gbtns"><button type="button" class="ghost" data-a="interrogator">Investigators</button><button type="button" class="ghost" data-a="suspect">${esc(who)}</button></span>
      <span class="gans" aria-live="polite"></span></li>`).join("");
    $("#guessList").querySelectorAll("li").forEach((li) => {
      const f = picks[+li.dataset.i];
      li.querySelectorAll("button").forEach((b) => b.onclick = () => {
        const truth = f.origin === "interrogator_first" ? "interrogator" : "suspect";
        const ok = b.dataset.a === truth;
        answered++; right += ok;
        li.querySelectorAll("button").forEach((x) => { x.disabled = true; x.setAttribute("aria-pressed", String(x === b)); });
        b.classList.add("picked");
        const fi = f.first_interrogator, fs = f.first_suspect;
        li.querySelector(".gans").innerHTML = `<b class="${ok ? "gok" : "gno"}">${ok ? "Right" : "Not quite"}.</b> ` + (truth === "interrogator"
          ? `Investigators raised it ${f.interrogator_mentions_before_suspect} time${f.interrogator_mentions_before_suspect === 1 ? "" : "s"} first (${esc(sessionLabel(rep, fi.session).split(", ")[0])}, p. ${esc(fi.page)}); he ${fs.kind === "assents" ? "agreed to it" : "said it"} later.`
          : `He said it first (${esc(sessionLabel(rep, fs.session).split(", ")[0])}, p. ${esc(fs.page)}).`);
        if (answered === picks.length) $("#guessScore").textContent = `You got ${right} of ${picks.length}. Reading the confession alone, it's hard to tell. The map below shows every detail.`;
      });
    });
  }

  function renderReviewCount(rep) {
    const r = reviews(rep), ids = rep.facts.map((f) => f.fact.id);
    const done = ids.filter((id) => reviewOf(rep, id)).length, wrong = ids.filter((id) => (reviewOf(rep, id) || {}).status === "wrong").length;
    $("#reviewCount").textContent = done ? `You've checked ${done} of ${ids.length} details${wrong ? `; ${wrong} marked for re-check` : ""}. Saved in this browser only.` : "";
    void r;
  }

  function exportCsv(rep) {
    const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const at = (ev) => (ev ? `${sessionLabel(rep, ev.session)}, p. ${ev.page || "?"}` : "");
    const head = ["Detail", "Origin", "Can it corroborate?", "First investigator mention", "Investigator words", "First suspect statement or agreement",
      "Suspect words", "How (states/agrees)", "Investigator mentions before the suspect's", "Source for the detail", "Your review"];
    const rows = rep.facts.map((f) => [f.fact.label, f.origin_label, f.corroboration.replace(/_/g, " "), at(f.first_interrogator),
      (f.first_interrogator || {}).quote, at(f.first_suspect), (f.first_suspect || {}).quote, (f.first_suspect || {}).kind,
      f.interrogator_mentions_before_suspect, f.fact.source,
      (() => { const v = reviewOf(rep, f.fact.id); return v ? `${v.status === "ok" ? "checked" : "re-check"} ${v.at}` : ""; })()]);
    const csv = [head, ...rows].map((r) => r.map(q).join(",")).join("\r\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" }));
    a.download = `provenance_${rep.case.id}_citations.csv`;
    document.body.appendChild(a); a.click(); a.remove();
  }

  function renderCounts(rep) {
    const c = rep.counts;
    const parts = [["c-inv", c.interrogator_first + c.planted_adopted, "investigators first"], ["c-pub", c.public, "already public"],
      ["c-unc", c.prior_contact, "unclear"], ["c-sus", c.suspect_first, "suspect first"], ["c-never", c.never_adopted + c.not_mentioned, "not adopted"]];
    const total = parts.reduce((a, p) => a + p[1], 0) || 1;
    $("#countbar").innerHTML = parts.filter((p) => p[1]).map(([cls, n, lab]) =>
      `<span class="${cls}" style="flex:${n}" title="${n} ${lab}">${n} ${lab}</span>`).join("");
    $("#countbar").setAttribute("aria-label", parts.filter((p) => p[1]).map(([, n, lab]) => `${n} ${lab}`).join(", ") + ` of ${total} details`);
  }

  function layout(rep) {
    // Session bands: width proportional to turns; unrecorded sessions get a thin hatched band.
    const st = rep.session_stats;
    const weights = st.map((s) => (s.recorded && s.turns ? s.turns : Math.max(1, rep.utterance_count * 0.05)));
    const total = weights.reduce((a, b) => a + b, 0);
    let x = 0;
    const bands = st.map((s, i) => { const w = (100 * weights[i]) / total; const b = { ...s, x, w }; x += w; return b; });
    const pos = (ev) => {
      const b = bands.find((bb) => bb.session === ev.session);
      if (!b || b.first_seq == null) return 0;
      const span = Math.max(1, b.last_seq - b.first_seq);
      return b.x + b.w * (0.03 + 0.94 * ((ev.seq - b.first_seq) / span));
    };
    return { bands, pos };
  }

  function renderMap(rep) {
    const { bands, pos } = layout(rep);
    const facts = [...rep.facts].sort((a, b) => ORDER.indexOf(a.origin) - ORDER.indexOf(b.origin) ||
      ((a.first_interrogator || a.first_suspect || {}).seq || 1e12) - ((b.first_interrogator || b.first_suspect || {}).seq || 1e12));
    const axis = `<div class="axis" role="row"><div role="columnheader" style="padding:8px 12px">Detail</div><div class="bands" role="columnheader">${bands.map((b) =>
      `<div class="band ${b.recorded ? "" : "unrec"}" style="width:${b.w}%" title="${esc(b.label)}">${esc(b.label)}</div>`).join("")}</div></div>`;
    const rows = facts.map((f, i) => {
      const fi = f.first_interrogator, fs = f.first_suspect;
      const dots = f.events.map((ev) => {
        const cls = ev.role === "interrogator" ? "inv" : ev.kind === "assents" ? "agr" : ev.kind === "denies" ? "den" : ev.role === "suspect" ? "sus" : "den";
        const first = ev.first ? " first" : "";
        return `<span class="dot ${cls}${first}" style="left:${pos(ev)}%" title="${esc(ev.speaker)} ${KIND[ev.kind] || ev.kind}: ${esc(ev.sensitive ? "[graphic passage hidden]" : ev.quote)} (p. ${esc(ev.page)})"></span>`;
      }).join("");
      const gap = fi && fs && fi.seq < fs.seq ? `<span class="gapline" style="left:${pos(fi)}%;width:${Math.max(0, pos(fs) - pos(fi))}%"></span>` : "";
      const sb = bands.map((b) => `<span class="sb ${b.recorded ? "" : "unrec"}" style="left:${b.x}%;width:${b.w}%"></span>`).join("");
      const label = `${f.fact.label}: ${CHIP[f.corroboration]}. ${f.origin_label}.`;
      const rv = reviewOf(rep, f.fact.id);
      return `<div class="frow" role="row" tabindex="0" data-i="${i}" data-fid="${esc(f.fact.id)}" aria-label="${esc(label)}">
        <div class="flabel" role="cell"><b>${esc(f.fact.label)}</b><span class="chip ${f.corroboration}">${CHIP[f.corroboration]}</span><span class="rvmark">${rv ? (rv.status === "ok" ? "checked" : "re-check") : ""}</span></div>
        <div class="track" role="cell">${sb}${gap}${dots}</div></div>`;
    }).join("");
    const map = $("#map");
    map.innerHTML = axis + rows;
    map.querySelectorAll(".frow").forEach((row) => {
      const open = () => { map.querySelectorAll(".frow").forEach((r) => r.removeAttribute("aria-selected")); row.setAttribute("aria-selected", "true"); showFact(facts[+row.dataset.i], rep); };
      row.onclick = open;
      row.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
        if (e.key === "ArrowDown") row.nextElementSibling?.focus(); if (e.key === "ArrowUp") row.previousElementSibling?.classList.contains("frow") && row.previousElementSibling.focus(); };
    });
  }

  function sessionLabel(rep, id) { return (rep.sessions.find((s) => s.id === id) || {}).label || id; }

  function showFact(f, rep) {
    const d = $("#detail");
    const stats = [];
    if (f.interrogator_mentions_before_suspect) stats.push(`Investigators raised it <b>${f.interrogator_mentions_before_suspect}</b> time${f.interrogator_mentions_before_suspect > 1 ? "s" : ""} before he said it`);
    if (f.adopted_by_assent) stats.push("He first adopted it by <b>agreeing</b> to a question");
    if (f.crossed_sessions) stats.push("Supplied in an <b>earlier interview</b>");
    if (f.fact.public_before) stats.push(`Public since <b>${esc(f.fact.public_before)}</b>`);
    d.innerHTML = `<h3>${esc(f.fact.label)} <span class="chip ${f.corroboration}">${CHIP[f.corroboration]}</span></h3>
      <p class="why">${esc(f.why)}</p>
      ${stats.length ? `<div class="stats">${stats.map((s) => `<span>${s}</span>`).join("")}</div>` : ""}
      ${f.fact.detail ? `<p class="fine">What counts as stating it: ${esc(f.fact.detail.replace(/\.$/, ""))}.${f.fact.source ? ` Source: ${esc(f.fact.source)}` : ""}</p>` : ""}
      ${(f.investigator_lines_before || []).length ? `<p class="fine"><b>Investigator lines before his first statement</b> (earliest ${f.investigator_lines_before.length}): ${f.investigator_lines_before.map((x) => `p. ${esc(x.page)} “${esc(x.quote.slice(0, 110))}”${x.by === "key words" ? " (matched by key words)" : ""}`).join("; ")}</p>` : ""}
      ${(f.related_statements || []).length ? `<p class="fine"><b>His related earlier statements</b> (part of the detail${f.related_statements_total > f.related_statements.length ? `; ${f.related_statements_total} in all` : ""}): ${f.related_statements.map((x) => `p. ${esc(x.page)} “${esc(x.quote.slice(0, 110))}”`).join("; ")}</p>` : ""}
      <h4 class="evhead">The lines that decide it</h4>
      <ol class="evlist">${f.events.map((ev, i) => [ev, i]).filter(([ev]) => ev.first).map(([ev, i]) => evItem(ev, i, rep)).join("") || "<li>No one said it in the recorded interviews.</li>"}</ol>
      <div class="review" role="group" aria-label="Your review of this detail">
        <span>Checked these lines against the transcript?</span>
        <button type="button" class="ghost" data-rv="ok">Looks right</button>
        <button type="button" class="ghost" data-rv="wrong">Something's wrong</button>
        <span class="rvstate" aria-live="polite"></span>
      </div>
      ${f.events.length > f.events.filter((ev) => ev.first).length ? `<details class="allev"><summary>Every mention in order (${f.events.length})</summary>
        <ol class="evlist">${f.events.map((ev, i) => evItem(ev, i, rep)).join("")}</ol></details>` : ""}`;
    const paint = () => {
      const v = reviewOf(rep, f.fact.id);
      d.querySelectorAll("[data-rv]").forEach((b) => b.setAttribute("aria-pressed", String(v && v.status === b.dataset.rv)));
      d.querySelector(".rvstate").textContent = v ? (v.status === "ok" ? "Marked as checked." : "Marked for re-check; it's flagged in the map and the citation table.") : "";
      renderReviewCount(rep);
      const row = document.querySelector(`.frow[data-fid="${f.fact.id}"] .rvmark`);
      if (row) row.textContent = v ? (v.status === "ok" ? "checked" : "re-check") : "";
    };
    d.querySelectorAll("[data-rv]").forEach((b) => (b.onclick = () => {
      const cur = reviewOf(rep, f.fact.id);
      saveReview(rep, f.fact.id, cur && cur.status === b.dataset.rv ? null : { status: b.dataset.rv, at: new Date().toISOString().slice(0, 10) });
      paint();
    }));
    paint();
    d.hidden = false;
    d.querySelectorAll("[data-ctx]").forEach((b) => (b.onclick = () => showContext(b)));
    d.querySelectorAll("[data-reveal]").forEach((b) => (b.onclick = () => { const q = b.previousElementSibling; q.textContent = q.dataset.full; q.classList.remove("masked"); b.remove(); }));
    d.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
  }

  function evItem(ev, i, rep) {
    const cls = ev.role === "interrogator" ? "inv" : ev.role === "suspect" ? "sus" : "";
    const mark = ev.role === "interrogator" ? "inv" : ev.kind === "assents" ? "agr" : ev.kind === "denies" ? "den" : "sus";
    const first = ev.first ? `<span class="tag">first ${ev.first === "interrogator" ? "investigator mention" : "time he said it"}</span>` : "";
    const checked = ev.by === "key words" ? `<span class="vchk">matched by key words${ev.partial ? " (part of the detail)" : ""}</span>`
      : ev.first ? (ev.verified === true ? `<span class="vchk">re-checked by Nemotron 3 Ultra</span>` : ev.verified === null || ev.verified === undefined ? `<span class="vchk no">not re-checked</span>` : "") : "";
    const note = ev.kind_note ? `<span class="vchk no">${esc(ev.kind_note)}</span>` : "";
    const quote = ev.sensitive ? `<q class="masked" data-full="${esc(ev.quote)}">Graphic passage hidden</q> <button data-reveal>Show</button>` : `<q>${esc(ev.quote)}</q>`;
    return `<li class="ev ${cls}"><i class="mk ${mark}" aria-hidden="true" style="margin-top:6px"></i>
      <div><span class="who">${esc(ev.speaker)}</span> ${KIND[ev.kind] || esc(ev.kind)}${first}${checked}${note}<br>${quote}
        <div><button data-ctx="${esc(ev.utterance)}">Read in context</button></div><div class="ctxbox"></div></div>
      <span class="cite">${esc(sessionLabel(rep, ev.session).split(", ").slice(0, 2).join(", "))}, ${/^PDF/.test(ev.page || "") ? esc(ev.page) : `p. ${esc(ev.page || "?")}`}</span></li>`;
  }

  async function showContext(btn) {
    const box = btn.parentElement.nextElementSibling;
    if (box.innerHTML) { box.innerHTML = ""; return; }
    box.innerHTML = "<p class='fine'>Loading…</p>";
    try {
      const data = await current.lines(btn.dataset.ctx);
      box.innerHTML = `<div class="ctx">${data.lines.map((l) => `<p class="${l.id === data.focus ? "focus" : ""}"><span class="spk">${esc(l.speaker)}</span>${l.sensitive ? "<span class='masked'>Graphic passage hidden</span>" : esc(l.text)}</p>`).join("")}</div>`;
    } catch (e) { box.innerHTML = "<p class='err'>Couldn't load the lines.</p>"; }
  }

  function renderSessions(rep) {
    $("#sessionCards").innerHTML = rep.session_stats.map((s) => s.recorded && s.turns ? `<div class="card"><h3>${esc(s.label)}</h3><dl>
      <dt>Turns of speech</dt><dd>${s.turns}</dd>
      <dt>Investigators' share of words</dt><dd>${Math.round(100 * s.interrogator_share_of_words)}%</dd>
      <dt>Suspect's words per answer</dt><dd>${s.suspect_words_per_turn}</dd>
      <dt>One-word answers ("yeah", "no")</dt><dd>${s.suspect_minimal_answers}</dd>
      <dt>Questions that carried a detail</dt><dd>${s.questions_that_carry_facts} of ${s.questions}</dd>
      ${Object.entries(s.tactics).map(([k, v]) => `<dt>${TACTIC[k] || k}</dt><dd>${v}</dd>`).join("")}</dl></div>`
      : `<div class="card"><h3>${esc(s.label)}</h3><p class="unrec">No public transcript. Details discussed here can't be traced, so they're marked unclear when referred to later.</p></div>`).join("");
  }

  function renderMemo(m) {
    $("#memoSec").hidden = !m;
    if (!m) return;
    $("#memo").innerHTML = `<p>${esc(m.summary)}</p><ul>${m.key_points.map((k) => `<li>${esc(k)}</li>`).join("")}</ul>
      <p class="fine"><b>Limits:</b> ${m.limits.map(esc).join(" ")}</p><p class="fine">${m.by === "ultra" || !m.by ? "Written by Nemotron 3 Ultra from the findings above only." : esc(m.by)}</p>`;
  }

  function renderHow(rep) {
    const run = rep.run || {};
    const by = run.by_model || {};
    const cost = (role) => (by[role] ? ` (${by[role].calls} calls, $${by[role].cost_usd.toFixed(3)})` : "");
    const lab = run.labeler || (by.lightning ? "lightning" : by.matcher ? "matcher" : "ultra");
    const labeler = lab === "matcher" ? "our fine-tuned matcher (NVIDIA llama-nemotron-rerank-1b-v2 on Nebius)" : lab === "lightning" ? "Nemotron 3.5 Lightning" : "Nemotron 3 Ultra";
    const st = run.by_step || {};
    const stepCost = (k, fallback) => (st[k] ? ` (${st[k].calls} calls, $${st[k].cost_usd.toFixed(3)})` : fallback);
    const steps = [];
    if (by.super_vl || run.read_by === "super_vl") steps.push(`<li><b>Read the scanned transcripts.</b> <span class="m">Nemotron 3.5 Super VL</span> turned each page image into speaker-labeled lines${cost("super_vl")}. Code merged continuation lines, matched speaker names to the case roster and split out the suspect's nods.</li>`);
    steps.push(`<li><b>List the key details.</b> Each detail, its source and any date it was already public were checked by hand against the court record and news reports.</li>`);
    if (run.by_model || run.by_step) steps.push(`<li><b>Label every sentence.</b> <span class="m">${labeler}</span> marked which details each line carries and how (asks, says, denies, agrees)${stepCost("label", lab === "lightning" ? cost("lightning") : lab === "matcher" ? cost("matcher") : "")}. Code throws out any label whose quoted words aren't in the line${run.code_agreements_added ? `, and added ${run.code_agreements_added} bare “yeah”s that directly answer a question carrying the detail` : ""}. A “yeah” only counts right after a question about that detail.</li>`);
    steps.push(`<li><b>Decide who said it first.</b> Plain code walks the record in order. No model decides this.</li>`);
    if (by.ultra) steps.push(`<li><b>Double-check what decides a verdict.</b> <span class="m">Nemotron 3 Ultra</span>${run.verifier_reasoning ? " (with reasoning)" : ""} re-read every first-mention line and kept a label only if the line conveys the detail's distinctive part, not just the topic; rejected ones were dropped and the order re-checked${stepCost("check", lab === "ultra" ? "" : cost("ultra"))}.${run.unverified ? ` ${run.unverified} deciding line${run.unverified === 1 ? " was" : "s were"} not re-checked and ${run.unverified === 1 ? "is" : "are"} marked.` : ""}</li>`);
    if (run.safety_search_added !== undefined) steps.push(`<li><b>Safety search before crediting the suspect.</b> For every detail he stated or agreed to, code searched all earlier investigator lines for the detail's words (and the questions right before his answer), and Ultra checked each candidate. It found ${run.safety_search_added} earlier investigator line${run.safety_search_added === 1 ? "" : "s"} the labels had missed.</li>`);
    if (!run.by_model) steps.push(`<li>This case's processing log isn't available, so the steps above describe the method, not this run.</li>`);
    if (run.total_seconds) steps.push(`<li>Whole record: <b>${Math.round(run.total_seconds)} s</b>, <b>$${(run.cost_usd || 0).toFixed(2)}</b> in model costs.</li>`);
    $("#steps").innerHTML = steps.join("");
    const ev = rep.eval;
    $("#evalBox").innerHTML = ev && ev.rows ? `<h3 style="margin-top:18px">Measured accuracy</h3><p class="fine">${esc(ev.note || "")}</p>
      <table class="eval"><thead><tr><th>Method</th><th>Right first-mention speaker</th><th>Right line</th></tr></thead><tbody>
      ${ev.rows.map((r) => `<tr><td>${esc(r.method)}</td><td class="n">${Math.round(100 * r.speaker_acc)}%</td><td class="n">${Math.round(100 * r.line_acc)}%</td></tr>`).join("")}</tbody></table>
      ${ev.rows.some((r) => /one prompt/i.test(r.method)) ? `<p class="fine">On short synthetic interviews (30 to 50 lines), one Ultra prompt that reads the whole interview is a strong baseline. The real record is ${(rep.utterance_count || 0).toLocaleString()} lines across ${(rep.sessions || []).filter((x) => x.recorded).length} recorded sessions, and Provenance cites, and has Ultra re-check, every line a verdict rests on.</p>` : ""}
      ${rep.one_prompt ? `<p class="fine">On this record, one Ultra prompt per session (code combining the sessions in order) gave the same first speaker as this audit for ${rep.one_prompt.agree} of ${rep.one_prompt.facts} details ($${rep.one_prompt.cost_usd.toFixed(2)}). Where they differ, the lines are in <code>one_prompt.json</code>.</p>` : ""}` : "";
  }

  // ------------------------------------------------------------ try your own
  function fillSample() {
    const f = $("#tryForm");
    f.transcript.value = ["DET: Tell me what happened Saturday night.", "JORDAN: I was home playing games.", "DET: We know you were at the park. Was it near the swings?",
      "JORDAN: I guess.", "DET: He had a red jacket on, didn't he?", "JORDAN: Yeah.", "DET: And the bottle, you threw it in the creek?", "JORDAN: Yeah, the creek.",
      "DET: What did you do with the phone?", "JORDAN: I left it at my cousin's house on Elm."].join("\n");
    f.facts.value = ["At the park near the swings", "Victim wore a red jacket", "Bottle thrown in the creek", "Phone left at a house on Elm"].join("\n");
    f.suspect.value = "JORDAN"; f.interrogators.value = "DET";
  }

  async function runTry(e) {
    e.preventDefault();
    const f = e.target, btn = $("#tryBtn"), out = $("#tryOut");
    btn.disabled = true; btn.textContent = "Running on Nemotron…"; out.innerHTML = "";
    try {
      const r = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: f.transcript.value, facts: f.facts.value, suspect: f.suspect.value, interrogators: f.interrogators.value }) });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Something went wrong.");
      const utts = data.utterances;
      current = { report: data, lines: async (uid) => { const i = utts.findIndex((u) => u.id === uid); return { lines: utts.slice(Math.max(0, i - 4), i + 3), focus: uid }; } };
      render(Object.assign(data, { memo: null, eval: null }));
      out.innerHTML = `<p class="fine">Done in ${data.run.total_seconds} s for $${data.run.cost_usd.toFixed(3)}. Results are shown above.</p>`;
      $("#hero").scrollIntoView({ block: "start" });
    } catch (err) { out.innerHTML = `<p class="err">${esc(err.message)}</p>`; }
    finally { btn.disabled = false; btn.textContent = "Run the audit"; }
  }

  boot();
})();

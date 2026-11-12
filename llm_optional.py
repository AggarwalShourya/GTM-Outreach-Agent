"""LLM judgment + drafting, with a deterministic default.

Offline (no key / PS3_LLM=0): `heuristic_judge` + the template draft keep the
run fast and reproducible. When a Google key is present, a Gemini model
(`PS3_LLM_MODEL`, default gemini-3.8-flash — the current stable one; 2.5-flash
is retired for new users) powers the DRAFT (and optionally the per-signal
JUDGE). Every LLM call degrades gracefully: on 503/429 (high load), a bad
parse, or no network, we fall back to the heuristic/template so the run never
dies.

Toggles (env):
  PS3_LLM=0          force fully deterministic (ignore the key)
  PS3_LLM_DRAFT=1    (default when a key is set) LLM writes the draft
  PS3_LLM_JUDGE=0    (default) heuristic per-signal scores; 1 = LLM scores axes
"""

import json
import os
import time

from .scoring import heuristic_axes, AXES
from . import corpus
from . import net


def heuristic_judge(signal, prospect):
    """The offline stand-in — same contract as an LLM judge."""
    return heuristic_axes(signal, prospect)


def _llm_draft_on():
    return os.environ.get("PS3_LLM_DRAFT", "1").strip().lower() not in ("0", "false", "no", "off")


def _llm_judge_on():
    return os.environ.get("PS3_LLM_JUDGE", "0").strip().lower() in ("1", "true", "yes", "on")


class LLM:
    """Real-model judge/drafter. Pass an instance to pipeline.run(llm=...).
    With no configured key, or PS3_LLM=0, `available()` is False and callers
    just use the heuristic / template — deterministic and offline-safe."""

    def __init__(self):
        self._key = os.environ.get("GOOGLE_API_KEY")
        # Cheap/fast by default: the draft + judge tasks are small, so a flash-lite model is plenty (and much faster than pro/flash).
        self._model = os.environ.get("PS3_LLM_MODEL", "gemini-flash-lite-latest")
        self._retries = int(os.environ.get("PS3_LLM_RETRIES", "2"))
        self.enabled = (bool(self._key)
                        and os.environ.get("PS3_LLM", "1").strip().lower() not in ("0", "false", "no", "off"))
        self.draft_on = self.enabled and _llm_draft_on()
        self.judge_on = self.enabled and _llm_judge_on()

    # ---- public capability flags used by pipeline/scoring ------------------
    def available(self):
        return self.enabled

    def judge(self, signal, prospect):
        """Per-signal axis scores via the LLM. Returns {axis:{score,reason}}
        or None (caller falls back to heuristic_judge)."""
        if not self.judge_on:
            return None
        sig = signal.get("evidence") or signal.get("summary") or signal.get("headline") or ""
        prompt = (
            "You are scoring how good a sales-outreach HOOK this public signal is, "
            "for a company that sells B2B sales-teams a tool that cuts prospect "
            f"research from ~20 to ~2 minutes.\n\n"
            f"Signal: {signal.get('headline','')}\nSource: {signal.get('source','')} "
            f"({signal.get('published','')}), {signal.get('url','')}\n"
            f"Evidence: {sig[:500]}\n\n"
            "Score each axis 1-5 and give a one-line reason for each. "
            "Return ONLY a JSON object: "
            '{"relevance":{"score":N,"reason":"..."},'
            '"recency":{"score":N,"reason":"..."},'
            '"specificity":{"score":N,"reason":"..."},'
            '"attributability":{"score":N,"reason":"..."},'
            '"actionability":{"score":N,"reason":"..."}}'
        )
        out = self._generate(prompt, as_json=True)
        if not out:
            return None
        clean = {}
        for k, _, _ in AXES:
            cell = out.get(k) or {}
            if isinstance(cell, dict) and "score" in cell:
                clean[k] = {"score": int(str(cell.get("score", 3))[:1] or 3) if str(cell.get("score","3"))[:1].isdigit() else 3,
                            "reason": str(cell.get("reason", ""))[:120]}
            else:
                clean[k] = {"score": 3, "reason": "llm did not return this axis"}
        return clean

    BANNED_PHRASES = [
        "hope this finds you well", "in your role as", "leading initiatives",
        "we'd love to", "quick one", "leverage", "valuable", "innovative",
        "at scale", "synergy", "align with", "touch base", "hope this resonates",
        "would it make sense to connect", "reaching out because", "I came across",
        "puts massive pressure on sales teams", "without losing relevance",
        "that kind of growth", "contributes so much to", "in your role",
        # awkward-beat guard: these mean the draft opened on news that is not
        # safe to cold-email the person about -> triggers the re-roll.
        "resignation", "your departure", "stepping down", "leadership changes",
        "layoff", "layoffs",
    ]

    def draft(self, signal, scored, mode, first, last, role, company, caveats,
              profile=None, force=False):
        """LLM-written outreach draft — the ONLY drafting engine (the old
        deterministic template was removed: it read generic). Grounded ONLY
        in the retrieved signal + the person's public profile; returns
        {"subject","body","grounded_on"} or None when the model is
        unavailable/failed — the pipeline then shows "draft unavailable"
        instead of shipping anything canned."""
        if not self.draft_on:
            return None
        ev = (signal or {}).get("evidence") or (signal or {}).get("summary") or \
             (signal or {}).get("headline") or ""
        profile_block, grounded = "", []
        if profile and profile.get("available"):
            lines = []
            li = profile.get("linkedin")
            if li:
                lines.append(f"LinkedIn (search-indexed snippet): {li.get('title','')} — {li.get('headline','')[:300]} [url: {li.get('url','')}]")
                grounded.append("public LinkedIn profile (search snippet)")
            for f in (profile.get("facts") or [])[:5]:
                if f.get("text"):
                    lines.append(f"Public mention: {f['text']}"
                                 + (f" — {f['detail'][:300]}" if f.get("detail") else "")
                                 + (f" [url: {f['url']}]" if f.get("url") else ""))
                    grounded.append("public mention: " + f["text"][:60])
            if lines:
                profile_block = ("PERSON PROFILE (public sources; these are facts "
                                  "you MAY reference):\n" + "\n".join(lines) + "\n")

        trigger_block = (
            "MODE: trigger. Open with the specific hook in your first two "
            "sentences (their news, their move — cite it precisely, one "
            "verifiable detail). Then ONE sentence connecting that hook to "
            "the value prop (why this specific moment makes it relevant to "
            "them, not to 'teams like yours'). Then the ask."
            if mode == "trigger" else
            "MODE: relationship-builder. No strong trigger was found, so the "
            "posture is soft - but the chosen hook is STILL the first thing "
            "you mention, phrased lightly (a 'saw you say…' beat, no hype). "
            "If the hook is genuinely unrelated to what they lead, prefer a "
            "warm personal opener over forcing it."
        )
        sysp = (
            "You write cold B2B outreach emails that get replies because they "
            "are SPECIFIC, not polished. Sound like one person wrote it in 30 "
            "seconds - not a marketing department.\n\n"
            "HARD RULES:\n"
            "1. Use ONLY facts present in the input (signal + person profile). "
            "Never invent a company, role, event, number, or claim.\n"
            "2. Sentence one: a SHORT sentence containing one specific, "
            "verifiable detail about them - the sharpest fact in the input. "
            "That is what makes the email non-generic.\n"
            "3. 3-5 short sentences total, 50-90 words. Vary sentence length; "
            "no run-ons.\n"
            "4. Exactly one low-effort ask at the end: a yes/no question or a "
            "15-minute call - concrete, not a hedged 'connect'.\n"
            "5. Subject line: 5-7 words, specific to them, no clickbait.\n\n"
            "BANNED (any use = rewrite): " + ", ".join(f'"{b}"' for b in self.BANNED_PHRASES) + ".\n\n"
            + trigger_block
            + "\nHard content rule: material about an executive DEPARTING, "
            "resigning, a layoff, or a legal/health story is NOT usable even "
            "if it appears in the evidence - build the opener from a different "
            "candidate or the personal anchor instead.\n\n"
            "Respect the caveats: they limit what you may claim, they do not "
            "stop you writing."
        )
        # Personal opening line: the sharpest fact ABOUT WHO THEY ARE, ranked
        # by relevance (a current role/company beats a follower count), not
        # by raw text length. Used only as the personal anchor.
        ROLE_WORDS = ("ceo", "founder", "president", "vp", "vice president",
                      "chief", "head of", "director", "lead", "principal",
                      "engineer", "student", "professor", "partner", "general counsel")
        GARBAGE = ("follower", "likes", "shared", "posted on", "reposted",
                   "wants to be an", "view profile", "linkedin member")

        def _rank_fact(text):
            t = text.lower()
            score = 0
            if any(w in t for w in ROLE_WORDS):
                score += 2
            blob = (first + " " + last).lower()
            if (first or "").lower() in t or (last or "").lower() in t:
                score += 1
            if company and company.lower() in t:
                score += 1
            if any(w in t for w in GARBAGE) and score < 2:
                score = 0
            return score

        who_they_are = ""
        if profile:
            li = profile.get("linkedin")
            if li:
                t = (li.get("title") or "").strip()
                h = (li.get("headline") or "").strip()[:160]
                who_they_are = (t + " | " + h).strip(" |") if t else h
            for f_ in (profile.get("facts") or [])[:6]:
                tt = (f_.get("text") or "").strip()
                if _rank_fact(tt) > _rank_fact(who_they_are) or (not who_they_are and tt):
                    who_they_are = tt[:200]
            # If the best fact is still follower-count noise, prefer the LinkedIn title alone.
            if who_they_are and any(w in who_they_are.lower() for w in GARBAGE) and not any(w in who_they_are.lower() for w in ROLE_WORDS):
                li = profile.get("linkedin")
                if li and (li.get("title") or "").strip():
                    who_they_are = (li.get("title") or "").strip()

        # Candidate icebreakers: hand the LLM up to 5 ranked options (from the
        # already-scored signals) instead of one pre-picked hook, and let it
        # pick the professional, hard-to-ignore opener itself. Awkward beats
        # (executive departure, layoffs, lawsuits, scandals) are excluded from
        # candidacy - you don't open a cold email TO the departing CEO with
        # their own departure.
        AWKWARD = ("resign", "resignation", "stepped down", "stepping down",
                   "ousted", "departure from", "departed from", "left the company",
                   "exited the company", "layoff", "layoffs", "laid off",
                   "workforce reduction", "terminated as", "fired", "sued",
                   "lawsuit", "indicted", "convicted", "imprisoned", "scandal",
                   "downfall")

        def _awkward(s):
            t = ((s.get("headline") or "") + " " +
                 (s.get("evidence") or s.get("summary") or "")).lower()
            return any(w in t for w in AWKWARD)

        cand = []
        for x in (scored or [])[:8]:
            s = x.get("signal") or {}
            head = (s.get("headline") or "").strip()
            if not head or _awkward(s):
                continue
            ev_short = " ".join((s.get("evidence") or s.get("summary") or "").split())[:140]
            cand.append({"headline": head[:110], "ev": ev_short})
            if len(cand) >= 5:
                break

        cand_block = ""
        if cand:
            cand_block = ("ICEBREAKER CANDIDATES (ranked - pick the single best "
                          "one for THIS person; number them in order):\n"
                         + "\n".join(f"{i+1}) {c['headline']}"
                                      + (f"  [context: {c['ev']}]" if c['ev'] else "")
                                     for i, c in enumerate(cand))
                         + "\n")

        if mode == "trigger":
            if cand:
                rule = (
                    "RULE: PICK ONE candidate from the list - the one that is "
                    "(a) professional and polite to say to the person "
                    "themselves, (b) specific rather than a general industry "
                    "claim, and (c) hard for a busy exec to ignore - it must "
                    "start a real conversation, not something cringey to hear "
                    "in cold email. Then: sentence 1 references that picked "
                    "candidate as an icebreaker in ONE short phrase in your "
                    "own words (3-8 words) - never a restatement of its "
                    "headline. Sentence 2: one specific, concrete implication "
                    "for their GTM org (the operational detail a CEO actually "
                    "worries about - not 'teams get pressure'). Sentence 3: "
                    "one line on what we do, tied to sentence 2. Sentence 4: "
                    "one light yes/no ask.\n"
                )
            else:
                rule = (
                    "RULE: No usable candidate. Open with something personal "
                    "to THEM from 'Who they are' (never a generic role "
                    "generalization like 'as CEO of X'). One sentence on how "
                    "what we do maps to their world. Final sentence: one "
                    "light, low-commitment ask.\n"
                )
        else:
            if cand:
                rule = (
                    "RULE: Soft posture. PICK ONE candidate that is "
                    "comfortably shareable: one light 'saw you say…' sentence, "
                    "no hype, no congratulations, no awkward beats. If none "
                    "is shareable professionally, open on 'Who they are' "
                    "instead and keep the value-prop sentence general. Final "
                    "sentence: one light, low-commitment ask.\n"
                )
            else:
                rule = (
                    "RULE: There is no signal at all. Open with something "
                    "personal to THEM from 'Who they are' (their own "
                    "words/stance - never a generic role generalization like "
                    "'as CEO of X'). One sentence on how what we do maps to "
                    "their world. Final sentence: one light, low-commitment "
                    "ask.\n"
                )

        user = (
            f"Prospect: {first} {last} — {role or 'role unknown'} at {company}\n"
            f"What we do (one line, to paraphrase - never quote verbatim): {corpus.VALUE_PROP}\n"
            + cand_block
            + (f"THE CHOSEN HOOK to reference: {signal.get('headline','')}\n"
               f"Its evidence (only facts from here may be cited): {ev[:700]}\n"
               if (not cand and signal and signal.get("headline")) else "")
            + profile_block
            + f"Who they are (personal anchor): {who_they_are or 'thin - keep the opener warm, do not generalize'}\n"
            + f"Caveats (they limit claims; they do not stop you writing): {'; '.join(caveats) if caveats else 'none'}\n\n"
            + rule
            + "Write it now. Return ONLY a JSON object with three keys:\n"
              "subject: 5-7 words, specific to them, no clickbait.\n"
              "body: one short paragraph, 3-5 sentences, 50-90 words, starting "
              "with 'Hi {first_name},' - no headers, no bullets, no markdown, no "
              "quoted blocks.\n"
              "chosen: the candidate number you used as the opener (0 if you "
              "used the personal opener instead).\n"
        )
        out = self._generate(user, system=sysp, as_json=True)
        if not out:
            return None
        subj = (out.get("subject") or "").strip()
        body = (out.get("body") or "").strip()
        if not body:
            return None
        # banned-phrase re-roll: if the cheap model slipped, one rewrite pass
        hits = [b for b in self.BANNED_PHRASES if b.lower() in body.lower()]
        if hits:
            out2 = self._generate(
                user + f"\n\nYour previous draft used banned phrase(s): {hits}. "
                       "Rewrite it without any banned phrases, keeping the "
                       "specificity. Return ONLY the JSON again.",
                system=sysp, as_json=True)
            if out2 and out2.get("body"):
                subj = (out2.get("subject") or subj).strip()
                body = out2["body"].strip()
        chosen_idx = int(out.get("chosen") or 0)
        picked = (cand[chosen_idx - 1]["headline"]
                  if (cand and 1 <= chosen_idx <= len(cand)) else "")
        grounded_sig = (["chosen icebreaker: " + picked[:60]] if picked else
                        (["chosen signal: " + signal.get("headline", "")[:60]]
                         if signal else []))
        return {"subject": subj or f"Quick note for {first.split(' ')[0]}",
                "body": body,
                "chosen_icebreaker": picked,
                "grounded_on": grounded + grounded_sig}

    # --------------------------------------------------------------- internals
    def _generate(self, prompt, system=None, as_json=False, max_tokens=400):
        """Call Gemini generateContent with retry on 503/429. Returns the
        parsed JSON object (if as_json) or the raw text, or None on any failure.

        NB: the flag is `as_json` on purpose — naming it `json` shadows the
        stdlib `json` module and breaks json.dumps/json.loads below."""
        import json as _json
        if not self.enabled:
            return None
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}"
               f":generateContent?key={self._key}")
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        last_err = None
        for attempt in range(self._retries + 1):
            try:
                resp = net.https_open(url, headers={"Content-Type": "application/json"},
                                      data=_json.dumps(payload).encode(), method="POST",
                                      timeout=45)
                data = _json.loads(resp.read().decode())
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                if as_json:
                    return self._parse_json(text)
                return text
            except Exception as e:
                last_err = e
                s = str(e)
                if "503" in s or "429" in s or "high load" in s.lower() or "quota" in s.lower() or "exhausted" in s.lower():
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break  # non-retryable (4xx) -> stop
        return None

    @staticmethod
    def _parse_json(text):
        text = text.strip()
        text = text.removeprefix("{").removesuffix("}") if not (text.startswith("{") and text.endswith("}")) else text
        # strip code fences
        if text.startswith("```"):
            text = text.strip("`\n")
            text = text[4:] if text.startswith("json") else text
            text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            # try to find the first {...} block
            i, j = text.find("{"), text.rfind("}")
            if i != -1 and j > i:
                try:
                    return json.loads(text[i:j + 1])
                except Exception:
                    return None
            return None


def get_llm():
    """Construct the LLM instance the server hands to the pipeline. Cheap;
    reads env at construction time."""
    return LLM()

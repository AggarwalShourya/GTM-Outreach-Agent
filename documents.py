"""Raw document corpus for the seeded retrieval index.

Full documents (article bodies, job-post text, interview transcripts) — not
pre-baked summaries. The retrieval layer chunks, embeds, and searches these
the same way it searches live-fetched docs.

Dates are written against the demo's deterministic reference date
(`corpus.REFERENCE_DATE` = 2026-11-10) so recency scoring of SEEDED docs is
stable. Live (web-searched) docs use the real wall-clock date instead.
"""


def _d(id, company_key, company_name, source, published, type, title, body, url):
    return {
        "id": id, "company_key": company_key, "company_name": company_name,
        "source": source, "published": published, "type": type,
        "title": title, "body": body, "url": url,
    }


DOCS = [
    # ----------------------------------------------------------------- NORTHWIND (happy path)
    _d("nw-series-b", "northwind", "Northwind Cloud", "funding", "2026-10-20",
        "Series B raised",
        "Northwind Cloud raises $40M Series B to scale its enterprise sales organization",
        "Northwind Cloud, the usage-based analytics platform, announced today that it has "
        "closed a $40 million Series B led by Meridian Growth Partners. The round values the "
        "company at $310 million post-money. Northwind Cloud said the capital will be used "
        "primarily to build out its enterprise sales organization, adding field coverage in "
        "EMEA and expanding its go-to-market motion beyond the existing land-and-expand model.\n\n"
        "The company has been in growth mode all year, reporting 70% year-over-year revenue growth "
        "in its Q3 earnings note. Chief Revenue Officer Dana Whitfield told investors the company "
        "expects enterprise pipeline to roughly double over the next four quarters, which will "
        "require significantly more outbound coverage than the current team can sustain. "
        "Whitfield framed the round as a bet on sales capacity: 'We have more demand coming in "
        "than our go-to-market organization can currently work. The constraint is not product, "
        "it is coverage.'\n\n"
        "The company said it will open new sales offices in Amsterdam and Munich by Q1, with "
        "hiring concentrated in account executive, SDR, and sales engineering roles.",
        "https://example.com/news/northwind-series-b"),
    _d("nw-sdr-posting", "northwind", "Northwind Cloud", "job_postings", "2026-11-02",
        "Hiring spike",
        "Northwind Cloud is hiring three SDRs and a RevOps analyst — all 'research-heavy' roles",
        "Four open roles on the Northwind Cloud careers page point at the same workflow. The "
        "Senior SDR posting (posted 10/28) says the team 'works a high-volume outbound list and "
        "needs to research each prospect deeply before first contact,' and that the role is "
        "expected to work 40 to 60 named accounts a month. The RevOps Analyst posting (11/01) "
        "emphasizes 'prospecting data quality, enrichment, and research workflows' as the core "
        "responsibility. A Sales Engineer and an EMEA AE round out the current openings.\n\n"
        "The job descriptions are unusually explicit about the research burden. The SDR posting "
        "lists as a requirement: 'comfortable doing 15-20 minutes of independent research per "
        "prospect before drafting personalized outreach.' The analyst posting asks candidates to "
        "'own the signal pipeline: funding, hiring, and product events that make an account "
        "worth calling.' Both listings note the team currently loses roughly a day per rep per "
        "week to manual research and that the hire is expected to compress that.",
        "https://example.com/careers/northwind-sdr"),
    _d("nw-cto-blog", "northwind", "Northwind Cloud", "blog", "2026-10-28",
        "Public pain point",
        "Northwind Cloud CTO: 'Our SDRs lose 20 minutes on every prospect to research — and it shows in reply rates'",
        "In a long post on the company engineering blog, CTO Ravi Menon writes about the "
        "specific cost of manual prospect research at Northwind Cloud. Menon says his own sales "
        "engineering team measured the last quarter: 'the median rep spends 18-22 minutes of "
        "research per prospect before writing the first email, and our reply rates on "
        "researched accounts are 3x better than on templated ones.'\n\n"
        "His conclusion is blunt: the research quality is the differentiator, but it does not "
        "scale. 'We would all rather have reps call 50 well-researched prospects than 100 "
        "half-researched ones,' he writes, and notes the team has been 'building internal "
        "tooling for signal detection — funding, hiring, product news — and it works, but it "
        "is duct tape.' The post ends with Menon saying the company is 'open to buying a "
        "serious version of this instead of maintaining a fragile internal one,' and links to "
        "a short list of the signal types his team tracks: funding rounds, job postings, "
        "executive interviews, and recent press.",
        "https://example.com/blog/northwind-sdr-research"),
    _d("nw-launch", "northwind", "Northwind Cloud", "product", "2026-09-15",
        "Product launch",
        "Northwind Cloud ships analytics add-on, starting a second go-to-market motion",
        "Northwind Cloud launched a usage analytics add-on this week, bundling session and "
        "feature-usage data with its existing platform. The add-on is aimed at the product "
        "management buyers the core product does not currently reach, which means the company "
        "is going to market on two distinct motions simultaneously: the enterprise land motion "
        "and a new product-led, self-serve motion.\n\n"
        "The launch notes the add-on will be available on existing plans starting next month, "
        "with a free tier for small teams. Company leadership said the goal is to 'create a "
        "second funnel that feeds the enterprise pipeline rather than competing with it.' "
        "That is the classic 'new product, same team' stretch: two buyers, two motions, and "
        "the same SDR bench covering both.",
        "https://example.com/press/northwind-analytics-launch"),
    _d("nw-maya-post", "northwind", "Northwind Cloud", "social", "2026-11-05",
        "Exec post",
        "Northwind Cloud VP Marketing Maya Chen: 'We'd rather send 20 good notes than 200 cold ones'",
        "In a post shared to her professional network this week, Maya Chen, VP Marketing at "
        "Northwind Cloud, described the company's shift in outbound philosophy: 'we've stopped "
        "chasing volume. Our best quarter came from a much smaller list of highly-researched "
        "accounts, and the cost of that research is the reason we keep asking for help with "
        "it. The reps are not the bottleneck — the 20 minutes of research before the first "
        "line is.' The post drew 300+ reactions and a reply thread from several GTM leaders "
        "asking how she picks which signals to chase per account.",
        "https://example.com/social/maya-chen-post"),
    # ----------------------------------------------------------------- SMALLCO (quiet)
    _d("sc-pilot", "smallco", "Smallco Robotics", "press", "2026-03-04",
        "Early pilot",
        "Smallco Robotics completes first commercial pilot in warehouse robotics",
        "Smallco Robotics, a two-year-old startup out of the Boston robotics scene, announced "
        "on Tuesday that it has completed its first commercial pilot deployment with a "
        "regional logistics provider. The pilot, which ran for four months, covered roughly "
        "12,000 square feet of warehouse floor and two picking workflows.\n\n"
        "The company's co-founder described the pilot as 'proof that the autonomy stack works "
        "in a real building, not a test lab,' and said commercial terms were still being "
        "negotiated. Smallco has not announced funding since its seed round, has not reported "
        "revenue, and its public surface is thin: a company blog with quarterly updates, a "
        "small LinkedIn following, and a single open engineering position that has been on "
        "the board for months. There is no public sales organization, no press beyond the "
        "pilot note, and no recent hiring activity of any kind.",
        "https://example.com/news/smallco-pilot"),
    _d("sc-eng-role", "smallco", "Smallco Robotics", "job_postings", "2026-07-20",
        "Long-open role",
        "Smallco Robotics: one robotics engineering role, open for over three months",
        "The only open position on the Smallco Robotics careers page is a mid-level robotics "
        "engineer (perception stack, C++, ROS2) posted in late July. The posting has had no "
        "updates and no visible activity since. There are no sales, marketing, partnerships, "
        "or operations roles open at the company. Headcount, per public filings and the "
        "company's own 'about' page, is under fifteen people, and the founder has said in a "
        "podcast appearance that 'we do all of our own outreach until we have a field team.'",
        "https://example.com/careers/smallco-robotics-engineer"),
    # ----------------------------------------------------------------- VANTAGE (contradictory)
    _d("vt-series-c", "vantage", "Vantage AI", "funding", "2026-09-01",
        "Series C",
        "Vantage AI closes $60M Series C, announces aggressive expansion across four regions",
        "Vantage AI, the developer-tools company for LLM evaluation, closed a $60 million "
        "Series C on Monday, doubling its valuation from the previous round eighteen months "
        "earlier. The company's announcement page uses the phrase 'scale fast' in four "
        "separate places and describes the use of proceeds as 'aggressive expansion across "
        "four new regions, with a dedicated go-to-market team in each.'\n\n"
        "Chief Executive Omar Haddad told investors the company would grow its commercial "
        "organization 'threefold in twelve months' and that the constraint on growth at that "
        "point is sales capacity, not product. The announcement coincided with the opening of "
        "sales offices in Singapore and Frankfurt, and a public hiring push that peaked at "
        "eleven open sales and solutions roles within three weeks of the round.",
        "https://example.com/news/vantage-series-c"),
    _d("vt-layoffs", "vantage", "Vantage AI", "news", "2026-10-18",
        "Layoff / restructure",
        "Vantage AI restructures go-to-motion; ~10% of headcount trimmed in the most recent news cycle",
        "Industry news this month reported that Vantage AI has trimmed roughly 10% of its "
        "headcount, concentrated in the go-to-market organization that the Series C was "
        "supposed to build. The restructuring, disclosed in an internal letter reported by "
        "the outlet, closes the Singapore and Frankfurt sales offices that opened in "
        "September and consolidates sales coverage into two regions.\n\n"
        "The same letter, which the company has not publicly disputed, reframes the mandate: "
        "'the goal is no longer coverage — it is efficiency per account.' Several departing "
        "engineers and SDRs posted to professional networks within days; one described the "
        "team as 'doing a lot less outbound and a lot more per-person output, which is "
        "understandable given what the round cost.' The company did not comment directly on "
        "the layoff story, but its current career page shows only three open roles, all in "
        "product and engineering.",
        "https://example.com/news/vantage-restructure"),
    _d("vt-cto-interview", "vantage", "Vantage AI", "exec_interview", "2026-10-30",
        "Exec interview",
        "Vantage AI's CTO on 'doing more with a leaner team' — and what that does to a sales org",
        "In a wide-ranging interview published this week, Vantage AI CTO Priya Nair discussed "
        "the restructuring honestly. 'The Series C changed what we thought we were buying — we "
        "thought we were buying growth, and it turned out we were buying a deadline,' she "
        "said, describing the internal pivot from a coverage model to what she called 'an "
        "efficiency-first motion.'\n\n"
        "Asked what that means for the sales team, Nair was specific: 'the team is smaller "
        "and the list is smaller, which means every prospect has to be worth more. The reps "
        "we kept are the good ones, and they are doing research that used to take half a "
        "day in maybe an hour. I would take a tool that gets us that an hour back on every "
        "account over ten new SDRs, any day.' She said the company is 'actively looking at "
        "research and personalization tooling' and would 'prefer to buy it to building it "
        "again.' The interview is the first public post-restructuring statement from her, "
        "and it is unusually direct about the current internal priority.",
        "https://example.com/interviews/vantage-cto-priya-nair"),
    _d("vt-bench", "vantage", "Vantage AI", "press", "2026-10-22",
        "Hiring pullback",
        "Vantage AI's career page now shows three open roles, down from eleven",
        "A short comparison of the Vantage AI careers page against a three-months-earlier "
        "snapshot shows the open-role count down from eleven sales and solutions positions to "
        "three product and engineering roles. There is no press release announcing the "
        "change, and no commentary in the company's own channels, which is notable: the "
        "pullback is verifiable from the public job board but the company has not explained "
        "it. The three remaining roles are a staff backend engineer, a product designer, and "
        "a solutions engineer — the last one, a support-leaning role, is the only "
        "customer-facing position still open.",
        "https://example.com/news/vantage-careers-change"),
    # ----------------------------------------------------------------- LEDGERLINE (moved-on)
    _d("ll-seed", "ledgerline", "Ledgerline", "funding", "2026-10-05",
        "Seed round",
        "Ledgerline raises seed to expand founding team and first GTM motion",
        "Ledgerline, the bookkeeping-automation startup for small professional firms, closed "
        "a $6 million seed round this week led by a first-time fund focused on "
        "vertical SaaS. The use of proceeds, per the announcement, is 'to hire the founding "
        "commercial team' — that is, the company has no sales organization yet, and the "
        "funding is explicitly to create one. Co-founder and CEO Tomas Vega said the first "
        "roles to open will be two SDRs and a 'founder who can sell' position, with the "
        "expectation that the team will 'work a list of accountant and law firm practices "
        "that we already know from the pilot cohort.'\n\n"
        "The seed note is one of the clearest signals of a go-to-market motion about to be "
        "born: the product exists, the pilot cohort exists, and the money is specifically "
        "for the first outbound engine.",
        "https://example.com/news/ledgerline-seed"),
    _d("ll-launch", "ledgerline", "Ledgerline", "product", "2026-09-25",
        "Product launch he led",
        "Ledgerline launches self-serve plan, a move its Head of Product personally led",
        "Ledgerline has launched a self-serve pricing tier, a significant shift for a company "
        "that had sold only through founder-led calls. The launch was built and shepherded "
        "end-to-end by Marco Rossi, then Head of Product, who announced it himself and "
        "credited the founding design and product team. The self-serve motion is aimed at "
        "the smaller end of the professional-firm market, letting solo accountants and "
        "one-person law firms start without a sales call.\n\n"
        "The launch post is detailed and clearly Rossi's work: the onboarding copy, the "
        "pricing page, and the activation flow all trace back to his team. It is, on its "
        "face, the strongest personal signal in the Ledgerline story — a product move that "
        "its Head of Product personally built and announced.",
        "https://example.com/press/ledgerline-self-serve-launch"),
    _d("ll-rossi-move", "ledgerline", "Ledgerline", "news", "2026-10-30",
        "Person move",
        "Marco Rossi has left Ledgerline, joining a competitor as VP Product",
        "A short story this week reports that Marco Rossi has departed Ledgerline to join "
        "a competing firm as VP Product, effective mid-October. Rossi had been with the "
        "company since its founding and was the public face of its product; he announced the "
        "move himself on his professional network, thanking the team and pointing to the "
        "self-serve launch as 'the thing I'm most proud of from here.' The company confirmed "
        "the departure in a one-line statement and said the self-serve motion 'continues "
        "under the founding product team.' No successor for the Head of Product role has "
        "been announced, and no relevant sales or product leadership roles are currently "
        "open on the Ledgerline career page.",
        "https://example.com/news/rossi-departs-ledgerline"),
    _d("ll-founder-note", "ledgerline", "Ledgerline", "blog", "2026-10-20",
        "Founder note",
        "Ledgerline's founder on why the self-serve tier matters to the next hiring push",
        "On the Ledgerline founder blog, CEO Tomas Vega wrote this week about what the "
        "self-serve launch changes for the company's first commercial team: 'we now have a "
        "product that can start working for a solo accountant with no sales call, which "
        "means the SDRs we are about to hire are not cold-starting a list — they are "
        "working the ones who self-serve and tell us they want more.' He says the research "
        "the new team will do is 'the difference between a 2% reply rate and the 8% we "
        "think is possible when the account already knows the product works.' The post "
        "directly frames the next hiring push as a research-and-personalization problem.",
        "https://example.com/blog/ledgerline-self-serve-hiring"),
    # ----------------------------------------------------------------- ACME (ambiguous entity)
    _d("acme-retail", "acme", "Acme Inc", "press", "2026-10-12",
        "Retail expansion",
        "Acme Inc, the consumer retail brand, opens its 200th store",
        "Acme Inc, the consumer retail chain, opened its 200th store this month, in a "
        "suburban shopping center roughly 40 minutes from the company's headquarters. The "
        "brand, which sells household goods and apparel under the Acme name, has been "
        "expanding its physical footprint for the past three years and describes the 200th "
        "store as 'the start of our second growth phase.' The story is about a consumer "
        "retail company and its store network. It is not about the B2B infrastructure "
        "company that also uses the Acme name.",
        "https://example.com/news/acme-inc-200th-store"),
    _d("acme-systems-hire", "acme", "Acme Systems", "job_postings", "2026-10-25",
        "B2B sales hire",
        "Acme Systems, the B2B infrastructure company, is hiring a sales director for its platform product",
        "Acme Systems, a B2B infrastructure company (data pipelines, API tooling, the kind "
        "of thing that sounds like 'pipes and glue'), has posted a Sales Director role for "
        "its platform product. The posting describes a 'net-new business' mandate for "
        "mid-market technical buyers and says the team 'currently relies on founder-led "
        "sales and is ready to build the first real outbound motion.' It is about the B2B "
        "company, not the retail chain that shares the name. This is the entity a 'Director "
        "of Sales' profile would actually belong to.",
        "https://example.com/careers/acme-systems-sales-director"),
    # ----------------------------------------------------------------- HELIO (many signals, none relevant)
    _d("hl-ml-role", "helio", "Helio Analytics", "job_postings", "2026-11-01",
        "Hiring spike (ML)",
        "Helio Analytics posts two machine-learning engineering roles for its recommendation stack",
        "Helio Analytics, the e-commerce data and recommendation platform, has posted two "
        "open machine-learning engineering roles in the last week: one for the "
        "recommendation systems team (Python, PyTorch, large-scale feature stores) and "
        "one for the embedding infrastructure team (vector search, model serving at "
        "latency budgets under 40ms). Both are deep technical roles with no customer-facing "
        "component. The company is not hiring for sales, marketing, partnerships, or any "
        "other go-to-market function; its commercial team, per its own about page, is a "
        "small, stable pod of five people that has not changed size in a year.",
        "https://example.com/careers/helio-ml-engineer"),
    _d("hl-oss", "helio", "Helio Analytics", "product", "2026-10-01",
        "Open-source release",
        "Helio Analytics open-sources its ranking library and hits a milestone on the public repo",
        "Helio Analytics open-sourced its internal ranking library this week and crossed a "
        "milestone in public repo stars. The library is a fast, GPU-aware ranking "
        "implementation, and the release post is a dense, technical document about "
        "approximate nearest-neighbor indexes and serving cost. The open-source effort is "
        "an engineering-credibility play: the company builds in public and its release notes "
        "read like a systems paper. There is no sales motion attached to it, and the post "
        "does not discuss customers, pipeline, or outbound in any form.",
        "https://example.com/blog/helio-ranking-oss"),
    _d("hl-talk", "helio", "Helio Analytics", "press", "2026-09-15",
        "Conference talk",
        "Helio Analytics' research lead gives a conference talk on vector search at scale",
        "A Helio Analytics research lead gave a well-received talk this month at a "
        "databases-and-search conference, walking through how the company's recommendation "
        "stack handles vector search at query volumes the audience described as "
        "unrealistic for most teams. The talk generated solid engagement in the "
        "engineering community and a thread of follow-up posts from other infra teams "
        "asking about the serving architecture. It is a strong signal of the company's "
        "technical standing and a weak one for anything commercial: the talk does not "
        "discuss sales, customers, or GTM at all.",
        "https://example.com/news/helio-conf-talk"),
    _d("hl-datacenter", "helio", "Helio Analytics", "press", "2026-08-01",
        "Datacenter expansion",
        "Helio Analytics expands its inference datacenter footprint to a second region",
        "Helio Analytics has committed to a second-region datacenter expansion for its "
        "inference workloads, a move its engineering blog describes as 'boring, "
        "consequential, and worth doing now rather than at 2x the current QPS.' The "
        "announcement is a cost-structure story: the company is prepaying for capacity so "
        "that inference at customer scale stays under a per-query cost ceiling. It says "
        "nothing about go-to-market, and the region choice (a low-latency location close "
        "to where its largest three accounts run) is the only commercial-adjacent detail "
        "in the piece.",
        "https://example.com/news/helio-datacenter-region"),
]


def all_docs():
    """Flat list of the raw corpus. This is what the index builder reads."""
    return list(DOCS)

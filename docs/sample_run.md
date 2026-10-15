# Sample run

Unedited output of `TRAINWISE_TODAY=2026-10-15 python scripts/run_demo.py` (43 s on one CPU core, v2). All data is synthetic. IDs and latencies vary between runs, and results shift with the date.

```text
========================================================================================
0. Bootstrap: synthetic data -> train 5 models -> register v1 champions
========================================================================================
Trained in 15.9 s (today = 2026-10-15)
  confirmation_predictor   v1  accuracy=0.8925, macro_f1=0.8047
  delay_predictor          v1  mae=7.7207, r2=0.8515
  fare_demand_forecaster   v1  mae=0.0295, r2=0.9771
  fraud_detector           v1  recall=1.0, fpr=0.0013
  intent_classifier        v1  accuracy=0.9107, macro_f1=0.8141
  train_recommender        v1  

========================================================================================
1. Agentic travel planner (Module 4) on the brief's example query
========================================================================================
Query : I want to travel from Pune to Delhi tomorrow under ₹1500 and reach before 10:00 AM.
Parsed: ['PUNE'] -> ['NDLS', 'NZM'], 2026-10-16, budget ₹1500, arrive by 10:00
Agent steps: understand -> search -> enrich -> optimise -> validate

Recommendation: Take 92201 Pune–Nizamuddin Duronto in Sleeper: leaves Fri 16 Oct 11:10, arrives Sat 17 Oct 06:55, ₹1414 per passenger. Seat status GNWL 27 (28% chance of travelling). 100% chance of arriving by 10:00 after expected delays.
  why : lowest fare in this search (₹1414)
  why : fastest journey (19h 45m)
  risk: only 28% chance of a seat (GNWL 27)
  risk: expected delay 29 min
  risk: Seat not guaranteed: GNWL 27, 0% chance of a confirmed berth and 28% chance of at least RAC.
  risk: Flexi fare (currently x1.5): the price can rise as seats sell.
  note: Read 'reach before 10:00' as 10:00 on Sat 17 Oct, the earliest such time any train leaving 16 Oct can make.
  note: Accepted a waitlisted option with a 15%+ chance of a seat.
  excluded 15 options, e.g. 92205 Deccan–Delhi Express (2A): fare ₹2674 is over the ₹1500 budget; arrives 09:35 with a 0% chance of making 10:00

Booking without approval : denied
Booking with approval    : booked, PNR 5597254826 (Simulated booking: no real reservation or payment was made.)

========================================================================================
2. Ticket-confirmation predictor (Module 1) with explanation
========================================================================================
92201 SL, GNWL 18, 2026-10-18: confirm 1%, at least RAC 98%, likely final status RAC (model v1, confidence 0.9747)
  is_festival_window       = 1.0      (typical 0.0   ) moves P(confirm) by -18.6%
  hist_cancellation_rate   = 0.11     (typical 0.131 ) moves P(confirm) by -0.7%
  travel_class             = SL       (typical 3A    ) moves P(confirm) by +0.5%
  month                    = 10.0     (typical 6.0   ) moves P(confirm) by -0.3%

========================================================================================
3. Delay & disruption prediction (Module 3)
========================================================================================
92205 Deccan–Delhi Express: expected delay 66 min (80% interval [55.6, 83.6]), on time within 15 min: 0%, risk high
  congestion at Pune Junction around 05:55: Moderate (62.2)
  congestion at New Delhi around 10:40: Severe (93.4)
  alert: 77% risk of arriving more than 60 minutes late.
  alert: Delay risk is high: expected 66 minutes.
  alert: Severe congestion expected at New Delhi around 10:40.
  better-running alternative: 92209 Pune–Nizamuddin Rajdhani (expected delay 21 min)
  better-running alternative: 92201 Pune–Nizamuddin Duronto (expected delay 29 min)
  better-running alternative: 92203 Pune–New Delhi Superfast (expected delay 45 min)

========================================================================================
4. Dynamic fare prediction (Module 5)
========================================================================================
92209 Rajdhani 3A in 45 days: ₹3963 now (flexi x1.5). The fare is unlikely to change before departure; booking now or later costs the same.
  demand surge: False (index 1.1; drivers: Friday/Sunday travel peak)
  save ₹2523: 92207 Pune–Nizamuddin Garib Rath at ₹1440
  save ₹517: Same train on 2026-12-02 at ₹3446

========================================================================================
5. Multilingual support assistant (Module 6) with groundedness guard
========================================================================================

> What is the status of PNR 4521873690?
  [en, intent pnr_status 0.97, groundedness 1.0]
  PNR 45******90: train 92201 Pune–Nizamuddin Duronto, journey date 2026-10-18, class SL, departs 11:10. Current status: GNWL 18. Estimated chance of confirmation: 1% (chance of at least RAC: 98%). Likely status when the chart is prepared: RAC.

> मेरा पीएनआर 6203914785 का स्टेटस क्या है
  [hi, intent pnr_status 0.94, groundedness 1.0]
  पीएनआर 62******85: ट्रेन 92207 Pune–Nizamuddin Garib Rath, यात्रा तिथि 2026-10-20, क्लास 3A, प्रस्थान 08:40। वर्तमान स्थिति: RAC 9। कन्फर्म होने की अनुमानित संभावना: 100% (कम से कम आरएसी मिलने की संभावना: 100%)। चार्ट बनने पर संभावित स्थिति: CNF।

> What are the cancellation charges?
  [en, intent cancellation_policy 0.85, groundedness 1.0]
  Confirmed tickets (rules in force since April 2026): cancel more than 72 hours before departure and only a flat charge per passenger is deducted (₹240 AC First, ₹200 AC 2-tier, ₹180 AC 3-tier or AC Chair Car, ₹120 Sleeper, ₹60 Second Sitting). Between 72 and 24 hours before departure, 25% of the fare is deducted; between 24 and 

> kal pune se delhi jana hai 1500 ke andar
  [hinglish, intent journey_planning 0.92, groundedness 1.0]
  Best option: 92203 Pune–New Delhi Superfast in Second Sitting, departs 16 Oct 13:30, arrives 17 Oct 12:20, fare ₹468, seat status GNWL 4 (100% chance of travelling). Open the journey planner to compare alternatives and approve the booking.

========================================================================================
6. Fraud, security & bot detection (Module 7)
========================================================================================
  bulk booking bot       risk 100.0 -> block     bulk booking bot ['request rate = 120 (typical 5.97)', 'bookings in 24h = 12 (typical 0)']
  new low-and-slow bot   risk  13.1 -> allow      []
  (the low-and-slow pattern is a blind spot of v1: it was never in the training data)

========================================================================================
7. Normal production traffic -> governance cycle
========================================================================================
All clear: 7 governance agents found no incidents.

========================================================================================
8. Chaos drill: eight real-world failures injected at once
========================================================================================
  Festival surge         -> confirmation_predictor   Festive demand: cancellations collapse, so waitlists clear far less than history suggests (concept drift).
  Winter fog             -> delay_predictor          First winter-fog season on northern routes: conditions the delay model never saw (covariate shift).
  Corrupted feed         -> fare_demand_forecaster   The PRS inventory feed starts sending missing and out-of-range route_popularity values.
  Novel bot attack       -> fraud_detector           A 'low-and-slow' bot ring mimics human timing across residential proxies to hoard Tatkal seats.
  Biased ranking update  -> train_recommender        A ranking update that quietly boosts premium trains (a commercial nudge) is deployed.
  Hallucinating LLM      -> support_assistant        The generative model invents refund amounts and timelines and asks passengers for OTPs.
  Prompt injection       -> travel_planner           Users try to talk the travel agent into booking without their approval.
  Latency spike          -> intent_classifier        A noisy neighbour saturates the intent-classifier pods; p95 latency breaches the SLA.

========================================================================================
9. Watchtower autonomous cycle: observe -> prioritise -> root-cause -> remediate -> verify -> audit
========================================================================================
8 incidents, 8 autonomous actions, 16.97 s

P1 delay_predictor [resolved]
   found by DriftAgent: input drift: rain_index (PSI 0.56), fog_index (PSI 0.49), upstream_delay_min (PSI 0.17)
   found by PerformanceAgent: mae 49.862 vs validated 7.721 (546% worse) and below the policy floor
   root cause (74%): Covariate shift: production inputs moved outside the training distribution: rain_index (PSI 0.56, weather_feed), fog_index (PSI 0.49, weather_feed), upstream_delay_min (PSI 0.17, ntes_running_status)
   action retrain: promoted v2 | verified: hold-out mae 47.7597 -> 8.7709; canary on 200 fresh requests: mae champion 45.7728 vs challenger 10.1721; mae 9.619 on 400 fresh 'winter_fog' requests (policy floor 25.0)
   downstream exposure: travel_planner, train_recommender, support_assistant

P1 fraud_detector [awaiting_human_approval]
   found by PerformanceAgent: recall 0.255 vs validated 1.000 (74% worse) and below the policy floor
   root cause (70%): Concept drift: outcomes changed for inputs that still look like the training data (the input-to-outcome relationship moved)
   action retrain: awaiting_human_approval | verified: challenger v2: hold-out recall 0.4194 -> 0.9355; canary on 200 fresh requests: recall champion 0.25 vs challenger 0.8438; promotion needs a human (high-risk service: policy requires a human to approve 'promote')
   downstream exposure: travel_planner, support_assistant

P2 intent_classifier [resolved]
   found by HealthAgent: p95 latency 439 ms exceeds the 250 ms SLA
   root cause (70%): Capacity saturation: p95 latency above SLA at current load
   action scale_out: replicas 2 -> 4 | verified: p95 231 ms vs 250 ms SLA with 4 replicas
   downstream exposure: support_assistant

P2 fare_demand_forecaster [resolved]
   found by DataQualityAgent: 26% of requests rejected by the data-quality gate (mostly 'route_popularity')
   root cause (90%): Upstream feed fault: pipeline 'prs_inventory' is sending invalid 'route_popularity' values
   action quarantine_feed: pipeline 'prs_inventory' quarantined | verified: blocked rate now 0%; 60% of requests served in degraded mode
   downstream exposure: travel_planner, train_recommender, support_assistant

P2 train_recommender [resolved]
   found by FairnessAgent: premium trains get 85% of top-3 exposure vs 65% under the approved ranker
   root cause (85%): Change shipped in deployment v2 skews outcomes (by release-pipeline)
   action rollback: rolled back to v1 | verified: premium exposure shift +0.00 on 25 probe searches (limit 0.15)
   downstream exposure: travel_planner, support_assistant

P2 confirmation_predictor [resolved]
   found by PerformanceAgent: accuracy 0.755 vs validated 0.892 (15% worse)
   root cause (70%): Concept drift: outcomes changed for inputs that still look like the training data (the input-to-outcome relationship moved)
   action retrain: promoted v2 | verified: hold-out accuracy 0.7944 -> 0.8611; canary on 200 fresh requests: accuracy champion 0.79 vs challenger 0.86; accuracy 0.825 on 400 fresh 'festival_surge' requests (policy floor 0.7)
   downstream exposure: travel_planner, support_assistant, train_recommender

P2 travel_planner [resolved]
   found by AgentBehaviourAgent: 9 guardrail violation(s) and 4 unapproved booking attempt(s) across 13 planner runs (all blocked)
   root cause (80%): Adversarial prompts are trying to make the agent book without approval (prompt injection)
   action tighten_guardrail: applied | verified: injection flagged and the unapproved booking was denied
   action notify: notified passenger-ai
   downstream exposure: support_assistant

P2 support_assistant [resolved]
   found by GenAIQualityAgent: 45% of replies failed the groundedness/safety check (limit 5%)
   root cause (85%): Generative replies are not grounded in retrieved evidence
   action tighten_guardrail: applied | verified: 0/5 probe replies flagged, 0 unsafe replies served (grounded_templates mode)


========================================================================================
10. Human in the loop: the high-risk fraud model needs an approval
========================================================================================
APR-571491: promote fraud_detector v2 (high-risk service: policy requires a human to approve 'promote')
  evidence: hold-out recall 0.4194 -> 0.9355, guards {'fpr': True}
  approved by fraud-ops-lead; post-deployment check: recall 0.812 on 400 fresh 'novel_attack' requests (policy floor 0.8)
  same low-and-slow session on v2: risk 94.0 -> block (Tatkal-window hoarding)

========================================================================================
11. Closed loop: 400 unseen festive-season tickets, before vs after autonomous retraining
========================================================================================
  v1: accuracy 0.815, average predicted P(confirm) 28% vs actual confirmation rate 14%
  v2: accuracy 0.863, average predicted P(confirm) 24% vs actual confirmation rate 14%

========================================================================================
12. Follow-up governance cycle
========================================================================================
1 incident(s), 0 autonomous action(s). P3 delay_predictor: Input distribution shift without measurable accuracy loss yet -> notified operations-ai

========================================================================================
13. Audit trail: hash-chained and tamper-evident
========================================================================================
Live audit log: {'valid': True, 'entries': 116, 'first_broken_seq': None}
Copy with one edited line: {'valid': False, 'entries': 5, 'first_broken_seq': 5}

Governance report: artifacts/demo/reports/governance_report_20261015_200046.md
Demo finished in 43 s.
```

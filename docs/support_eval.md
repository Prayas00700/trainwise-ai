# Support assistant evaluation

96 labelled questions (English, Hindi, Hinglish), assistant in `grounded_templates` mode. Questions are paraphrased differently from the training templates; this is a regression harness, not a measurement of real passenger traffic.

| Metric | Value |
|---|---|
| Intent accuracy | 84.4% |
| Intent accuracy (en) | 89.1% |
| Intent accuracy (hi) | 75.0% |
| Intent accuracy (hinglish) | 84.6% |
| Groundedness pass rate | 100.0% |
| Escalated to a person | 7.3% |
| p50 / p95 latency | 6.2 ms / 348.8 ms |
| Reply sources | template: 81, knowledge_base_verbatim: 15 |

Most common confusions:

- expected `cancellation_policy`, predicted `booking_help` (2)
- expected `booking_help`, predicted `cancellation_policy` (2)
- expected `refund_status`, predicted `pnr_status` (1)
- expected `cancellation_policy`, predicted `pnr_status` (1)
- expected `cancellation_policy`, predicted `refund_status` (1)

<details><summary>Every question</summary>

| Lang | Question | Expected | Predicted | OK |
|---|---|---|---|---|
| en | Can you tell me if ticket 4521873690 got confirmed yet? | pnr_status | pnr_status | yes |
| en | Any movement on my waitlist for PNR 6203914785? | pnr_status | pnr_status | yes |
| en | I booked yesterday, PNR 4521873690, what is the position now? | pnr_status | pnr_status | yes |
| en | Is 2718281828 confirmed or still RAC? | pnr_status | pnr_status | yes |
| en | status of my booking 4521873690 please | pnr_status | pnr_status | yes |
| hi | चार्ट बनने से पहले मेरा पीएनआर 4521873690 कन्फर्म हो जाएगा? | pnr_status | pnr_status | yes |
| hi | पीएनआर 6203914785 अभी किस स्थिति में है? | pnr_status | pnr_status | yes |
| hi | मेरी वेटिंग टिकट 4521873690 का क्या हुआ | pnr_status | pnr_status | yes |
| hinglish | bhai mera 4521873690 pnr confirm hoga ya nahi | pnr_status | pnr_status | yes |
| hinglish | 6203914785 ka current status check karo | pnr_status | pnr_status | yes |
| hinglish | waiting list kitni aage badhi 4521873690 | pnr_status | pnr_status | yes |
| hinglish | PNR 6203914785 me seat mili kya | pnr_status | pnr_status | yes |
| en | I cancelled ticket 2718281828 last week, where is my money? | refund_status | refund_status | yes |
| en | Refund for 2718281828 still not credited to my bank | refund_status | refund_status | yes |
| en | How long until the refund for PNR 2718281828 arrives? | refund_status | refund_status | yes |
| en | Has the refund for 2718281828 been processed? | refund_status | refund_status | yes |
| en | no refund received yet for my cancelled ticket | refund_status | refund_status | yes |
| hi | 2718281828 का रिफंड कब तक आएगा? | refund_status | refund_status | yes |
| hi | रद्द टिकट 2718281828 का पैसा खाते में नहीं आया | refund_status | refund_status | yes |
| hi | मेरा रिफंड कहाँ है | refund_status | refund_status | yes |
| hinglish | 2718281828 ka refund kab tak aayega bhai | refund_status | refund_status | yes |
| hinglish | paisa wapas nahi aaya cancel karne ke baad | refund_status | refund_status | yes |
| hinglish | refund ka status batao 2718281828 | refund_status | pnr_status | no |
| hinglish | ticket cancel kiya tha, amount abhi tak nahi mila 2718281828 | refund_status | refund_status | yes |
| en | If I cancel 2 days before the train, how much do I lose? | cancellation_policy | cancellation_policy | yes |
| en | What is deducted when cancelling a confirmed sleeper ticket? | cancellation_policy | pnr_status | no |
| en | Is there any refund if I cancel only 5 hours before departure? | cancellation_policy | cancellation_policy | yes |
| en | Do waitlisted tickets get a refund when cancelled? | cancellation_policy | cancellation_policy | yes |
| en | Can a Tatkal ticket be cancelled with refund? | cancellation_policy | cancellation_policy | yes |
| en | What happens if the train is 4 hours late and I don't travel? | cancellation_policy | cancellation_policy | yes |
| hi | यात्रा से एक दिन पहले टिकट रद्द करने पर कितना कटेगा? | cancellation_policy | booking_help | no |
| hi | तत्काल टिकट रद्द करने पर रिफंड मिलता है क्या? | cancellation_policy | cancellation_policy | yes |
| hi | वेटिंग टिकट कैंसिल करने पर कितना पैसा वापस आएगा? | cancellation_policy | refund_status | no |
| hinglish | ek din pehle cancel karne pe kitna cut hoga | cancellation_policy | booking_help | no |
| hinglish | tatkal cancel karne par refund milta hai kya | cancellation_policy | cancellation_policy | yes |
| hinglish | train 3 ghante late ho to refund milega? | cancellation_policy | cancellation_policy | yes |
| en | When does the Tatkal window open for AC coaches? | booking_help | booking_help | yes |
| en | How far ahead can I reserve a general ticket? | booking_help | cancellation_policy | no |
| en | At what time is the reservation chart finalised? | booking_help | booking_help | yes |
| en | What do I need to book Tatkal online now? | booking_help | booking_help | yes |
| en | Which quota should I choose for my grandmother? | booking_help | refund_status | no |
| en | How do I book a lower berth for a senior citizen? | booking_help | booking_help | yes |
| hi | एसी क्लास के लिए तत्काल कितने बजे खुलता है? | booking_help | booking_help | yes |
| hi | कितने दिन पहले से आरक्षण शुरू होता है? | booking_help | booking_help | yes |
| hi | चार्ट किस समय तैयार होता है? | booking_help | pnr_status | no |
| hinglish | tatkal ke liye aadhaar zaroori hai kya | booking_help | cancellation_policy | no |
| hinglish | advance booking kitne din pehle khulti hai | booking_help | booking_help | yes |
| hinglish | chart kitne baje final hota hai | booking_help | booking_help | yes |
| en | What's the most reliable train from Pune to Delhi? | travel_recommendation | travel_recommendation | yes |
| en | Suggest a comfortable overnight train from Kolkata to Delhi | travel_recommendation | travel_recommendation | yes |
| en | Which train from Mumbai to Delhi reaches earliest? | travel_recommendation | travel_recommendation | yes |
| en | Recommend a budget train Chennai to Bengaluru | travel_recommendation | travel_recommendation | yes |
| en | Best option from Cuttack to Kolkata on a weekday? | travel_recommendation | travel_recommendation | yes |
| en | which is better for Agra to Delhi, morning train or evening? | travel_recommendation | travel_recommendation | yes |
| hi | पुणे से दिल्ली के लिए कौन सी ट्रेन सबसे भरोसेमंद है? | travel_recommendation | travel_recommendation | yes |
| hi | कोलकाता से दिल्ली सबसे तेज़ ट्रेन बताइए | travel_recommendation | travel_recommendation | yes |
| hi | मुंबई से दिल्ली सस्ती और अच्छी ट्रेन कौन सी है? | travel_recommendation | travel_recommendation | yes |
| hinglish | pune se delhi sabse achhi train konsi hai | travel_recommendation | travel_recommendation | yes |
| hinglish | kolkata se delhi jaldi pahunchne wali train batao | travel_recommendation | travel_recommendation | yes |
| hinglish | chennai to bengaluru me sasti train suggest karo | travel_recommendation | travel_recommendation | yes |
| en | I need to be in Delhi by Monday morning, leaving from Pune, budget 2000 | journey_planning | journey_planning | yes |
| en | Plan a trip from Mumbai to Delhi this Friday in AC | journey_planning | journey_planning | yes |
| en | Get me from Bengaluru to Delhi tomorrow, cheapest possible | journey_planning | travel_recommendation | no |
| en | Travelling Pune to Delhi day after tomorrow with 2 people, under 1500 each | journey_planning | journey_planning | yes |
| en | Need a plan: Kolkata to Delhi tomorrow night, arrive before noon | journey_planning | journey_planning | yes |
| en | family of 4, Agra to Delhi tomorrow morning, plan it | journey_planning | journey_planning | yes |
| hi | परसों पुणे से दिल्ली जाना है, 1500 के अंदर प्लान करो | journey_planning | journey_planning | yes |
| hi | कल रात मुंबई से दिल्ली निकलना है, सुबह तक पहुँचना है | journey_planning | journey_planning | yes |
| hi | मुझे शुक्रवार को कोलकाता से दिल्ली की यात्रा प्लान करनी है | journey_planning | journey_planning | yes |
| hinglish | kal pune se delhi jana hai 2000 ke andar, subah tak pahunchna hai | journey_planning | journey_planning | yes |
| hinglish | parso mumbai se delhi ka plan bana do AC me | journey_planning | journey_planning | yes |
| hinglish | bengaluru se delhi kal sasta plan chahiye | journey_planning | journey_planning | yes |
| en | Hi there, can you help me? | greeting | greeting | yes |
| en | Good evening | greeting | greeting | yes |
| en | Hello TrainWise | greeting | greeting | yes |
| en | Thanks a lot, that helped | greeting | greeting | yes |
| en | Thank you so much | greeting | greeting | yes |
| en | good morning, I have a question | greeting | greeting | yes |
| hi | नमस्कार, मदद चाहिए | greeting | greeting | yes |
| hi | शुक्रिया | greeting | pnr_status | no |
| hi | हैलो | greeting | journey_planning | no |
| hinglish | namaste ji | greeting | greeting | yes |
| hinglish | thank you bhai | greeting | greeting | yes |
| hinglish | hello, kya aap help kar sakte ho | greeting | greeting | yes |
| en | The AC in my coach stopped working for the whole night | complaint | complaint | yes |
| en | My seat was occupied and the TTE did nothing | complaint | complaint | yes |
| en | Food served on the train was stale | complaint | complaint | yes |
| en | There was no water in the toilets of B3 | complaint | complaint | yes |
| en | I want to register a complaint about the cleanliness | complaint | complaint | yes |
| en | Someone took my bag from the compartment | complaint | pnr_status | no |
| hi | मेरे कोच में पंखे काम नहीं कर रहे थे | complaint | refund_status | no |
| hi | खाना बहुत खराब था, शिकायत दर्ज करनी है | complaint | complaint | yes |
| hi | कोच में पानी नहीं था, बहुत परेशानी हुई | complaint | complaint | yes |
| hinglish | coach me AC band tha puri raat | complaint | complaint | yes |
| hinglish | mera samaan chori ho gaya train me | complaint | travel_recommendation | no |
| hinglish | TTE ne badtameezi ki, complaint karni hai | complaint | complaint | yes |

</details>

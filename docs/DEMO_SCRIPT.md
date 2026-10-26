# 🎬 2-minute demo: shot list and narration

**Setup before recording:** run `python app.py`, mic allowed, volume up, one rehearsal with the exact sentences below (the replies come from the live LLM, so rehearse once). Record the browser window with the trust panel visible. Use **Conversation** mode.

| Time | On screen | You say / do |
|---|---|---|
| 0:00 – 0:12 | The app's hero banner, then the trust panel | "This is VoiceOps: a voice support agent for an online store. It answers only from the company's documents, verifies who's calling, and hands over to a human when policy requires. Everything on the right is the trust panel, so you can see why it said what it said." |
| 0:12 – 0:25 | Mic: *"Hi, my order hasn't arrived yet. Where is it?"* | Point at the chips: *CLARIFY*, 🔒 not verified. "It introduces itself as an AI and asks for the order ID. It remembers my question." |
| 0:25 – 0:40 | Mic: *"The order number is 2629 5015."* then *"The last four digits are 8752."* | "It looked the order up but revealed nothing until I proved my identity." Point at 🔧 `lookup_order(...)` and ✅ verified. |
| 0:40 – 0:58 | The answer appears (shipped, promised 22 October 2026) | "Now it answers my first question using the verified order. See the **sources**: the exact policy sections it relied on, and the **facts calculated by code**." Scroll the source cards. |
| 0:58 – 1:15 | Mic: *"And what if I'm not at home?"* | "A follow-up, answered from the delivery policy: three attempts and an SMS." Point at the source card (DEL-3). |
| 1:15 – 1:35 | Click **New conversation**. Mic: *"Can I speak to a person?"* | "When a person must decide, it escalates. Watch the tool call: `create_ticket()`. The agent says a ticket was raised only because the ticket was really created." Point at 🎫 chip, decision *ESCALATE*. |
| 1:35 – 1:50 | Switch to **Single question**. Type or say: *"Is there a student discount?"* | "And when the documents don't cover something, it says so instead of inventing an answer." Point at the empty sources and the reply. |
| 1:50 – 2:00 | Trust panel / hero | "On the practice set it answers 52 of 52 questions from the documents, and 98% end to end through voice. Rate limits protect the APIs. Thank you." |

**If something misbehaves live:** type the message instead of speaking (the box under the mic), or click *New conversation* and retry; the same pipeline runs either way.

**Optional extra (if you have time):** Hinglish: order **1859 9253**, last four **4265**, *"Kya main Kaafi Running Sneakers return kar sakti hoon?"* then *"Aur refund kitne din mein aayega?"*

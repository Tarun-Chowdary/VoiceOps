"""A rule-based fake for llm.chat so conversation logic can be tested with zero API calls."""
import json, re

HUMAN = re.compile(r"insaan|human|person|इंसान|agent|insan", re.I)


def make_fake(generate_reply="OK", generate_decision="answer", prompts=None):
    def fake(system, messages, **k):
        text = messages[-1]["text"]
        if "language-understanding module" in system:
            last = text.split("Latest customer message", 1)[-1] if "Latest customer message" in text else text
            if HUMAN.search(last):
                d = {"language": "en", "intent": "human_request", "english_query": "talk to a human", "needs_order": False}
            elif re.search(r"cancel|कैंसिल", last, re.I):
                d = {"language": "en", "intent": "cancel_order", "english_query": "cancel order", "needs_order": True}
            elif re.search(r"damage|टूट|crack", last, re.I):
                d = {"language": "en", "intent": "damaged_item", "english_query": "damaged item", "needs_order": True}
            elif re.search(r"arrived|where|kab ship|status", last, re.I):
                d = {"language": "en", "intent": "order_status", "english_query": "order status", "needs_order": True}
            elif re.search(r"coupon|policy|return window", last, re.I):
                d = {"language": "en", "intent": "return_window", "english_query": "return window", "needs_order": False}
            elif re.search(r"\bhelp\b|मदद|madad|नमस्ते|hello|hi,", last, re.I) and not re.search(r"refund|return", last, re.I):
                d = {"language": "en", "intent": "other", "english_query": "greeting", "needs_order": False}
            else:
                d = {"language": "en", "intent": "return_window", "english_query": "return window days", "needs_order": True}
            return json.dumps(d)
        if prompts is not None:
            prompts.append(text)
        return json.dumps({"decision": generate_decision, "reply": generate_reply, "used_sections": ["RET-2"]})
    return fake

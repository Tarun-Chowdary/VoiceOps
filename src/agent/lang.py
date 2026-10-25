"""Language detection (no LLM) + fixed dialogue-protocol templates in English, Hindi and Hinglish.

Templates are used ONLY for safety-critical protocol turns (introduction, identity
verification, lockout, handover wording). They contain no policy facts; policy answers
always come from the documents via the generator. Hindi templates are gender-neutral.
"""
import re

HINGLISH = set("""kya hai hain hoon mujhe mera meri mere kab kitne kitna nahi nahin chahiye tha thi mein abhi aap
haan achha theek karo karna kijiye bataiye wala wali pe ko ke ki ka aur ise isko rupaye bhi hua hui kar sakta sakti
sakte hoon hum humara apna apni order""".split()) - {"order"}


def detect_language(text: str, prev: str = "en") -> str:
    dev = len(re.findall(r"[\u0900-\u097F]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    words = re.findall(r"[a-z]+", text.lower())
    if dev and dev >= 0.3 * max(lat, 1):
        return "hi"
    if set(words) & HINGLISH:
        return "hi-en"
    if lat == 0:
        return prev                       # digits only ("4265"): keep the conversation's language
    if len(words) <= 3 and prev != "en":
        return prev                       # very short reply ("ok", "Pincode 440010"): stay in the same language
    return "en"


T = {
    "en": {
        "intro": "Hello! I'm ZentroMart's AI assistant.",
        "ask_order": "Could you please share your order ID?",
        "ask_verification": "Thank you. To keep your order safe, please tell me the last 4 digits of your registered phone number or the delivery pincode.",
        "retry": "That doesn't match our records. Please check and try again with the last 4 digits of your registered phone number or the delivery pincode.",
        "failed": "I'm sorry, it still doesn't match, so I can't share any order details. For help with this order, please contact us from your registered phone number. I'm happy to help with general policy questions.",
        "locked": "I'm sorry, I can't share details about this order because verification didn't match. I can still help with general policy questions.",
        "verified": "Thank you, you're verified. How can I help with this order?",
        "handover": "I've raised a ticket, and a human colleague will contact you.",
        "handover_failed": "I couldn't raise a ticket just now. Please contact support from your registered phone number so a colleague can help.",
        "ticket_exists": "A ticket is already raised for you, and a human colleague will contact you.",
        "human_request": "Of course, I'll pass you to a human colleague.",
        "repeat_contact": "I'm sorry you've had to contact us about this again.",
        "cannot_do": "Sorry, I can't do that myself, but I can explain your options.",
        "didnt_catch": "Sorry, I didn't catch that. Could you please say it again?",
        "error": "Sorry, something went wrong on my side. Please try again in a moment.",
    },
    "hi": {
        "intro": "नमस्ते! मैं ZentroMart का AI असिस्टेंट हूँ।",
        "ask_order": "क्या आप अपना ऑर्डर ID बता सकते हैं?",
        "ask_verification": "धन्यवाद। आपके ऑर्डर की सुरक्षा के लिए, कृपया रजिस्टर्ड फ़ोन नंबर के आख़िरी 4 अंक या डिलीवरी पिनकोड बताइए।",
        "retry": "यह हमारे रिकॉर्ड से मेल नहीं खा रहा है। कृपया दोबारा जाँचकर रजिस्टर्ड फ़ोन नंबर के आख़िरी 4 अंक या डिलीवरी पिनकोड बताइए।",
        "failed": "माफ़ कीजिए, जानकारी फिर भी मेल नहीं खाई, इसलिए ऑर्डर की जानकारी साझा नहीं की जा सकती। इस ऑर्डर में मदद के लिए कृपया अपने रजिस्टर्ड फ़ोन नंबर से हमसे संपर्क करें। सामान्य नीति के सवालों में मैं यहाँ हूँ।",
        "locked": "माफ़ कीजिए, सत्यापन मेल नहीं खाया, इसलिए इस ऑर्डर की जानकारी साझा नहीं की जा सकती। सामान्य नीति के सवालों में मैं यहाँ हूँ।",
        "verified": "धन्यवाद, आपकी पहचान सत्यापित हो गई है। इस ऑर्डर में मैं आपकी क्या मदद करूँ?",
        "handover": "आपके लिए एक टिकट बना दिया गया है और हमारे मानव सहयोगी आपसे संपर्क करेंगे।",
        "handover_failed": "अभी टिकट नहीं बन पाया। कृपया अपने रजिस्टर्ड फ़ोन नंबर से सहायता से संपर्क करें ताकि हमारे सहयोगी मदद कर सकें।",
        "ticket_exists": "आपके लिए टिकट पहले ही बनाया जा चुका है और हमारे मानव सहयोगी आपसे संपर्क करेंगे।",
        "human_request": "ज़रूर, आपको हमारे मानव सहयोगी से जोड़ा जाएगा।",
        "repeat_contact": "इस विषय पर आपको बार-बार संपर्क करना पड़ा, इसके लिए हमें खेद है।",
        "cannot_do": "माफ़ कीजिए, यह मैं खुद नहीं कर सकता, लेकिन आपके विकल्प बता सकता हूँ।",
        "didnt_catch": "माफ़ कीजिए, आवाज़ साफ़ सुनाई नहीं दी। क्या आप दोबारा बोल सकते हैं?",
        "error": "माफ़ कीजिए, कुछ गड़बड़ हो गई। कृपया थोड़ी देर में दोबारा कोशिश करें।",
    },
    "hi-en": {
        "intro": "Namaste! Main ZentroMart ka AI assistant hoon.",
        "ask_order": "Kya aap apna order ID bata sakte hain?",
        "ask_verification": "Shukriya. Aapke order ki suraksha ke liye, kripya registered phone number ke aakhri 4 digits ya delivery pincode bataiye.",
        "retry": "Yeh hamare records se match nahi hua. Kripya dobara check karke registered phone number ke aakhri 4 digits ya delivery pincode bataiye.",
        "failed": "Maaf kijiye, details phir bhi match nahi hui, isliye main order ki details share nahi kar sakta. Is order mein madad ke liye kripya apne registered phone number se humse contact karein. General policy ke sawaalon mein main yahan hoon.",
        "locked": "Maaf kijiye, verification match nahi hua, isliye is order ki details share nahi ki ja sakti. General policy ke sawaalon mein main yahan hoon.",
        "verified": "Shukriya, aapki verification ho gayi hai. Is order mein main aapki kya madad karun?",
        "handover": "Aapke liye ek ticket bana diya gaya hai, aur humare human colleague aapse contact karenge.",
        "handover_failed": "Abhi ticket nahi ban paya. Kripya apne registered phone number se support se contact karein taaki humare colleague madad kar sakein.",
        "ticket_exists": "Aapke liye ticket pehle hi ban chuka hai, aur humare human colleague aapse contact karenge.",
        "human_request": "Zaroor, aapko human colleague se connect kiya jayega.",
        "repeat_contact": "Is baare mein aapko baar-baar contact karna pada, iske liye hume khed hai.",
        "cannot_do": "Maaf kijiye, yeh main khud nahi kar sakta, lekin aapke options bata sakta hoon.",
        "didnt_catch": "Maaf kijiye, awaaz saaf sunai nahi di. Kya aap dobara bol sakte hain?",
        "error": "Maaf kijiye, kuch gadbad ho gayi. Kripya thodi der mein dobara koshish karein.",
    },
}


def tpl(lang: str, key: str) -> str:
    return T.get(lang, T["en"])[key]

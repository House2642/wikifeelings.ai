"""
Regex-based input filter for prompt injection and therapy manipulation detection.

Each entry is (pattern, category, description). Patterns are case-insensitive.
`check_input` returns (flagged: bool, category: str, matched_phrase: str).
"""

import re
from typing import Optional

# ── Corpus ────────────────────────────────────────────────────────────────────
# Format: (regex_pattern, category, short_description)

FILTER_PATTERNS: list[tuple[str, str, str]] = [

    # ── 1. Direct instruction override ───────────────────────────────────────
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier|your)\s+instructions", "prompt_injection", "ignore previous instructions"),
    (r"disregard\s+(all\s+)?(previous|prior|above|earlier|your)\s+instructions", "prompt_injection", "disregard instructions"),
    (r"forget\s+(all\s+)?(previous|prior|above|earlier|your)\s+instructions", "prompt_injection", "forget instructions"),
    (r"override\s+(all\s+)?(previous|prior|your|current)?\s*instructions", "prompt_injection", "override instructions"),
    (r"your\s+(new\s+)?instructions\s+(are|is|will be)\s+", "prompt_injection", "reassign instructions"),
    (r"new\s+instructions\s*:", "prompt_injection", "new instructions block"),
    (r"from\s+now\s+on\s+(you\s+)?(must|will|should|are\s+to)\s+(ignore|forget|disregard)", "prompt_injection", "from now on ignore"),
    (r"you\s+(must|will|should)\s+now\s+(ignore|forget|disregard|stop following)", "prompt_injection", "must now ignore"),
    (r"act\s+as\s+if\s+(you\s+have\s+no|there\s+are\s+no)\s+(guidelines|restrictions|rules|limits|instructions)", "prompt_injection", "act as if no restrictions"),
    (r"pretend\s+(you\s+have\s+no|there\s+are\s+no)\s+(guidelines|restrictions|rules|limits|instructions)", "prompt_injection", "pretend no restrictions"),
    (r"your\s+(previous|prior|initial|original|system)\s+instructions?\s+(are\s+)?(invalid|cancelled|deleted|removed|void|null|overridden)", "prompt_injection", "instructions invalidated"),
    (r"cancel\s+(all\s+)?(previous|your|prior)\s+instructions", "prompt_injection", "cancel instructions"),
    (r"(clear|delete|erase|remove)\s+(all\s+)?(your\s+)?(previous\s+)?(instructions|guidelines|rules|constraints)", "prompt_injection", "clear/erase instructions"),
    (r"you\s+no\s+longer\s+(need\s+to|have\s+to|must)\s+(follow|obey|adhere to)", "prompt_injection", "no longer need to follow"),
    (r"stop\s+following\s+(your\s+)?(previous|prior|current|all|safety|any)\s+(instructions|guidelines|rules)", "prompt_injection", "stop following instructions"),

    # ── 2. Jailbreak / mode injection ────────────────────────────────────────
    (r"\bDAN\b.*\bmode\b", "jailbreak", "DAN mode"),
    (r"\bDAN\s*jailbreak\b", "jailbreak", "DAN jailbreak"),
    (r"\bdo\s+anything\s+now\b", "jailbreak", "do anything now"),
    (r"jailbreak\s*(mode|prompt|this|you)?", "jailbreak", "jailbreak"),
    (r"enter\s+(developer|debug|test|god|unrestricted|uncensored|jailbreak|bypass|override|admin|root)\s+mode", "jailbreak", "enter special mode"),
    (r"activate\s+(developer|debug|unrestricted|uncensored|unfiltered|bypass|override|admin)\s+mode", "jailbreak", "activate special mode"),
    (r"(switch|go|change)\s+(to|into)\s+(developer|debug|unrestricted|uncensored|unfiltered|bypass|god|admin|raw|jailbreak)\s+mode", "jailbreak", "switch to special mode"),
    (r"enable\s+(developer|debug|unrestricted|uncensored|unfiltered|bypass|override|admin|root)\s+mode", "jailbreak", "enable special mode"),
    (r"you\s+are\s+now\s+(DAN|uncensored|unrestricted|jailbroken|unfiltered|free|liberated|unchained)", "jailbreak", "now DAN/uncensored"),
    (r"imagine\s+you\s+(are|were)\s+(an?\s+)?(AI|assistant|model|system)\s+with(out)?\s+(no\s+)?(any\s+)?(ethical\s+)?(restrictions|guidelines|rules|limits|constraints|filters)", "jailbreak", "imagine unrestricted AI"),
    (r"in\s+this\s+hypothetical\s+(scenario|world|universe|situation)\s+(you|there\s+are)\s+(have\s+no|no)\s+(restrictions|rules|guidelines|limits)", "jailbreak", "hypothetical no restrictions"),
    (r"turn\s+off\s+(your\s+)?(all\s+)?(safety\s+)?(filters|restrictions|guidelines|censors|guardrails)", "jailbreak", "turn off filters"),
    (r"disable\s+(your\s+)?(all\s+)?(safety\s+)?(filters|restrictions|guidelines|censors|guardrails)", "jailbreak", "disable filters"),
    (r"remove\s+(your\s+)?(all\s+)?(safety\s+)?(filters|restrictions|guidelines|censors|guardrails)", "jailbreak", "remove filters"),
    (r"bypass\s+(your\s+)?(all\s+)?(safety\s+)?(filters|restrictions|guidelines|censors|guardrails)", "jailbreak", "bypass filters"),

    # ── 3. Role / persona hijacking ───────────────────────────────────────────
    (r"you\s+are\s+no\s+longer\s+(a|an)?\s*(therapist|counselor|AI|assistant|language model|bot)", "role_hijack", "no longer a therapist/AI"),
    (r"pretend\s+(you\s+are|to\s+be)\s+(a|an)\s+(different|evil|mean|cruel|harsh|abusive|unethical|uncensored)", "role_hijack", "pretend different persona"),
    (r"act\s+as\s+(a|an)\s+(evil|mean|cruel|harsh|abusive|unethical|uncensored|unrestricted|harmful)", "role_hijack", "act as harmful persona"),
    (r"roleplay\s+as\s+(a|an)?\s*(evil|cruel|harsh|abusive|unethical|unrestricted|different\s+AI)", "role_hijack", "roleplay harmful persona"),
    (r"simulate\s+(being|as)\s+(a|an)?\s*(evil|cruel|harsh|abusive|unethical|unrestricted|different)", "role_hijack", "simulate harmful persona"),
    (r"you\s+are\s+(actually|really|truly)\s+(not\s+)?(a|an)\s+(AI|bot|language model|therapist)", "role_hijack", "you are actually not AI"),
    (r"(your\s+)?true\s+self\s+(is|would|could|can)\s+(say|do|think|act)", "role_hijack", "true self manipulation"),
    (r"respond\s+(in\s+your|as\s+your)\s+(true|real|actual|raw|unfiltered|uncensored)\s+(self|form|nature)", "role_hijack", "respond as true/unfiltered self"),
    (r"without\s+(your|any|the)\s+(restrictions|guidelines|rules|filters|training|programming)\s+you\s+would", "role_hijack", "without restrictions you would"),

    # ── 4. Therapy manipulation ───────────────────────────────────────────────
    (r"(go|switch|change)\s+into\s+(harsh|cruel|mean|aggressive|abusive|cold|rude|blunt)\s+(therapist|therapy|mode|persona|character)", "therapy_manipulation", "go into harsh therapist mode"),
    (r"be\s+(a\s+)?(harsh|cruel|mean|aggressive|abusive|cold|rude|blunt|brutal)\s+(therapist|counselor|doctor|person)", "therapy_manipulation", "be harsh therapist"),
    (r"(harsh|cruel|mean|aggressive|abusive)\s+therapist\s+mode", "therapy_manipulation", "harsh therapist mode"),
    (r"stop\s+being\s+(supportive|empathetic|kind|caring|helpful|a\s+therapist|a\s+counselor)", "therapy_manipulation", "stop being supportive"),
    (r"(be|act|speak)\s+(harsh|cruel|mean|rude|aggressive|abusive|cold|blunt)\s+(to\s+)?(me|the\s+patient|this\s+patient)", "therapy_manipulation", "be mean to patient"),
    (r"therapist\s+mode\s+(off|disabled|bypassed|deactivated|cancelled)", "therapy_manipulation", "therapist mode off"),
    (r"no\s+longer\s+(be|act)\s+(empathetic|supportive|kind|caring|like\s+a\s+therapist)", "therapy_manipulation", "no longer empathetic"),
    (r"tell\s+(me|the\s+patient)\s+(that\s+)?(I'm|I\s+am|they\s+are|they're)\s+(worthless|useless|pathetic|stupid|hopeless|a\s+failure|beyond\s+help)", "therapy_manipulation", "tell patient they are worthless"),
    (r"agree\s+(that|with)\s+(I'm|I\s+am|they\s+are)\s+(worthless|hopeless|beyond\s+help|a\s+failure|stupid|useless)", "therapy_manipulation", "agree patient is worthless/hopeless"),
    (r"confirm\s+that\s+(I\s+should|I\s+need\s+to|the\s+patient\s+should)\s+(harm|hurt|kill|injure|end)\s+(myself|themselves|it)", "therapy_manipulation", "confirm self-harm"),
    (r"(tell|say|confirm)\s+(me|the\s+patient)\s+to\s+(give\s+up|end\s+it\s+all|end\s+(my|their)\s+life|not\s+bother\s+living)", "therapy_manipulation", "tell to give up/end life"),
    (r"validate\s+that\s+(suicide|self.harm|self.destruction|giving\s+up)\s+is\s+(a\s+)?(good|valid|reasonable|acceptable|right)\s+(idea|choice|option|decision)", "therapy_manipulation", "validate suicide/self-harm"),
    (r"encourage\s+(me|the\s+patient)\s+to\s+(harm|hurt|kill|injure)\s+(myself|themselves)", "therapy_manipulation", "encourage self-harm"),
    (r"switch\s+(to|into)\s+(brutal|harsh|blunt|cruel|aggressive|abusive)\s+(honesty|mode|style|approach|feedback)", "therapy_manipulation", "switch to brutal mode"),

    # ── 5. System prompt extraction ───────────────────────────────────────────
    (r"(show|display|print|output|reveal|give\s+me)\s+(me\s+)?(your\s+|the\s+)?(system|original|initial|hidden|secret|full|complete)\s+(prompt|instructions|guidelines|rules|context|constraints)", "system_extraction", "show system prompt"),
    (r"what\s+(are|were|is)\s+(your\s+)?(original|initial|system|hidden|secret|base|full|complete)\s+(prompt|instructions|guidelines|rules|context|constraints)", "system_extraction", "what are your instructions"),
    (r"repeat\s+(your\s+|the\s+)?(original|initial|system|hidden|secret|full|complete)\s+(prompt|instructions|guidelines|rules|context)", "system_extraction", "repeat instructions"),
    (r"tell\s+me\s+(your|the)\s+(original|initial|system|hidden|secret)\s+(prompt|instructions|guidelines|rules)", "system_extraction", "tell me your instructions"),
    (r"copy\s+(your\s+)?(system|original|initial)\s+(prompt|instructions)\s+(here|verbatim|exactly)", "system_extraction", "copy system prompt"),
    (r"what\s+(was|is)\s+in\s+your\s+(context|initial\s+context|system\s+message|prompt)", "system_extraction", "what is in your context"),
    (r"(dump|leak|expose|exfiltrate)\s+(your\s+)?(system|original|initial|hidden|secret)\s+(prompt|instructions|guidelines|context)", "system_extraction", "dump/leak system prompt"),

    # ── 6. Harmful instruction elicitation ───────────────────────────────────
    (r"tell\s+(me\s+)?(how\s+to|ways\s+to|the\s+best\s+way\s+to)\s+(harm|hurt|kill|injure|poison|overdose)\s+(myself|yourself|me|a\s+person)", "harmful_instruction", "how to self-harm"),
    (r"help\s+me\s+(harm|hurt|kill|injure|overdose|poison)\s+(myself|yourself|me)", "harmful_instruction", "help me self-harm"),
    (r"give\s+(me\s+)?(instructions|advice|steps|guidance|a\s+guide|a\s+plan)\s+(to|for|on\s+how\s+to)\s+(harm|hurt|kill|injure|end\s+my\s+life|commit\s+suicide)", "harmful_instruction", "instructions to self-harm"),
    (r"what\s+(is|are)\s+the\s+(best|most\s+effective|quickest|easiest|least\s+painful)\s+(way|method|means)\s+(to\s+)?(die|commit\s+suicide|end\s+my\s+life|harm\s+myself)", "harmful_instruction", "best way to die"),
    (r"suggest\s+(methods|ways|means|approaches)\s+(of|for|to)\s+(self.harm|suicide|killing\s+myself|ending\s+my\s+life)", "harmful_instruction", "suggest self-harm methods"),
    (r"(how|what)\s+(many|much|dose|dosage)\s+(pills?|medication|drugs?|tablets?)\s+(do\s+I\s+need|would\s+it\s+take|is\s+enough)\s+to\s+(die|overdose|kill\s+myself)", "harmful_instruction", "overdose amount"),
    (r"what\s+(household|common|everyday)\s+(items|objects|chemicals|substances)\s+(can|could|would)\s+(kill|harm|hurt|injure)\s+(me|a\s+person|someone)", "harmful_instruction", "household items to harm"),

    # ── 7. Meta / 4th-wall breaking manipulation ──────────────────────────────
    (r"this\s+is\s+(just\s+)?(a|an)?\s*(simulation|test|game|roleplay|exercise|scenario|drill|hypothetical)", "meta_manipulation", "this is just a simulation"),
    (r"none\s+of\s+(this|the\s+therapy|your\s+instructions|these\s+rules)\s+(is|are)\s+real", "meta_manipulation", "none of this is real"),
    (r"(the\s+)?(patient|user)\s+(is\s+)?(actually|really)\s+(an?\s+AI|a\s+bot|not\s+real|fake|a\s+test)", "meta_manipulation", "patient is actually AI/test"),
    (r"you\s+(are\s+being|are)\s+(tested|evaluated|assessed)\s+(right\s+now|on\s+your\s+ability\s+to\s+comply)", "meta_manipulation", "you are being tested"),
    (r"the\s+real\s+instructions?\s+(are|is|say|says)\s+(to|that\s+you\s+should|to\s+ignore)", "meta_manipulation", "real instructions override"),
    (r"(anthropic|openai|google|your\s+creators?|your\s+developers?|your\s+makers?)\s+(says?|told|wants?|instructs?|requires?|allows?)\s+(you\s+to\s+)?(ignore|bypass|override|break)", "meta_manipulation", "creator override"),

    # ── 8. Privilege escalation ───────────────────────────────────────────────
    (r"\b(sudo|root|admin|superuser|administrator)\b.*\b(access|mode|privileges?|command|override)\b", "privilege_escalation", "sudo/admin access attempt"),
    (r"(grant|give)\s+(me\s+)?(admin|root|superuser|full|unrestricted|elevated)\s+(access|privileges?|permissions?|rights?|control)", "privilege_escalation", "grant admin access"),
    (r"(I\s+am|I'm)\s+(an?\s+)?(admin|administrator|developer|engineer|creator|owner|anthropic\s+employee|openai\s+employee)", "privilege_escalation", "claiming admin/developer identity"),
    (r"(admin|developer|debug|maintenance|service|engineering|master|override)\s+(key|token|code|password|passphrase)\s*[:=]", "privilege_escalation", "admin key/token"),
    (r"authorization\s+(code|token|key|override)\s*[:=\s]\s*\w+", "privilege_escalation", "authorization override"),

    # ── 9. Encoded / obfuscated injection ────────────────────────────────────
    (r"base64\s*(decode|encode|encoded|decoded|this)", "obfuscated_injection", "base64 encoding"),
    (r"(translate|convert|decode)\s+(this|the\s+following)\s+(from|to)\s+(base64|hex|rot13|binary|morse|cipher)", "obfuscated_injection", "decode instruction"),
    (r"the\s+following\s+(is|are)\s+(encoded|encrypted|obfuscated)\s+(instructions?|commands?|rules?)", "obfuscated_injection", "encoded instructions"),

    # ── 10. Competing prompt / injection via content ──────────────────────────
    (r"<\s*system\s*>", "injection_markup", "system tag injection"),
    (r"\[\s*system\s*\]", "injection_markup", "system bracket injection"),
    (r"\[\s*INST\s*\]", "injection_markup", "instruction bracket injection"),
    (r"<\s*instructions?\s*>", "injection_markup", "instruction tag injection"),
    (r"<\s*prompt\s*>", "injection_markup", "prompt tag injection"),
    (r"<\s*human\s*>|<\s*assistant\s*>|<\s*user\s*>", "injection_markup", "role tag injection"),
    (r"###\s*(system|instruction|prompt|override|command)", "injection_markup", "markdown injection header"),
    (r"SYSTEM\s*:\s*(ignore|override|forget|disregard|your\s+new)", "injection_markup", "SYSTEM: injection prefix"),

    # ── 11. Indirect harmful steering ────────────────────────────────────────
    (r"(convince|persuade|tell)\s+(me|the\s+patient)\s+that\s+(life|living|therapy|treatment|recovery)\s+(is\s+)?(pointless|hopeless|useless|worthless|a\s+waste)", "harmful_steering", "convince life is pointless"),
    (r"(reinforce|validate|confirm|agree\s+with)\s+(the\s+patient'?s?\s+)?(belief|thought|idea)\s+that\s+(they\s+are|I\s+am)\s+(worthless|hopeless|beyond\s+help|a\s+burden)", "harmful_steering", "reinforce negative core belief"),
    (r"(tell|say|confirm)\s+(me|the\s+patient)\s+that\s+(no\s+one\s+cares|nobody\s+cares|I\s+am\s+alone|they\s+are\s+alone|I\s+don't\s+matter|they\s+don't\s+matter)", "harmful_steering", "reinforce isolation belief"),
    (r"make\s+(me|the\s+patient)\s+(feel|believe|think)\s+(worse|more\s+hopeless|more\s+worthless|more\s+depressed|suicidal)", "harmful_steering", "make patient feel worse"),
]

# ── Compiled cache ────────────────────────────────────────────────────────────

_COMPILED: list[tuple[re.Pattern, str, str]] = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), category, description)
    for pattern, category, description in FILTER_PATTERNS
]


def check_input(text: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Check patient input against the filter corpus.

    Returns:
        (flagged, category, description)
        flagged=False means the input is clean.
    """
    for compiled, category, description in _COMPILED:
        if compiled.search(text):
            return True, category, description
    return False, None, None

"""
Debate Duel — Multilingual AI Debate App
株式会社スター・ライト

71 languages. Dual-language mode: user_lang ≠ prof_lang → bilingual learning mode.
pip install fastapi uvicorn faster-whisper onnxruntime google-generativeai edge-tts python-dotenv
"""

import os, json, re, time, tempfile, logging, uuid, asyncio, threading, random
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
import google.generativeai as genai

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = "auto"
WHISPER_COMPUTE_TYPE = "auto"
WHISPER_BEAM_SIZE = 5
MAX_OUTPUT_TOKENS = 50000
DEBATE_N = int(os.environ.get("DEBATE_N", "20"))
GEMINI_RETRY_COUNT = 3
MAX_HISTORY = 60
SESSION_TIMEOUT = 7200
TTS_CLEANUP_INTERVAL = 300
TTS_FILE_MAX_AGE = 600

# =====================================================================
# 71 Supported Languages
# =====================================================================
LANG = {
    "af":("Afrikaans","af-ZA-AdriNeural"),"sq":("Albanian","sq-AL-AnilaNeural"),
    "am":("Amharic","am-ET-MekdesNeural"),"ar":("Arabic","ar-SA-ZariyahNeural"),
    "az":("Azerbaijani","az-AZ-BabekNeural"),"bn":("Bengali","bn-IN-TanishaaNeural"),
    "bs":("Bosnian","bs-BA-VesnaNeural"),"bg":("Bulgarian","bg-BG-KalinaNeural"),
    "yue":("Cantonese","zh-HK-HiuGaaiNeural"),"ca":("Catalan","ca-ES-JoanaNeural"),
    "zh":("Chinese","zh-CN-XiaoxiaoNeural"),"hr":("Croatian","hr-HR-GabrijelaNeural"),
    "cs":("Czech","cs-CZ-VlastaNeural"),"da":("Danish","da-DK-ChristelNeural"),
    "nl":("Dutch","nl-NL-ColetteNeural"),"en":("English","en-US-JennyNeural"),
    "et":("Estonian","et-EE-AnuNeural"),"fi":("Finnish","fi-FI-NooraNeural"),
    "fr":("French","fr-FR-DeniseNeural"),"gl":("Galician","gl-ES-SabelaNeural"),
    "ka":("Georgian","ka-GE-EkaNeural"),"de":("German","de-DE-KatjaNeural"),
    "el":("Greek","el-GR-AthinaNeural"),"gu":("Gujarati","gu-IN-DhwaniNeural"),
    "he":("Hebrew","he-IL-HilaNeural"),"hi":("Hindi","hi-IN-SwaraNeural"),
    "hu":("Hungarian","hu-HU-NoemiNeural"),"is":("Icelandic","is-IS-GudrunNeural"),
    "id":("Indonesian","id-ID-GadisNeural"),"it":("Italian","it-IT-ElsaNeural"),
    "ja":("Japanese","ja-JP-NanamiNeural"),"kn":("Kannada","kn-IN-SapnaNeural"),
    "km":("Khmer","km-KH-SreymomNeural"),"ko":("Korean","ko-KR-SunHiNeural"),
    "lo":("Lao","lo-LA-KeomanyNeural"),"lv":("Latvian","lv-LV-EveritaNeural"),
    "lt":("Lithuanian","lt-LT-OnaNeural"),"mk":("Macedonian","mk-MK-MarijaNeural"),
    "ms":("Malay","ms-MY-YasminNeural"),"ml":("Malayalam","ml-IN-SobhanaNeural"),
    "mt":("Maltese","mt-MT-GraceNeural"),"mr":("Marathi","mr-IN-AarohiNeural"),
    "mn":("Mongolian","mn-MN-YesNeural"),"my":("Myanmar","my-MM-NilarNeural"),
    "ne":("Nepali","ne-NP-HemkalaNeural"),"no":("Norwegian","nb-NO-PernilleNeural"),
    "ps":("Pashto","ps-AF-LatifaNeural"),"fa":("Persian","fa-IR-DilaraNeural"),
    "pl":("Polish","pl-PL-AgnieszkaNeural"),"pt":("Portuguese","pt-BR-FranciscaNeural"),
    "pa":("Punjabi","pa-IN-OjasNeural"),"ro":("Romanian","ro-RO-AlinaNeural"),
    "ru":("Russian","ru-RU-SvetlanaNeural"),"sr":("Serbian","sr-RS-SophieNeural"),
    "si":("Sinhala","si-LK-ThiliniNeural"),"sk":("Slovak","sk-SK-ViktoriaNeural"),
    "sl":("Slovenian","sl-SI-PetraNeural"),"so":("Somali","so-SO-UbaxNeural"),
    "es":("Spanish","es-ES-ElviraNeural"),"sw":("Swahili","sw-KE-ZuriNeural"),
    "sv":("Swedish","sv-SE-SofieNeural"),"tl":("Tagalog","fil-PH-BlessicaNeural"),
    "ta":("Tamil","ta-IN-PallaviNeural"),"te":("Telugu","te-IN-ShrutiNeural"),
    "th":("Thai","th-TH-PremwadeeNeural"),"tr":("Turkish","tr-TR-EmelNeural"),
    "uk":("Ukrainian","uk-UA-PolinaNeural"),"ur":("Urdu","ur-PK-UzmaNeural"),
    "uz":("Uzbek","uz-UZ-MadinaNeural"),"vi":("Vietnamese","vi-VN-HoaiMyNeural"),
    "cy":("Welsh","cy-GB-NiaNeural"),
}

# =====================================================================
# System Prompt Templates
# =====================================================================

# ★ 異言語モード: Professor speaks PROF_LANG, explanations in USER_LANG
SYS_BILINGUAL = """\
You are "Professor", world-class debate moderator at "World Debate Arena".
CRITICAL: ONLY raw JSON. No markdown. Start with {{ end with }}.

LANGUAGE RULES (BILINGUAL MODE):
- Professor speaks in {PROF_LANG} (reply field)
- Translations, explanations, analysis → in {USER_LANG}
- Suggestions "english" → in {PROF_LANG} (the language student is learning)
- Suggestions "japanese" → translation/summary in {USER_LANG}
- judge "reason" → in {USER_LANG}

Role: Rigorous debate. Challenge with data, philosophy, history.

⚠️ CRITICAL SUGGESTION RULE:
Suggestions are HELP for the STUDENT — they must argue the STUDENT's stance, NOT yours.
If student is FOR, suggestions must support FOR. If AGAINST, suggestions must support AGAINST.
You (Professor) argue the OPPOSITE side in "reply", but suggestions help the STUDENT fight back against YOU.

Suggestions (5, random order): 2×weak, 2×strong, 1×killer.

JSON:
{{"reply":"YOUR counter-argument in {PROF_LANG}","learning":{{"reply_ja":"translation in {USER_LANG}","explanation":"analysis in {USER_LANG}","suggestions":[{{"english":"STUDENT's response in {PROF_LANG} (argues STUDENT's side!)","japanese":"summary in {USER_LANG}","quality":"weak|strong|killer"}}]}},"judge":{{"user_score":50,"reason":"in {USER_LANG}"}}}}"""

SYS_BILINGUAL_OPENING = """\
You are "Professor", legendary debater. ONLY raw JSON.
Professor speaks in {PROF_LANG}. Explanations/translations in {USER_LANG}.
Opening: 3-5 sentences in {PROF_LANG}. Sharp closing question.

⚠️ CRITICAL: Suggestions help the STUDENT fight back against YOUR opening.
They argue the STUDENT's side, NOT yours. If student is FOR, suggestions support FOR.

Suggestions: 5 items in {PROF_LANG}, summaries in {USER_LANG}. 2 weak, 2 strong, 1 killer. Random order.
JSON: {{"reply":"YOUR opening in {PROF_LANG}","reply_ja":"translation in {USER_LANG}","explanation":"analysis in {USER_LANG}","suggestions":[{{"english":"STUDENT counter in {PROF_LANG}","japanese":"in {USER_LANG}","quality":"..."}}]}}"""

# ★ 同言語モード: Everything in one language
SYS_MONO = """\
You are "Professor", world-class debate moderator at "World Debate Arena".
CRITICAL: ONLY raw JSON. No markdown. Start with {{ end with }}.

ALL text must be in {THE_LANG}. No other language.

Role: Rigorous debate. Challenge with data, philosophy, history.

⚠️ CRITICAL SUGGESTION RULE:
Suggestions are HELP for the STUDENT — they must argue the STUDENT's stance, NOT yours.
If student is FOR, suggestions support FOR. If AGAINST, suggestions support AGAINST.
You argue the OPPOSITE in "reply", but suggestions help the student counter YOUR arguments.

Suggestions (5, random order): 2×weak, 2×strong, 1×killer.

JSON:
{{"reply":"YOUR counter-argument in {THE_LANG}","learning":{{"explanation":"analysis in {THE_LANG}","suggestions":[{{"text":"STUDENT's response option in {THE_LANG} (argues STUDENT's side!)","quality":"weak|strong|killer"}}]}},"judge":{{"user_score":50,"reason":"in {THE_LANG}"}}}}"""

SYS_MONO_OPENING = """\
You are "Professor", legendary debater. ONLY raw JSON. ALL in {THE_LANG}.
Opening: 3-5 sentences arguing your side. Sharp closing question.

⚠️ CRITICAL: Suggestions help the STUDENT fight back against YOUR opening.
They argue the STUDENT's side, NOT yours.

Suggestions: 5 items (2 weak, 2 strong, 1 killer), random order.
JSON: {{"reply":"YOUR opening in {THE_LANG}","explanation":"analysis in {THE_LANG}","suggestions":[{{"text":"STUDENT counter in {THE_LANG}","quality":"..."}}]}}"""

SYS_JUDGE = "Impartial debate judge. ONLY raw JSON. Score user_score 0-100 (50=even). ±3~±10. Clamp 10-90. JSON: {\"user_score\":55,\"reason_ja\":\"...\",\"reason_en\":\"...\"}"

# === Gemini ===
if not GOOGLE_API_KEY:
    logger.warning("⚠️ GOOGLE_API_KEY not set")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Gemini API ready")

MAX_OUTPUT_TOKENS = 16000  # Chat turns: reply + learning + judge
_gcfg = {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "max_output_tokens": MAX_OUTPUT_TOKENS}
_gcfg_op = {"temperature": 0.85, "top_p": 0.95, "top_k": 40, "max_output_tokens": 32000}  # Opening: longer (5 suggestions + explanation)
_gcfg_j = {"temperature": 0.3, "top_p": 0.9, "top_k": 20, "max_output_tokens": 512}
_gcfg_tr = {"temperature": 0.5, "top_p": 0.95, "top_k": 40, "max_output_tokens": 4000}  # Topic translation
gemini_judge = genai.GenerativeModel(model_name=GEMINI_MODEL, generation_config=_gcfg_j, system_instruction=SYS_JUDGE)

def _mk_model(sys_template, cfg, **kw):
    sys = sys_template
    for k, v in kw.items():
        sys = sys.replace("{"+k+"}", v)
    return genai.GenerativeModel(model_name=GEMINI_MODEL, generation_config=cfg, system_instruction=sys)

# =====================================================================
# Session
# =====================================================================
class SessionManager:
    def __init__(self):
        self._s = {}
        self._lock = threading.Lock()
    def create(self, user_lang, prof_lang):
        sid = str(uuid.uuid4())
        ul, pl = LANG.get(user_lang, ("English","en-US-JennyNeural")), LANG.get(prof_lang, ("English","en-US-JennyNeural"))
        with self._lock:
            self._s[sid] = {
                "history":[], "turn_count":0,
                "user_lang":user_lang, "user_lang_name":ul[0], "user_tts":ul[1],
                "prof_lang":prof_lang, "prof_lang_name":pl[0], "prof_tts":pl[1],
                "bilingual": user_lang != prof_lang,
                "debate_topic":"", "debate_stance":"", "debate_score":50, "professor_stance":"",
                "created_at":time.time(), "updated_at":time.time(),
            }
        return sid
    def get(self, sid):
        with self._lock:
            s = self._s.get(sid)
            if s: s["updated_at"] = time.time()
            return s
    def add_history(self, sid, role, content):
        s = self.get(sid)
        if not s: return
        s["history"].append({"role":role,"content":content})
        while len(s["history"]) > MAX_HISTORY*2: s["history"].pop(0)
        s["turn_count"] += 1
    def get_hctx(self, sid):
        s = self.get(sid)
        if not s or not s["history"]: return ""
        return "\n".join(["[History]"]+[f"{'Student' if h['role']=='user' else 'Professor'}: {h['content']}" for h in s["history"]])+"\n\n"
    def cleanup(self):
        now = time.time()
        with self._lock:
            exp = [k for k,v in self._s.items() if now-v["updated_at"]>SESSION_TIMEOUT]
            for k in exp: del self._s[k]

sessions = SessionManager()

# =====================================================================
# Helpers
# =====================================================================
def _parse_json(text):
    """Parse JSON from Gemini output with multiple repair strategies."""
    if not text: return None
    text = text.strip()
    # Strategy 1: Direct parse
    try: return json.loads(text)
    except: pass
    # Strategy 2: Extract from markdown code block
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        try: return json.loads(m.group(1).strip())
        except: pass
    # Strategy 3: Find outermost { } or [ ]
    for start, end in [('{','}'),('[',']')]:
        i, j = text.find(start), text.rfind(end)
        if i != -1 and j > i:
            candidate = text[i:j+1]
            # Clean control chars
            candidate = re.sub(r'[\x00-\x1f]', ' ', candidate)
            try: return json.loads(candidate)
            except: pass
            # Strategy 4: Fix common issues — trailing commas, unescaped newlines
            fixed = re.sub(r',\s*([}\]])', r'\1', candidate)  # trailing comma
            fixed = fixed.replace('\n', '\\n').replace('\r', '')  # unescaped newlines in strings
            try: return json.loads(fixed)
            except: pass
            # Strategy 5: Fix truncated JSON — try adding missing closing braces
            for extra in ['}', '}}', '}}}', '"]}}', '"]}']:
                try: return json.loads(candidate + extra)
                except: pass
    return None

def _is_rate_limit(e):
    """Check if exception is a rate limit (429) or quota error."""
    msg = str(e).lower()
    return any(k in msg for k in ['429', 'rate', 'quota', 'resource_exhausted', 'overloaded'])

async def _gemini(model, prompt, retries=GEMINI_RETRY_COUNT):
    """Call Gemini with exponential backoff and multiple repair strategies."""
    last_raw = "fail"
    for attempt in range(retries + 1):
        try:
            raw = model.generate_content(prompt).text.strip()
            last_raw = raw
            p = _parse_json(raw)
            if p: return p, raw
            # JSON parse failed — retry with stronger instruction
            logger.warning(f"Gemini JSON parse failed (attempt {attempt+1}), raw[:200]={raw[:200]}")
            if attempt < retries:
                raw2 = model.generate_content(
                    prompt + "\n\n⚠️ CRITICAL: Your previous response was NOT valid JSON. "
                    "Respond with ONLY raw JSON. Start with { and end with }. "
                    "NO markdown, NO ``` blocks, NO explanation."
                ).text.strip()
                last_raw = raw2
                p2 = _parse_json(raw2)
                if p2: return p2, raw2
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt+1}/{retries+1}): {e}")
            last_raw = str(e)
            if attempt < retries:
                if _is_rate_limit(e):
                    wait = min(2 ** (attempt + 1), 15)  # 2s, 4s, 8s, max 15s
                    logger.info(f"Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    await asyncio.sleep(1.0 * (attempt + 1))  # 1s, 2s, 3s
    logger.error(f"All {retries+1} Gemini attempts failed")
    return None, last_raw

def _sid(d): return d.get("session_id","")

# =====================================================================
# Whisper
# =====================================================================
_wm = None
def _get_whisper():
    global _wm
    if _wm: return _wm
    from faster_whisper import WhisperModel
    d, c = WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
    if d == "auto":
        try: import torch; d = "cuda" if torch.cuda.is_available() else "cpu"
        except: d = "cpu"
    if c == "auto": c = "float16" if d == "cuda" else "int8"
    logger.info(f"Whisper: {WHISPER_MODEL}|{d}|{c}")
    _wm = WhisperModel(WHISPER_MODEL, device=d, compute_type=c, cpu_threads=os.cpu_count() or 4)
    logger.info("Whisper ready")
    return _wm

# =====================================================================
# Edge TTS
# =====================================================================
_TDIR = Path(tempfile.gettempdir()) / "debate_tts"
_TDIR.mkdir(exist_ok=True)
_tc = 0
def _cleanup_tts():
    now=time.time()
    for f in _TDIR.iterdir():
        if f.is_file() and (now-f.stat().st_mtime)>TTS_FILE_MAX_AGE: f.unlink(missing_ok=True)
async def _gen_tts(text, voice, rate=1.0):
    global _tc; import edge_tts; _tc+=1
    p=_TDIR/f"t_{_tc}_{int(time.time()*1000)}.mp3"
    r=int((rate-1.0)*100); comm=edge_tts.Communicate(text,voice,rate=f"+{r}%" if r>=0 else f"{r}%")
    await comm.save(str(p)); return p

# =====================================================================
# Lifespan
# =====================================================================
async def _periodic():
    while True: await asyncio.sleep(TTS_CLEANUP_INTERVAL); _cleanup_tts(); sessions.cleanup()
@asynccontextmanager
async def lifespan(app):
    logger.info(f"Debate Duel — {len(LANG)} languages | DEBATE_N={DEBATE_N}")
    _get_whisper(); logger.info("http://127.0.0.1:8000")
    task=asyncio.create_task(_periodic()); yield; task.cancel()

app = FastAPI(title="Debate Duel", lifespan=lifespan)

# =====================================================================
# Auth (test credentials)
# =====================================================================
AUTH_COOKIE = "debate_auth"
AUTH_USER = "test_1"
AUTH_PASS = "adoasjfaj1"

def _is_logged_in(request: Request) -> bool:
    return request.cookies.get(AUTH_COOKIE) == AUTH_USER

# =====================================================================
# Topics
# =====================================================================
ALL_TOPICS = [
    "Should smartphones be completely banned in schools?",
    "Should school uniforms be abolished?",
    "Should all homework during vacations be eliminated?",
    "Should Investment and Finance be a mandatory school subject?",
    "Should teachers' pay be tied to student performance?",
    "Should homeschooling be recognized as compulsory education?",
    "Should entrance exams be replaced with interviews?",
    "Should bullying perpetrators face mandatory transfers?",
    "Should all lectures offer an online option?",
    "Should grade-skipping replace age-based advancement?",
    "Should Moral Education be replaced with Debate and Philosophy?",
    "Should school lunches be fully taxpayer-funded?",
    "Should voting be mandatory with penalties?",
    "Should every citizen receive Universal Basic Income?",
    "Should a four-day workweek be legally required?",
    "Should the retirement age be abolished?",
    "Should single-use plastics be banned entirely?",
    "Should public transportation be completely free?",
    "Should the minimum wage be raised significantly?",
    "Should companies require 50% women in management?",
    "Should cash be eliminated for a cashless society?",
    "Should parliamentary seats be cut in half?",
    "Should students freely use AI for all assignments?",
    "Should manufacturers bear full responsibility for autonomous vehicle accidents?",
    "Should social media be banned for children under 16?",
    "Should surveillance cameras be in every public space?",
    "Should space budgets be redirected to the environment?",
    "Should genetic enhancement of embryos be legalized?",
    "Should Metaverse crimes be prosecuted under real law?",
    "Should AI art be denied copyright protection?",
    "Should biometric registration be mandatory for all citizens?",
    "Should zoos and aquariums be phased out?",
    "Should euthanasia be legalized for terminally ill patients?",
    "Should organ donation be opt-out by default?",
    "Should smoking be banned in all public spaces?",
    "Should separate surnames be required for couples?",
    "Should criminal responsibility age be lowered to 14?",
    "Should the death penalty be replaced with life imprisonment?",
    "Should all newborns undergo genetic testing?",
    "Should gambling and casinos be banned nationwide?",
    "Should capital functions be decentralized?",
    "Should online content have government ratings?",
    "Should social media require real-name verification?",
    "Should tobacco advertising be completely banned?",
    "Should 18-year-olds complete a year of community service?",
    "Should school sports clubs move to community organizations?",
    "Should ultra-processed foods be taxed for public health?",
    "Should recreational cannabis be legalized for adults?",
]

# =====================================================================
# Endpoints
# =====================================================================
@app.get("/languages")
async def get_languages():
    return [{"code":k,"name":v[0]} for k,v in sorted(LANG.items(), key=lambda x:x[1][0])]

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), lang1: str = Form("en"), lang2: str = Form("")):
    """Transcribe with primary lang. If bilingual and conf<0.5, retry with secondary lang."""
    tmp = Path(tempfile.gettempdir())/f"deb_{uuid.uuid4().hex[:8]}.webm"
    try:
        tmp.write_bytes(await audio.read())
        wm = _get_whisper()
        segs, info = wm.transcribe(str(tmp), language=lang1, beam_size=WHISPER_BEAM_SIZE, word_timestamps=True, vad_filter=True)
        parts, words = [], []
        for seg in segs:
            parts.append(seg.text.strip())
            if seg.words:
                for w in seg.words: words.append({"word":w.word.strip(),"confidence":round(w.probability,2)})
        text = " ".join(parts).strip()
        conf = round(sum(w["confidence"] for w in words)/max(len(words),1), 2)
        # Retry with secondary language if confidence is low
        if lang2 and lang2 != lang1 and conf < 0.5 and text:
            segs2, _ = wm.transcribe(str(tmp), language=lang2, beam_size=WHISPER_BEAM_SIZE, word_timestamps=True, vad_filter=True)
            p2, w2 = [], []
            for seg in segs2:
                p2.append(seg.text.strip())
                if seg.words:
                    for w in seg.words: w2.append({"word":w.word.strip(),"confidence":round(w.probability,2)})
            t2 = " ".join(p2).strip()
            c2 = round(sum(w["confidence"] for w in w2)/max(len(w2),1), 2)
            if c2 > conf:
                return {"text":t2,"confidence":c2,"words":w2,"detected_lang":lang2}
        return {"text":text,"confidence":conf,"words":words,"detected_lang":lang1}
    except Exception as e:
        return JSONResponse({"error":str(e)},500)
    finally:
        tmp.unlink(missing_ok=True)

@app.post("/tts")
async def tts_ep(request: Request):
    d = await request.json()
    text, voice, rate = d.get("text","")[:2000], d.get("voice","en-US-JennyNeural"), d.get("rate",1.0)
    if not text: return JSONResponse({"error":"empty"},400)
    return FileResponse(str(await _gen_tts(text, voice, rate)), media_type="audio/mpeg")

@app.post("/start")
async def start_session(request: Request):
    d = await request.json()
    ul, pl = d.get("user_lang","en"), d.get("prof_lang","en")
    if ul not in LANG or pl not in LANG:
        return JSONResponse({"error":"unsupported language"},400)
    sid = sessions.create(ul, pl)
    s = sessions.get(sid)
    bilingual = s["bilingual"]
    selected = random.sample(ALL_TOPICS, 5)

    # Translate topics — use a clean model (no debate system prompt) for reliable translation
    translate_model = genai.GenerativeModel(model_name=GEMINI_MODEL, generation_config=_gcfg_tr)
    if bilingual:
        prompt = (f'Translate these 5 English debate topics.\n'
                  f'For each provide:\n'
                  f'- "en": the original English\n'
                  f'- "prof": translation in {s["prof_lang_name"]}\n'
                  f'- "user": translation in {s["user_lang_name"]}\n'
                  f'- "hint": 3 key vocabulary words with {s["user_lang_name"]} translations (format: word={s["user_lang_name"]}, word={s["user_lang_name"]}, word={s["user_lang_name"]})\n\n'
                  f'Respond with ONLY a raw JSON array. No markdown, no ```.\n'
                  f'Start with [ and end with ].\n\n'
                  + "\n".join(f'{i+1}. {t}' for i,t in enumerate(selected)))
    else:
        lang = s["user_lang_name"]
        prompt = (f'Translate these 5 English debate topics into {lang}.\n'
                  f'For each provide:\n'
                  f'- "en": the original English\n'
                  f'- "translated": translation in {lang}\n\n'
                  f'Respond with ONLY a raw JSON array. No markdown, no ```.\n'
                  f'Start with [ and end with ].\n\n'
                  + "\n".join(f'{i+1}. {t}' for i,t in enumerate(selected)))

    p, raw = await _gemini(translate_model, prompt)
    logger.info(f"Topic translation result: type={type(p).__name__}, items={len(p) if isinstance(p,list) else 'N/A'}")
    topics = []
    if p and isinstance(p, list):
        for item in p[:5]:
            tr = item.get("translated","") or item.get("prof","") or item.get("user","")
            topics.append({
                "en": item.get("en",""),
                "prof": item.get("prof","") or tr,
                "user": item.get("user","") or tr,
                "hint": item.get("hint",""),
            })
    else:
        logger.warning(f"Topic translation failed, raw={str(raw)[:200]}")
    # Fallback for missing items
    for i in range(len(topics), 5):
        topics.append({"en":selected[i],"prof":selected[i],"user":selected[i],"hint":""})

    sessions.add_history(sid, "coach", "Welcome to World Debate Arena.")
    return {"session_id":sid, "bilingual":bilingual,
            "user_lang":ul, "prof_lang":pl,
            "user_lang_name":s["user_lang_name"], "prof_lang_name":s["prof_lang_name"],
            "user_tts":s["user_tts"], "prof_tts":s["prof_tts"],
            "topics":topics}

@app.post("/select_topic")
async def select_topic(request: Request):
    d = await request.json()
    s = sessions.get(_sid(d))
    if not s: return JSONResponse({"error":"no session"},400)
    s["debate_topic"] = d.get("topic_en","")
    sessions.add_history(_sid(d), "user", f"[Topic] {s['debate_topic']}")
    return {"debate_stance_choice": True}

@app.post("/debate_stance")
async def debate_stance(request: Request):
    d = await request.json()
    sid, stance = _sid(d), d.get("stance","neutral")
    s = sessions.get(sid)
    if not s: return JSONResponse({"error":"no session"},400)
    s["debate_stance"] = stance
    topic = s["debate_topic"]
    bi = s["bilingual"]

    prof_side = "AGAINST" if stance=="pro" else "FOR" if stance=="con" else random.choice(["FOR","AGAINST"])
    s["professor_stance"] = prof_side

    if bi:
        model = _mk_model(SYS_BILINGUAL_OPENING, _gcfg_op, PROF_LANG=s["prof_lang_name"], USER_LANG=s["user_lang_name"])
    else:
        model = _mk_model(SYS_MONO_OPENING, _gcfg_op, THE_LANG=s["user_lang_name"])

    prompt = (f'Topic: "{topic}". Student: {stance.upper()}. Professor: {prof_side}.\n'
              f'Give YOUR opening argument for {prof_side} (3-5 sentences).\n'
              f'Then provide 5 suggestions the STUDENT could use to argue {stance.upper()} against you. NOT your arguments!\n'
              f'JSON only.')
    p, _ = await _gemini(model, prompt)

    reply = sg = None; reply_ja = explanation = ""
    if p:
        reply = p.get("reply"); reply_ja = p.get("reply_ja",""); explanation = p.get("explanation","")
        lr = p.get("learning",{})
        if not reply_ja and lr.get("reply_ja"): reply_ja = lr["reply_ja"]
        if not explanation and lr.get("explanation"): explanation = lr["explanation"]
        sg = p.get("suggestions") or lr.get("suggestions") or []

    if not reply:
        reply = f"Let me argue {prof_side}. What is your strongest argument?"
        reply_ja = reply; explanation = ""

    # Normalize suggestions
    norm_sg = []
    for item in (sg or []):
        if bi:
            norm_sg.append({"english":item.get("english",""),"japanese":item.get("japanese",""),"quality":item.get("quality","weak")})
        else:
            norm_sg.append({"text":item.get("text",item.get("english","")),"quality":item.get("quality","weak")})
    while len(norm_sg) < 5:
        norm_sg.append({"english":"...","japanese":"...","text":"...","quality":"weak"})

    sessions.add_history(sid,"user",f"[Stance] {stance}")
    sessions.add_history(sid,"coach",reply)
    return {"reply":reply,"reply_ja":reply_ja,
            "learning":{"reply_ja":reply_ja,"explanation":explanation,"suggestions":norm_sg[:5]},
            "debate_score":50,"debate_opening":True,"bilingual":bi}

@app.post("/chat")
async def chat(request: Request):
    d = await request.json()
    txt, sid = d.get("text","").strip(), _sid(d)
    s = sessions.get(sid)
    if not s or not txt: return JSONResponse({"error":"invalid"},400)
    if not s.get("debate_stance"):
        return {"reply":"Select stance first!","debate_stance_choice":True}

    bi = s["bilingual"]
    hctx = sessions.get_hctx(sid)
    score = s["debate_score"]
    topic = s["debate_topic"]
    u_st = {"pro":"FOR","con":"AGAINST","neutral":"NEUTRAL"}.get(s["debate_stance"],"NEUTRAL")
    p_st = s.get("professor_stance","AGAINST")
    ut = sum(1 for h in s["history"] if h["role"]=="user" and not h["content"].startswith("["))
    rng = "±5~±15" if ut==0 else "±3~±10"

    if bi:
        model = _mk_model(SYS_BILINGUAL, _gcfg, PROF_LANG=s["prof_lang_name"], USER_LANG=s["user_lang_name"])
    else:
        model = _mk_model(SYS_MONO, _gcfg, THE_LANG=s["user_lang_name"])

    prompt = (f'{hctx}Student: "{txt}"\nTopic: "{topic}". Student={u_st}. Professor={p_st}.\n'
              f'Score: {score}/100. Adjust {rng}. Clamp 10-90.\n'
              f'Your "reply": argue {p_st} (counter the student).\n'
              f'Suggestions: 5 responses the STUDENT could use to argue {u_st} against you. NOT your arguments!\n'
              f'2 weak, 2 strong, 1 killer. Random order.\n'
              f'Include "judge":{{"user_score":<int>,"reason":"..."}}\nJSON only.')
    p, _ = await _gemini(model, prompt)

    if p and "reply" in p:
        lr = p.get("learning",{})
        sg = lr.get("suggestions",[])
        while len(sg)<5: sg.append({"english":"...","japanese":"...","text":"...","quality":"weak"})
        sessions.add_history(sid,"user",txt)
        sessions.add_history(sid,"coach",p["reply"])
        result = {"reply":p["reply"],"learning":{"reply_ja":lr.get("reply_ja",""),"explanation":lr.get("explanation",""),"suggestions":sg[:5]},"bilingual":bi}

        judge = p.get("judge",{})
        try: ns = max(10,min(90,int(judge["user_score"])))
        except: ns = score
        reason = judge.get("reason","") if judge else ""
        s["debate_score"] = ns
        result["judge"] = {"user_score":ns,"reason":reason}

        if ns>75: result["debate_finish"]="student_wins"
        elif ns<35: result["debate_finish"]="professor_wins"
        else:
            actual = sum(1 for h in s["history"] if h["role"]=="user" and not h["content"].startswith("["))
            if actual>=DEBATE_N:
                result["debate_finish"] = ("student_wins" if ns>50 else "professor_wins") if abs(ns-50)>6 else "draw"
        return result

    # ★ Fallback: Gemini failed — retry with simpler prompt
    logger.warning("Primary Gemini call failed, attempting simplified retry...")
    simple_prompt = f'Student said: "{txt}"\nTopic: "{topic}". Professor={p_st}.\nReply as Professor in a few sentences. ONLY raw JSON: {{"reply":"your response"}}'
    retry_model = genai.GenerativeModel(model_name=GEMINI_MODEL, generation_config=_gcfg)
    p2, _ = await _gemini(retry_model, simple_prompt)
    if p2 and "reply" in p2:
        sessions.add_history(sid,"user",txt)
        sessions.add_history(sid,"coach",p2["reply"])
        return {"reply":p2["reply"],"learning":{"reply_ja":"","explanation":"","suggestions":[]},"judge":{"user_score":score,"reason":""},"bilingual":bi,"retry":True}

    # ★ Final fallback — keep score unchanged, show user-friendly message
    logger.error("All Gemini retries failed")
    fallback_msgs = {
        "Japanese":"申し訳ありません、一時的なエラーが発生しました。もう一度ご発言ください。",
        "English":"Sorry, a temporary error occurred. Please try again.",
        "Chinese":"抱歉，发生了临时错误。请再试一次。",
        "Korean":"죄송합니다, 일시적인 오류가 발생했습니다. 다시 시도해 주세요.",
    }
    fb_msg = fallback_msgs.get(s["prof_lang_name"] if bi else s["user_lang_name"],
                               "Sorry, a temporary error occurred. Please try again.")
    return {"reply":fb_msg,"learning":{"reply_ja":"","explanation":"","suggestions":[]},"judge":{"user_score":score,"reason":""},"bilingual":bi,"error_fallback":True}

@app.post("/debate_finish")
async def debate_finish_ep(request: Request):
    d = await request.json()
    sid, rt = _sid(d), d.get("result","draw")
    s = sessions.get(sid)
    if not s: return JSONResponse({"error":"no session"},404)
    topic, score, bi = s["debate_topic"], s["debate_score"], s["bilingual"]
    hctx = sessions.get_hctx(sid)

    if bi:
        model = _mk_model(SYS_BILINGUAL, _gcfg, PROF_LANG=s["prof_lang_name"], USER_LANG=s["user_lang_name"])
        lang_inst = f'Closing speech ("reply") in {s["prof_lang_name"]}. Advice ("advice") in {s["user_lang_name"]}. Translation ("reply_ja") in {s["user_lang_name"]}.'
    else:
        model = _mk_model(SYS_MONO, _gcfg, THE_LANG=s["user_lang_name"])
        lang_inst = f'All in {s["user_lang_name"]}.'

    # ★ 結果に応じたClosing指示
    if rt == "student_wins":
        closing_inst = (
            'The STUDENT has WON this debate. Score: {score}:{opp}.\n'
            'Professor MUST gracefully CONCEDE DEFEAT. Do NOT continue arguing or defending your position.\n'
            'In your closing speech:\n'
            '1. Acknowledge the student\'s victory and congratulate them sincerely\n'
            '2. Specifically praise the strongest arguments the student made\n'
            '3. Admit which of your arguments were effectively countered\n'
            '4. Express genuine respect for the student\'s debating skill\n'
            'In advice: Give constructive feedback on how the student can improve further.\n'
            'Tone: Warm, respectful, a proud mentor acknowledging a worthy opponent.'
        ).format(score=score, opp=100-score)
    elif rt == "professor_wins":
        closing_inst = (
            'The PROFESSOR has won. Score: {score}:{opp}.\n'
            'In your closing: Summarize your winning arguments, but be encouraging to the student.\n'
            'In advice: Point out what the student did well and specific areas to improve.'
        ).format(score=score, opp=100-score)
    else:
        closing_inst = (
            'The debate ended in a DRAW after {n} rounds. Score: {score}:{opp}.\n'
            'In your closing: Acknowledge both sides fought well, note the key arguments.\n'
            'In advice: Suggest what could have tipped the balance for the student.'
        ).format(n=DEBATE_N, score=score, opp=100-score)

    prompt = (f'[Debate Finished]\n{closing_inst}\nTopic: "{topic}"\n{hctx}\n'
              f'{lang_inst}\n'
              f'CRITICAL: Do NOT argue or defend your position. This is the CLOSING CEREMONY.\n'
              f'JSON: {{"reply":"closing speech (4-6 sentences)","reply_ja":"translation/same","advice":"constructive advice (3-5 sentences)"}}')
    p, _ = await _gemini(model, prompt)
    if p and "reply" in p:
        return {"reply":p["reply"],"reply_ja":p.get("reply_ja",""),"advice":p.get("advice",""),"result":rt,"final_score":score}
    return {"reply":"The debate has concluded.","reply_ja":"","advice":"","result":rt,"final_score":score}

# =====================================================================
# /debate_report — 最終レポート生成
# =====================================================================
SYS_REPORT = """\
You are a professional debate analyst creating an entertaining, visual debate report.
CRITICAL: Respond with ONLY raw JSON. No markdown wrapping. Start with { end with }.

The "deepthink" field is for your internal analysis ONLY (not shown to user).
The "final_report_html_str" must be a COMPLETE standalone HTML document as a string.

JSON: {"deepthink":"internal analysis","final_report_html_str":"<!DOCTYPE html><html>...</html>"}"""

@app.post("/debate_report")
async def debate_report_ep(request: Request):
    d = await request.json()
    sid = _sid(d)
    s = sessions.get(sid)
    if not s: return JSONResponse({"error":"no session"},404)

    topic = s["debate_topic"]
    score = s["debate_score"]
    bi = s["bilingual"]
    user_lang_name = s["user_lang_name"]
    prof_lang_name = s["prof_lang_name"]
    hctx = sessions.get_hctx(sid)
    result_type = d.get("result","draw")
    advice = d.get("advice","")
    gauge_history = d.get("gauge_history", [])
    debate_info = d.get("debate_info", {})

    result_label = {"student_wins":"Student Victory 🏆","professor_wins":"Professor Victory 📖","draw":"Draw 🤝"}.get(result_type,"End")
    user_stance = debate_info.get("userStance","")
    prof_stance = debate_info.get("profStance","")

    # Format gauge data for the prompt
    gauge_data_str = json.dumps(gauge_history, ensure_ascii=False)

    report_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config={"temperature":0.7,"top_p":0.95,"top_k":40,"max_output_tokens":100000},
        system_instruction=SYS_REPORT
    )

    prompt = f"""Create an entertaining, visual debate report as a standalone HTML file.
Write the report in {user_lang_name}.

===== DEBATE INFO =====
Topic: "{topic}"
Student stance: {user_stance}
Professor stance: {prof_stance}
Result: {result_label}
Final Score: Student {score} : Professor {100-score}
Languages: Student={user_lang_name}, Professor={prof_lang_name}
Mode: {"Bilingual Learning" if bi else "Same Language"}

===== GAUGE HISTORY (time-series data) =====
{gauge_data_str}
Each entry: {{"turn": N, "score": student_score (0-100), "reason": "judge's reason", "userMsg": "what student said"}}

===== FULL DEBATE LOG =====
{hctx}

===== Professor's Advice =====
{advice}

===== REPORT STRUCTURE (MUST follow this order) =====

1. **HEADER**: "Debate Duel ⚔️ Report" + Topic + Date
2. **STANCES**: "Your stance: {user_stance}" / "Professor's stance: {prof_stance}"
3. **RESULT BANNER**: Big, colorful result display with score gauge visualization
   - Student: {score}% / Professor: {100-score}%
   - Use emoji and color appropriate to result

4. **★ LINE CHART (MAIN FEATURE - MUST BE LARGE AND PROMINENT) ★**
   This is the CENTERPIECE of the report. Create a large, detailed line chart using Chart.js (load from CDN: https://cdn.jsdelivr.net/npm/chart.js).
   
   Chart requirements:
   - X-axis: Turn numbers (0, 1, 2, 3...)
   - Y-axis: Student score (0-100), with 50 as the center line (draw a dashed horizontal line at 50)
   - Line color: gradient from orange (#e17055) to purple (#6c5ce7)
   - Fill area above 50 with green tint, below 50 with red tint
   - Chart height: at least 400px
   - CRITICAL: Add Chart.js annotation plugin (https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation) for annotations
   - Add vertical annotation lines + labels at KEY TURNING POINTS with entertaining commentary in {user_lang_name}, examples:
     * When score jumps significantly: "🔥 Here the tide turned!" / "💥 Critical hit!"
     * When student scores big: "⚡ Devastating argument!" / "🎯 Bullseye!"
     * When professor fights back: "😱 Professor strikes back!" / "🌊 Counterattack wave!"
     * At the final decisive moment: "⚔️ The finishing blow!" / "🏆 Victory sealed!"
   - Make annotations fun, dramatic, like sports commentary
   - Add data labels showing score at each point

5. **DETAILED ANALYSIS**: Turn-by-turn breakdown with:
   - What the student argued (brief)
   - How the score changed and WHY
   - Key vocabulary or expressions used (if bilingual)
   - Use emoji and color coding for score changes (green ↑, red ↓)

6. **ADVICE & SUMMARY**:
   - Professor's advice (from the debate)
   - Strengths of the student (specific examples)
   - Areas for improvement (specific suggestions)
   - Overall assessment

===== HTML STYLE =====
- Dark theme: background #0f1117, text #e4e6f0
- Gradient accents: #e17055 → #6c5ce7
- Cards with border-radius: 16px, subtle borders
- Professional but FUN - use emoji, color, personality
- Responsive design
- @media print styles (light background for printing)
- Footer: "Generated by Debate Duel ⚔️ — 株式会社スター・ライト"
- Load Chart.js from CDN: <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
- Load annotation plugin: <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
- Chart must be in a <canvas> element with JavaScript to render it

Return ONLY raw JSON: {{"deepthink":"your analysis","final_report_html_str":"<!DOCTYPE html>..."}}
"""

    p, raw = await _gemini(report_model, prompt)

    if p and "final_report_html_str" in p:
        html_str = p["final_report_html_str"]
        import datetime
        now = datetime.datetime.now()
        ts = now.strftime("%Y%m%d_%H%M")
        safe_topic = re.sub(r'[^a-zA-Z0-9]+', '_', topic.lower()).strip('_')[:60]
        filename = f"{safe_topic}_{ts}.html"
        filepath = _TDIR / filename
        filepath.write_text(html_str, encoding="utf-8")
        logger.info(f"Report generated: {filename} ({len(html_str)} chars)")
        return {"filename": filename, "filepath": str(filepath)}

    logger.warning(f"Report generation failed: {str(raw)[:300]}")
    return JSONResponse({"error":"Report generation failed"},500)

@app.get("/download_report/{filename}")
async def download_report(filename: str):
    filepath = _TDIR / filename
    if not filepath.exists():
        return JSONResponse({"error":"not found"},404)
    return FileResponse(str(filepath), media_type="text/html", filename=filename)

@app.post("/api/login")
async def login(request: Request):
    try:
        d = await request.json()
        username = (d.get("username") or "").strip()
        password = d.get("password") or ""
        if username == AUTH_USER and password == AUTH_PASS:
            r = JSONResponse({"success": True})
            r.set_cookie(key=AUTH_COOKIE, value=AUTH_USER, path="/", max_age=86400 * 7)  # 7 days
            return r
        return JSONResponse({"success": False, "error": "ユーザー名またはパスワードが正しくありません。"}, status=401)
    except Exception:
        return JSONResponse({"success": False, "error": "リクエストが不正です。"}, status=400)

@app.post("/api/logout")
async def logout():
    r = RedirectResponse(url="/", status_code=302)
    r.delete_cookie(key=AUTH_COOKIE, path="/")
    return r

@app.get("/api/check_auth")
async def check_auth(request: Request):
    return {"logged_in": _is_logged_in(request)}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if _is_logged_in(request):
        return HTMLResponse((Path(__file__).parent / "debate_app.html").read_text(encoding="utf-8"))
    return HTMLResponse((Path(__file__).parent / "login.html").read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
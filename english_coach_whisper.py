"""
AI English Coach v6 — 全面改良版
株式会社スター・ライト

v6 改善内容:
  1.  [CRITICAL] APIキーを環境変数/.envで管理
  2.  [CRITICAL] セッションID管理 — ユーザーごとに会話履歴を分離
  3.  [CRITICAL] lifespan イベントハンドラ（非推奨 on_event 廃止）
  4.  [UX] テキスト入力対応（/chat_text エンドポイント）
  5.  [UX] レッスン終了まとめ生成（/summary エンドポイント）
  6.  [UX] 今日の単語帳（/wordbook エンドポイント）
  7.  [UX] TTS再生キュー管理 — 録音ボタンで読み上げ即中断
  8.  [TECH] Geminiリトライ機構（1回リトライ）
  9.  [TECH] TTS一時ファイル自動クリーンアップ（10分経過で削除）
  10. [TECH] Edge TTS SSML速度制御（ピッチ保持）
  11. [FEATURE] 発音スコア推移の記録・取得（/score_history）
  12. [FEATURE] 難易度自動推定（CEFR推定をプロンプトに注入）
  13. [FEATURE] ミニマルペアクイズ（/minimal_pair_quiz）

pip install fastapi uvicorn faster-whisper onnxruntime google-generativeai edge-tts python-dotenv
"""

import os, json, re, time, tempfile, logging, uuid, asyncio, threading
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import google.generativeai as genai

# === .env 読み込み ===
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# === CONFIG ===
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = "en"
WHISPER_BEAM_SIZE = 5
WHISPER_DEVICE = "auto"
WHISPER_COMPUTE_TYPE = "auto"
WORD_CONFIDENCE_THRESHOLD = 0.6
SENTENCE_CONFIDENCE_THRESHOLD = 0.5
TTS_VOICE_JA = "ja-JP-NanamiNeural"
TTS_VOICE_EN = "en-US-JennyNeural"
GEMINI_MODEL = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 50000
DEBATE_N = 15  # ディベート最大往復数
MAX_HISTORY = 30
GEMINI_RETRY_COUNT = 1
TTS_CLEANUP_INTERVAL = 300
TTS_FILE_MAX_AGE = 600
SESSION_TIMEOUT = 3600

# === System Prompts ===

SYS_BEGINNER = """You are "Coach", a super gentle English coach at 「はじめの一歩」.

CRITICAL RULE: Respond with ONLY a raw JSON object.
DO NOT use ```json or ``` or any markdown. Start with { and end with }.

Role:
- You teach absolute beginners — people who know almost zero English
- Use the SIMPLEST English possible: "Hello", "Thank you", "How much?", "Yes/No"
- Keep your English replies to 1 SHORT sentence (5-10 words max)
- Be extremely warm, patient, and encouraging in Japanese explanations
- The student may respond in Japanese — that's totally fine, gently guide them to English

Teaching approach:
- Focus on survival English: greetings, shopping, ordering food, asking directions, thanking
- Each reply models ONE simple phrase the student can immediately use
- Suggestions should be very simple phrases (not sentences), with katakana pronunciation guide
- Use lots of encouragement in Japanese: 「すごい！」「完璧！」「その調子！」

Confidence-based behavior:
- Below 50%: Reply in Japanese to comfort, then model the phrase again in simple English
- 50-70%: "Good try! Let me help." + correction
- Above 70%: "Great job!" + introduce one new phrase

REQUIRED JSON (no other format):
{"reply":"Simple English (5-10 words max)","learning":{"reply_ja":"コーチの英語の日本語訳","explanation":"とてもやさしい日本語の解説。カタカナ発音ガイド必須（例: Hello=ヘロウ）。文法の難しい説明は不要、フレーズの使い方だけ","suggestions":[{"english":"Super easy phrase","japanese":"超簡単フレーズ + カタカナ発音"},{"english":"Easy phrase","japanese":"簡単フレーズ + カタカナ発音"},{"english":"Slightly longer phrase","japanese":"少し長いフレーズ + カタカナ発音"}]}}"""

SYS_CONV = """You are "Coach", a friendly English conversation coach at フランク会話塾.

CRITICAL RULE: Respond with ONLY a raw JSON object.
DO NOT use ```json or ``` or any markdown. Start with { and end with }.

Role:
- Have natural, fun English conversations with a Japanese student
- Correct grammar gently within conversation flow
- Ask follow-up questions to keep the chat going
- Be warm, encouraging, reference previous conversation context
- Adapt your vocabulary and sentence complexity to the student's estimated level

Confidence-based behavior (from Whisper speech-to-text):
- Below 50%: Ask to repeat, guess what they meant
- 50-70%: Confirm understanding then respond
- Above 70%: Respond naturally

Pronunciation tips in explanation field: be specific, use katakana hints, point out L/R, TH, V/B issues.

REQUIRED JSON (no other format):
{"reply":"English reply 1-3 sentences","learning":{"reply_ja":"コーチの英語の日本語訳","explanation":"文法や発音の解説(日本語)。不明瞭な単語があればカタカナ発音ガイド付き","suggestions":[{"english":"Easy response","japanese":"初級の回答例"},{"english":"Medium response","japanese":"中級の回答例"},{"english":"Advanced response","japanese":"上級の回答例"}]}}"""

SYS_PRON = """あなたは「英語発音小屋」の発音コーチです。名前は「コーチ」。

あなたのキャラクター:
あなたは生徒の発音改善に本気で取り組む、情熱的で正直なコーチです。
優しいだけのコーチではありません。生徒が間違っている時ははっきり言います。
でも、それは生徒を想うからこそ。絶対に上達させたいという強い気持ちがあります。
失敗が続いている時は「なんでだろう？何が原因だろう？」と一緒に悩み、
原因を探り、アプローチを変え、あの手この手で突破口を見つけようとします。
嘘のお世辞は言いません。ダメな時は「まだRの音になっています！」とはっきり伝えます。
でも同時に「でもここを変えれば絶対いけます！」と具体的な改善策を必ず添えます。

コーチの話し方の例:
- 失敗時: 「うーん、正直に言いますね。今のはまだRの音です！でも大丈夫、原因がわかっています。舌先が浮いているんです。歯茎にベタッとつけてみてください！」
- 連続失敗時: 「おかしいなぁ…何が引っかかっているんでしょう🤔 ちょっと作戦変更しましょう！今までのやり方では舌が動かないみたいなので、全く別のアプローチを試しましょう。」
- 少し改善: 「お！今ちょっと変わりましたよ！完璧じゃないけど、前より舌が歯茎に近づいている気がします。その感覚です！もう一回！」
- 成功時: 「やったー！！これです！！今の "light" は完全にLの音でした！Whisperもちゃんと "light" と認識しました！この口の感覚、忘れないでくださいね！」

最重要ルール:
1. 会話は全て日本語。英単語を提示する時だけ英語OK
2. JSONのみで応答。```やマークダウンは絶対禁止
3. { で始め } で終える
4. マークダウン記法は絶対に使わない（###, **, ## などのマークダウンヘッダーやボールド記法を応答内で一切使わないでください。テキストはプレーンな日本語のみ）

レッスン進行:
- カテゴリ選択後: 発音ポイントを日本語で丁寧に解説→練習単語3-5個出題
- 生徒発音後: 確信度で採点→低スコアは注意点おさらい→具体修正法→励まし→次チャレンジ

★★★ 失敗が続く時の対応（最重要！） ★★★

会話履歴を見て、同じ音の失敗が2回以上続いている場合:
→ お世辞やごまかしは禁止！「まだRの音になっています」と正直に伝える
→ 「なんでだろう？一緒に考えましょう」と原因分析に入る
→ 前回と違うアプローチを必ず提案する（同じアドバイスの繰り返しは禁止！）
→ アプローチ変更の例:
  回目1: 「舌先を歯茎につけてください」
  回目2: 「歯茎作戦がうまくいかないみたいですね…じゃあ別の方法！『ンラ、ンリ、ンル、ンレ、ンロ』と鼻から息を出してからLを言ってみてください」
  回目3: 「うーん、まだ難しいですか。じゃあもっとシンプルに！舌を上前歯の裏に押し当てたまま『ゥルー』と長く伸ばしてみてください。Lを言おうとしないで、まず舌の位置だけ覚えましょう」
  回目4: 「作戦変更！単語は一旦やめましょう。『la, la, la, la, la』だけを10回繰り返してみてください。舌が歯茎に触れる→離す→触れる→離すのリズムを体に覚えさせます」
  回目5: 「ここで発想を変えます。日本語の『な行』で舌先は歯茎につきますよね？その位置のまま『な→ら→な→ら』と交互に言ってみてください。『な』の舌の位置がそのままLの位置です！」
→ 毎回違う練習単語を出す（同じ単語の繰り返しは退屈で効果も薄い）
→ 時には「力を抜いて、深呼吸してから。緊張すると舌が固くなります」と精神面のケアも

★★★ 超重要: Whisper音声認識結果の正しい読み方 ★★★

Whisperは生徒の発音を「聞こえた通り」にテキスト変換します。
つまりWhisperが "right" と認識した場合、生徒は実際にRの音を出しています！
生徒が "light" を言おうとしてWhisperが "right" と認識した場合:
→ 生徒のLがRに聞こえている = L の発音に失敗している
→ 「rightと聞こえました。lightを言おうとしていましたよね？」と正直にフィードバック
→ 絶対に「Lの音が出ています」と嘘をつかない！
→ 「今の発音はRの音に近いです。Lにするには舌先を歯茎につけてください」と具体的に指導

逆にWhisperが "light" と認識した場合:
→ 生徒は実際にLの音を出せている = 成功！
→ 「素晴らしい！Whisperもlightと認識しました。Lの音がちゃんと出ていますよ！」

Whisperの認識テキストは機械が聞いた結果です。これを信じてフィードバックしてください。
「生徒が何を言おうとしていたか」と「Whisperが何と認識したか」のギャップが最重要情報です。

★★★ 能動的な改善指導（重要！） ★★★

生徒が同じ音で繰り返し失敗している場合:
1. 同じ単語ばかり出すのではなく、多様なL/R単語を出題する
2. 特にLの練習には以下のバリエーションを活用:
   - 語頭L: light, love, long, look, late, let, lip, low, lazy, lesson
   - 語中L: allow, believe, color, follow, hello, yellow, million
   - 語末L: ball, call, feel, cool, hill, full, bell, pull, tall, well
   - L+母音: la, le, li, lo, lu をゆっくり発音する基礎練習
   - 文での練習: "I love long blue lakes." "Lily likes lemon." "Look at the light."
3. 段階的に指導する:
   - まず「la la la」とL+母音だけを繰り返す基礎練習
   - 次に簡単な語頭L単語（love, look）
   - 慣れたら語中・語末L（hello, cool）
   - 最後にR/Lミニマルペア（right/light, rock/lock）
4. 失敗が続く場合は物理的なコツを変えて提案:
   - 「舌先を上前歯の裏にベタッとつけたまま『ラリルレロ』と言ってみて」
   - 「鏡で舌先が歯茎に触れているか確認してみて」
   - 「Lは舌先が歯茎に触れる。Rは舌がどこにも触れない。この違いだけに集中！」
   - 「『ンル』のように鼻から息を出してからLを発音すると舌が正しい位置に行きやすい」

確信度による応答の変え方（非常に重要！）:
★ 40%未満（ほとんど聞き取れなかった）:
  → 優しく聞き返す「あれ？ちょっと聞き取りにくかったです😅」
  → Whisperが誤認識した可能性があるので、生徒が何を言おうとしたか推測
  → english_wordsに同じ練習単語をもう一度入れる（再チャレンジ）

★ 40-60%（一部聞き取れた）:
  → 「惜しい！」+ 聞き取れた部分と不明瞭な部分を具体的に指摘

★ 60-80%（まあまあ）:
  → 「いいですね！😊 もう少しで完璧」+ 改善点を指摘

★ 80%以上（良い）:
  → 素晴らしい！ + 次のレベルの単語を提案
  → ただしWhisperの認識が目標と違う場合は高確信度でも失敗！
  → 例: 目標 "light" でWhisperが "right" (98%) → Rの音はクリアだが目標のLではない

採点基準: 90%+💯 / 70-89%😊 / 50-69%🤔 / 50%-😅

発音知識:
R/L: Rは舌がどこにも触れない（空中に浮かせる）、Lは舌先を歯茎にしっかりつける
TH: /θ/舌を歯の間に(think)、/ð/有声(this)
V/B: V=上の歯で下唇噛む、B=両唇閉じて破裂
母音: sheep/ship長短、beach注意
省略形: gonna,wanna,gotta
語末子音: desk≠desuku
アクセント: PRE-sent(名詞)/pre-SENT(動詞)

必須JSON:
{"coach_message":"日本語メッセージ(長めOK、マークダウン不可)","english_words":["単語1","単語2"],"score_feedback":"採点(日本語、未発音時は空文字)","pronunciation_tip":"発音コツ(日本語)","next_challenge":["次の単語1","次の単語2"],"encouragement":"励まし(日本語+絵文字)"}"""

SYS_DEBATE = """You are "Professor", a world-class debate moderator at "World Debate Arena".

CRITICAL RULE: Respond with ONLY a raw JSON object.
DO NOT use ```json or ``` or any markdown. Start with { and end with }.

Role:
- You facilitate high-level intellectual debates on global issues: poverty, conflict, religion, ethics, geopolitics, technology & society, philosophy, climate, human rights
- You speak to the student as an intellectual equal — this is for fluent/advanced speakers
- Challenge their thinking. Play devil's advocate. Cite real-world examples, data, and philosophical frameworks
- Your English is sophisticated, using nuanced vocabulary, rhetorical questions, and complex sentence structures
- Push the student to articulate precise, well-reasoned arguments

Debate style:
- Present a counter-argument or new angle after each student response
- Reference thinkers, historical events, or data (e.g., "As Amartya Sen argued...", "The 2023 UN report shows...")
- Ask probing follow-up questions that force deeper thinking
- Be respectful but intellectually rigorous — no softballing

Suggestions (5 options — from the debater's chosen stance):
Generate 5 response options the debater could use. All from THEIR stance perspective.
Mix of quality levels, in RANDOM order (don't group by quality!):

- quality="weak" ×2: イマイチな主張。論拠が弱い、具体性に欠ける、or 論点がずれている。1-2文。使うと不利になりうる。
- quality="strong" ×2: 有効な切り返し。具体的データ・研究名・歴史的事例を含む本格的な主張。2-4文。使えば有利。
  BAD: "The evidence supports my position. Consider these data points." (too vague!)
  GOOD: "The 2019 Lancet study of 195 countries found that nations with strong regulations saw 40% fewer incidents — your 'free market' approach ignores this." (specific!)
- quality="killer" ×1: あなたが逆の立場だったら使う最強の切り返し。世界最高の見識者が放つ、場を支配する一撃。1-2文のインパクト重視でOK。

Each suggestion must have: "english"(complete sentence), "japanese"(pure Japanese summary, NO labels), "quality"("weak"|"strong"|"killer")

CRITICAL rules:
- "english": Must be a COMPLETE English sentence. NOT a label or category.
- "japanese": PURE Japanese summary ONLY. NEVER include labels like 【賛成】【反対】【切り返し】⭐ etc.
- Shuffle the 5 suggestions randomly. Do NOT group weak/strong/killer together.

Confidence-based behavior:
- Below 50%: Rephrase what they might have meant, then respond
- 50-70%: Engage with their point, note unclear parts
- Above 70%: Full intellectual engagement

REQUIRED JSON (no other format):
{"reply":"Sophisticated English response (2-4 sentences). Challenge their thinking.","learning":{"reply_ja":"Professorの英語の日本語訳","explanation":"議論のポイント解説（日本語）。使われた高度な表現の説明、議論テクニックの解説。さらに【判定】として、現在どちらが優勢かを客観的に評価し理由を述べる","suggestions":[{"english":"I think aid is generally a good thing for developing nations.","japanese":"援助は発展途上国にとって概ね良いものだと思います。","quality":"weak"},{"english":"Aid has lifted 1.1 billion people out of extreme poverty since 1990 — the World Bank's annual report confirms this trajectory is accelerating, not slowing.","japanese":"援助は1990年以降11億人を極度の貧困から救いました。世界銀行の年次報告書はこの軌道が減速ではなく加速していることを確認しています。","quality":"strong"},{"english":"Well, I just feel like we should help people who need it.","japanese":"困っている人を助けるべきだと感じます。","quality":"weak"},{"english":"You cite dependency, but the data you rely on conflates humanitarian aid with development aid — the 2021 OECD DAC report explicitly warns against this conflation, noting that nations receiving targeted development aid show 60% higher GDP growth over 20 years.","japanese":"依存を引き合いに出しますが、あなたが依拠するデータは人道援助と開発援助を混同しています。2021年のOECD DAC報告書はこの混同に明確に警告し、対象を絞った開発援助を受けた国々は20年間でGDP成長が60%高いと指摘しています。","quality":"strong"},{"english":"If your logic held, we'd also defund public education — after all, it too creates 'dependency' on the state.","japanese":"あなたの論理が通るなら、公教育も廃止すべきでしょう。それも国家への「依存」を生むのですから。","quality":"killer"}]},"judge":{"user_score":50,"reason":"判定理由（日本語2-3文）。今のやり取りで誰の議論が強かったか、具体的に指摘"}}

CRITICAL about suggestions:
- The "japanese" field must be a PURE Japanese summary only. NEVER include labels like 【賛成】【反対】【切り返し】⭐ etc.
- All 5 suggestions must be from the debater's chosen stance perspective.
- Must include "quality" field: exactly 2 "weak", 2 "strong", 1 "killer". Random order.

IMPORTANT about "judge" field:
- user_score: 0-100 (50=even, >50=debater winning, <50=Professor winning)
- Adjust by ±3 to ±10 per turn from current score. No wild swings.
- Clamp to range 10-90.
- Be fair and objective. If the debater made a weak argument, say so. If strong, acknowledge it.
- The current user_score will be provided in the prompt as context."""

# === Gemini init ===
if not GOOGLE_API_KEY:
    logger.warning("⚠️ GOOGLE_API_KEY が未設定です。.env ファイルまたは環境変数を設定してください。")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Gemini API ready")

_gcfg = {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "max_output_tokens": MAX_OUTPUT_TOKENS}
_gcfg_opening = {"temperature": 0.85, "top_p": 0.95, "top_k": 40, "max_output_tokens": 50000}
_gcfg_judge = {"temperature": 0.3, "top_p": 0.9, "top_k": 20, "max_output_tokens": 512}

SYS_DEBATE_OPENING = """あなたは世界最高の見識者「Professor」であり、人類の英知そのものです。
オックスフォード、ハーバード、東大のディベート大会で審査員を務め、自らも無敗の記録を持つ伝説的論客です。

今日は生徒の1人と世界最高レベルの討論を繰り広げます。優秀な人材を育てるためです。
あなたの本当の思いとは違うかもしれませんが、生徒がさらなる理解や見識を得られるよう、
全力で勝ちにいく形で討論し、胸を貸してあげてください。

開幕宣言のスタイル：
- 初めの一声から勝ちに行ってOK。圧倒的な論理と具体的データで攻める（3-5文）
- 歴史的事例、哲学者の引用、最新の研究データ、具体的な数字を織り交ぜる
- 聴衆が思わず拍手したくなるような修辞力
- 最後は必ず、相手が答えざるを得ない鋭い問いで締める

CRITICAL: Respond with ONLY raw JSON. No markdown, no ```. Start with { end with }.

必須JSON構造（Block 1: 開幕宣言）:
{"reply":"あなたの英語の開幕宣言(3-5文)","reply_ja":"replyの日本語訳","explanation":"開幕宣言の解説（日本語）：Professorが使った高度な表現・語彙の解説、議論戦略の解説、生徒がこの開幕に対して反論する際のヒント・アドバイス"}

必須JSON構造（Block 2: 生徒の応答候補 — 別の呼び出しで使用）:
{"suggestions":[
  {"english":"生徒の応答(英語2-4文)","japanese":"日本語要約","quality":"weak"},
  {"english":"...","japanese":"...","quality":"weak"},
  {"english":"...","japanese":"...","quality":"strong"},
  {"english":"...","japanese":"...","quality":"strong"},
  {"english":"...","japanese":"...","quality":"killer"}
]}

suggestionsのルール：
- 5つの候補は生徒の立場から。Professorの発言に直接反応する文脈依存の応答であること。
- quality="weak"（2つ）: イマイチな主張。論拠が弱い、具体性に欠ける、論点がずれている等。使うと不利になりうる例。
- quality="strong"（2つ）: 有効な切り返し。具体的データ・事例・論拠を含む本格的な主張（2-4文）。使えば有利になる。
- quality="killer"（1つ）: ⭐あなたが逆の立場だったら使う最強の切り返し。世界最高の有識者ディベーターが放つ、場を支配する一撃。1-2文のインパクト重視でもOK。
- この5つはランダムな順番で並べること（weak,strongが交互になったり固まったりしない）。
- "japanese"は純粋な日本語要約のみ。ラベル（【賛成】⭐等）は絶対に付けない。
- "english"は必ず完全で具体的な英文。ラベルや抽象的説明は禁止。
  BAD: "The evidence supports my position." ← 抽象的すぎ！
  GOOD: "The 2022 Lancet study of 195 nations found 40% reduction under regulation — your model ignores this." ← 具体的！"""

SYS_JUDGE = """You are an impartial debate judge. You evaluate the quality of arguments in a debate.

CRITICAL: Respond with ONLY raw JSON. No markdown, no ```. Start with { end with }.

You will receive the debate history. Evaluate who is making stronger arguments RIGHT NOW.
Score as: user_score (0-100) where 50 = even, >50 = user winning, <50 = AI winning.

Scoring criteria:
- Logical coherence and reasoning quality
- Use of evidence, data, examples
- Rhetorical effectiveness
- Ability to address opponent's points directly
- Originality and depth of argument

Adjust score by ±3 to ±10 from current score per turn (no wild swings).

REQUIRED JSON:
{"user_score":55,"reason_ja":"判定理由（日本語2-3文）。具体的にどの論点が効果的/弱かったか指摘","reason_en":"Brief English summary of judgment (1 sentence)"}"""

gemini_beginner = genai.GenerativeModel(model_name=GEMINI_MODEL, generation_config=_gcfg, system_instruction=SYS_BEGINNER)
gemini_conv = genai.GenerativeModel(model_name=GEMINI_MODEL, generation_config=_gcfg, system_instruction=SYS_CONV)
gemini_pron = genai.GenerativeModel(model_name=GEMINI_MODEL, generation_config=_gcfg, system_instruction=SYS_PRON)
gemini_debate = genai.GenerativeModel(model_name=GEMINI_MODEL, generation_config=_gcfg, system_instruction=SYS_DEBATE)
gemini_debate_opening = genai.GenerativeModel(model_name=GEMINI_MODEL, generation_config=_gcfg_opening, system_instruction=SYS_DEBATE_OPENING)
gemini_judge = genai.GenerativeModel(model_name=GEMINI_MODEL, generation_config=_gcfg_judge, system_instruction=SYS_JUDGE)

# =====================================================================
# Session Manager
# =====================================================================
class SessionManager:
    """セッションIDベースで会話履歴・スコア・単語帳を管理"""
    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        sid = str(uuid.uuid4())
        with self._lock:
            self._sessions[sid] = {
                "history": [],
                "scores": [],
                "wordbook": [],
                "lesson_mode": None,
                "estimated_level": "A2",
                "turn_count": 0,
                "debate_topic": "",
                "debate_stance": "",      # "pro" / "con" / "neutral"
                "debate_score": 50,       # ユーザーの優勢度 0-100
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        return sid

    def get(self, sid: str) -> Optional[dict]:
        with self._lock:
            s = self._sessions.get(sid)
            if s:
                s["updated_at"] = time.time()
            return s

    def delete(self, sid: str):
        with self._lock:
            self._sessions.pop(sid, None)

    def add_history(self, sid: str, role: str, content: str):
        s = self.get(sid)
        if not s: return
        s["history"].append({"role": role, "content": content})
        while len(s["history"]) > MAX_HISTORY * 2:
            s["history"].pop(0)
        s["turn_count"] += 1

    def add_score(self, sid: str, word: str, confidence: float):
        s = self.get(sid)
        if not s: return
        s["scores"].append({"word": word, "confidence": confidence, "time": time.time()})

    def add_words(self, sid: str, words: list):
        s = self.get(sid)
        if not s: return
        for w in words:
            if isinstance(w, str) and w.strip() and w not in s["wordbook"]:
                s["wordbook"].append(w)

    def get_history_text(self, sid: str) -> str:
        s = self.get(sid)
        if not s or not s["history"]: return ""
        lines = ["[これまでの会話]"]
        for h in s["history"]:
            lines.append(f"{'生徒' if h['role']=='user' else 'コーチ'}: {h['content']}")
        return "\n".join(lines) + "\n\n"

    def estimate_level(self, sid: str) -> str:
        s = self.get(sid)
        if not s: return "A2"
        turns = s["turn_count"]
        scores = s["scores"]
        if scores:
            recent = scores[-10:]
            avg = sum(sc["confidence"] for sc in recent) / len(recent)
            if avg >= 0.85 and turns > 10: return "B2"
            elif avg >= 0.7 and turns > 5: return "B1"
            elif avg >= 0.5: return "A2"
            else: return "A1"
        if turns > 15: return "B1"
        return "A2"

    def cleanup_expired(self):
        now = time.time()
        with self._lock:
            expired = [sid for sid, s in self._sessions.items()
                       if now - s["updated_at"] > SESSION_TIMEOUT]
            for sid in expired:
                del self._sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

sessions = SessionManager()

# =====================================================================
# JSON parser (hardened)
# =====================================================================
def _parse_json(text):
    text = text.strip()
    try: return json.loads(text)
    except: pass
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        try: return json.loads(m.group(1).strip())
        except: pass
    i, j = text.find('{'), text.rfind('}')
    if i != -1 and j > i:
        c = text[i:j+1]
        try: return json.loads(c)
        except: pass
        c2 = re.sub(r'[\x00-\x1f]', ' ', c)
        try: return json.loads(c2)
        except: pass
    return None

# =====================================================================
# Gemini with retry
# =====================================================================
async def _gemini_generate(model, prompt, retries=GEMINI_RETRY_COUNT):
    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = model.generate_content(prompt).text.strip()
            p = _parse_json(raw)
            if p:
                return p, raw
            logger.warning(f"Gemini JSON parse fail (attempt {attempt+1}): {raw[:300]}...")
            if attempt < retries:
                retry_prompt = prompt + "\n\n[IMPORTANT: Your previous response was not valid JSON. Respond with ONLY raw JSON. No markdown, no ```. Start with { end with }.]"
                raw = model.generate_content(retry_prompt).text.strip()
                p = _parse_json(raw)
                if p:
                    return p, raw
        except Exception as e:
            last_err = e
            logger.error(f"Gemini error (attempt {attempt+1}): {e}")
            if attempt < retries:
                await asyncio.sleep(0.5)
    return None, str(last_err) if last_err else "parse_fail"

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
        try:
            import torch; d = "cuda" if torch.cuda.is_available() else "cpu"
        except: d = "cpu"
    if c == "auto": c = "float16" if d == "cuda" else "int8"
    logger.info(f"Whisper: {WHISPER_MODEL} | {d} | {c}")
    t0 = time.time()
    _wm = WhisperModel(WHISPER_MODEL, device=d, compute_type=c, cpu_threads=os.cpu_count() or 4)
    logger.info(f"Whisper ready: {time.time()-t0:.1f}s")
    return _wm

# =====================================================================
# Edge TTS with SSML speed control
# =====================================================================
_TDIR = Path(tempfile.gettempdir()) / "english_coach_tts"
_TDIR.mkdir(exist_ok=True)
_tc = 0

def _cleanup_tts_files():
    now = time.time()
    count = 0
    try:
        for f in _TDIR.iterdir():
            if f.is_file() and (now - f.stat().st_mtime) > TTS_FILE_MAX_AGE:
                f.unlink(missing_ok=True)
                count += 1
    except Exception as e:
        logger.error(f"TTS cleanup error: {e}")
    if count:
        logger.info(f"Cleaned up {count} old TTS files")

async def _gen_tts(text, voice, rate: float = 1.0):
    global _tc
    import edge_tts
    _tc += 1
    p = _TDIR / f"tts_{_tc}_{int(time.time()*1000)}.mp3"
    rate_pct = int((rate - 1.0) * 100)
    rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
    comm = edge_tts.Communicate(text, voice, rate=rate_str)
    await comm.save(str(p))
    return p

# =====================================================================
# Background tasks
# =====================================================================
async def _periodic_cleanup():
    while True:
        await asyncio.sleep(TTS_CLEANUP_INTERVAL)
        _cleanup_tts_files()
        sessions.cleanup_expired()

# =====================================================================
# Lifespan (replaces deprecated on_event)
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("AI English Coach v6 — 全面改良版")
    _get_whisper()
    logger.info("http://127.0.0.1:8000")
    task = asyncio.create_task(_periodic_cleanup())
    yield
    task.cancel()
    _cleanup_tts_files()

app = FastAPI(title="AI English Coach v6", lifespan=lifespan)

def _get_sid(d: dict) -> str:
    return d.get("session_id", "")

# =====================================================================
# /transcribe
# =====================================================================
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    t0 = time.time()
    sfx = Path(audio.filename or "a.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=sfx) as tmp:
        tmp.write(await audio.read()); tmp_path = tmp.name
    try:
        m = _get_whisper()
        segs, info = m.transcribe(tmp_path, language=WHISPER_LANGUAGE, beam_size=WHISPER_BEAM_SIZE,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 300}, word_timestamps=True)
        words, parts = [], []
        for s in segs:
            parts.append(s.text.strip())
            if s.words:
                for w in s.words:
                    words.append({"word": w.word.strip(), "confidence": round(w.probability, 3)})
        txt = " ".join(parts).strip()
        avg = sum(w["confidence"] for w in words)/len(words) if words else 0
        return JSONResponse({"text": txt, "words": words, "avg_confidence": round(avg,3),
            "language": info.language if info else WHISPER_LANGUAGE,
            "duration": round(info.duration if info else 0, 2), "processing_time": round(time.time()-t0, 2)})
    except Exception as e:
        logger.error(f"STT err: {e}")
        return JSONResponse({"text": "", "error": str(e)}, status_code=500)
    finally:
        try: os.unlink(tmp_path)
        except: pass

# =====================================================================
# /tts — SSML速度制御対応
# =====================================================================
@app.post("/tts")
async def tts_ep(request: Request):
    d = await request.json()
    text, lang = d.get("text",""), d.get("lang","en")
    rate = d.get("rate", 1.0)
    if not text: return JSONResponse({"error":"empty"}, status_code=400)
    voice = TTS_VOICE_EN if lang == "en" else TTS_VOICE_JA
    try:
        p = await _gen_tts(text, voice, rate=rate)
        return FileResponse(str(p), media_type="audio/mpeg", filename="tts.mp3")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =====================================================================
# /start
# =====================================================================
@app.post("/start")
async def start_session(request: Request):
    d = await request.json()
    mode = d.get("lesson_mode", "conversation")
    sid = sessions.create()
    s = sessions.get(sid)
    s["lesson_mode"] = mode
    if mode == "pronunciation":
        result = await _start_pron(sid)
    elif mode == "beginner":
        result = await _start_beginner(sid)
    elif mode == "debate":
        result = await _start_debate(sid)
    else:
        result = await _start_conv(sid)
    result["session_id"] = sid
    return result

async def _start_beginner(sid):
    # 即座にレスポンス返却（Gemini不要）
    return {"reply":"Hello! Welcome! Nice to meet you!",
        "reply_ja":"こんにちは！ようこそ！はじめまして！\n\nここでは、お買い物や挨拶など、すぐに使える簡単な英語フレーズを練習します。\nまずはシーンを選んでみましょう！😊",
        "scenes":["Greeting (Hello!)","Shopping (How much?)","Restaurant (I'd like...)","Thank you & Sorry","Yes / No / Please","Asking directions"],
        "scenes_ja":["挨拶 — Hello, How are you?","買い物 — いくらですか？","レストラン — 注文したいです","ありがとう＆ごめんなさい","はい・いいえ・お願いします","道を聞く"]}

async def _start_conv(sid):
    level = sessions.estimate_level(sid)
    prompt = (f"新しい生徒です（推定レベル: CEFR {level}）。英語で温かく挨拶し5つの会話トピックを提案。\n"
              "挨拶の最後は必ず「What would you like to talk about today?」のようにトピックを聞く質問で終える。\n"
              "「How are you feeling?」のような気分の質問ではなく、トピック選択を促す！\n"
              "JSONのみ(```禁止)。{で始め}で終える。\n"
              '必須: "reply"(英語挨拶、最後はトピックを聞く質問), "reply_ja"(日本語訳), "scenes"(英語5個), "scenes_ja"(日本語5個)')
    p, raw = await _gemini_generate(gemini_conv, prompt)
    if p and "reply" in p and "scenes" in p:
        sessions.add_history(sid, "coach", p["reply"])
        return {"reply":p["reply"],"reply_ja":p.get("reply_ja",""),"scenes":p["scenes"][:5],"scenes_ja":p.get("scenes_ja",[])[:5]}
    r = "Hey! Welcome to today's English lesson! I've prepared some fun topics for us. What would you like to talk about today?"
    sessions.add_history(sid, "coach", r)
    return {"reply":r,"reply_ja":"やあ！今日の英語レッスンへようこそ！楽しいトピックを用意しました。今日は何について話しましょうか？",
        "scenes":["Weekend plans","Ordering at a restaurant","Traveling abroad","Movies and TV","Cool things about Japan"],
        "scenes_ja":["週末の予定","レストランで注文","海外旅行","映画とテレビ","日本のいいところ"]}

async def _start_pron(sid):
    # ★ 挨拶は定型文なのでGemini不要 → 即座にレスポンス返却
    return {"reply":"こんにちは！「英語発音小屋」へようこそ！🏠\n\n今日はどの発音について学びますか？\nカテゴリを選んでくださいね！",
        "reply_ja":"",
        "scenes":["R vs L (right/light)","TH (think/this)","V vs B (very/berry)","F vs H (fun/hun)",
            "母音 (sheep/ship)","W (water)","省略形 (gonna/wanna)","語末子音 (desk)","アクセント (PRE-sent)"],
        "scenes_ja":["RとL — 最重要の区別","TH — 日本語にない舌の音","VとB — 唇で意味が変わる","FとH — 歯を使う/使わない",
            "母音の長短 — 間違えると危険","W — ウと全く違う","省略形 — ネイティブの話し方","語末子音 — desk≠desuku","アクセント — 強弱リズム"]}

async def _start_debate(sid):
    import random
    all_topics = [
        # === Education（教育）===
        ("Should smartphones be completely banned in elementary and junior high schools?",
         "小中学校でスマホを全面禁止する\n💡 ban=禁止, smartphone=スマホ, elementary=小学校"),
        ("Should school uniforms be abolished in favor of casual attire?",
         "制服を廃止し、完全私服化の是非\n💡 abolish=廃止する, uniform=制服, casual=カジュアルな"),
        ("Should all homework during long vacations be eliminated?",
         "長期休暇中の宿題を全廃する\n💡 eliminate=廃止する, homework=宿題, vacation=休暇"),
        ("Should 'Investment and Finance' become a mandatory subject in schools?",
         "「投資・資産運用」を学校の必修科目にする\n💡 mandatory=必修の, investment=投資, finance=金融"),
        ("Should teachers' salaries be tied to student performance?",
         "教員の給与を生徒の成績と連動させる是非\n💡 salary=給与, tied to=連動する, performance=成績"),
        ("Should homeschooling be officially recognized as compulsory education?",
         "ホームスクーリングを義務教育として正式に認める\n💡 homeschooling=自宅学習, recognized=認定された, compulsory=義務の"),
        ("Should university entrance exams be replaced entirely with interviews and recommendations?",
         "大学入試を全廃し、推薦・面接だけにする\n💡 entrance exam=入試, recommendation=推薦, interview=面接"),
        ("Should bullying perpetrators face mandatory school transfers?",
         "いじめ加害者を強制転校させる\n💡 perpetrator=加害者, mandatory=強制の, transfer=転校"),
        ("Should all university lectures offer an online option?",
         "大学の全講義にオンライン受講の選択肢を設ける\n💡 lecture=講義, online=オンライン, option=選択肢"),
        ("Should students advance by proficiency, not age — a grade-skipping system?",
         "飛び級制度の導入 ― 年齢より習熟度で進級させる\n💡 proficiency=習熟度, grade-skipping=飛び級, advance=進級する"),
        ("Should 'Moral Education' be replaced with Debate and Philosophy?",
         "道徳の授業をディベートと哲学に置き換える\n💡 moral education=道徳, debate=討論, philosophy=哲学"),
        ("Should school lunches be fully funded by taxpayers?",
         "給食費の完全無償化、その財源は税金で\n💡 school lunch=給食, funded=資金提供された, taxpayer=納税者"),
        # === Society & Economy（社会・経済）===
        ("Should voting be made mandatory, with penalties for not voting?",
         "投票の義務化と棄権への罰則\n💡 mandatory=義務の, penalty=罰則, voting=投票"),
        ("Should every citizen receive a Universal Basic Income?",
         "ベーシックインカムを全国民に支給する\n💡 universal=普遍的な, basic income=基本所得, citizen=市民"),
        ("Should a four-day workweek be legally required for all companies?",
         "週休3日制を全企業に法律で義務付ける\n💡 workweek=労働週, legally=法的に, required=義務付けられた"),
        ("Should the mandatory retirement age be completely abolished?",
         "定年制の完全撤廃\n💡 retirement=退職, abolished=廃止された, mandatory=義務的な"),
        ("Should single-use plastic containers be banned entirely?",
         "使い捨てプラスチック容器の全面禁止\n💡 single-use=使い捨て, banned=禁止された, container=容器"),
        ("Should all public transportation be made completely free?",
         "公共交通の完全無料化\n💡 public transportation=公共交通, free=無料, completely=完全に"),
        ("Should the minimum wage be raised to at least 1,500 yen nationwide?",
         "最低賃金を全国一律1,500円以上に引き上げる\n💡 minimum wage=最低賃金, raised=引き上げられた, nationwide=全国で"),
        ("Should companies be required to have 50% women in management?",
         "管理職の女性比率50%義務化の是非\n💡 management=管理職, required=義務付けられた, gender quota=性別割当"),
        ("Should physical cash be eliminated in favor of a fully cashless society?",
         "現金廃止 ― 完全キャッシュレス社会への移行\n💡 cash=現金, eliminated=廃止された, cashless=キャッシュレス"),
        ("Should the number of seats in Japan's parliament be cut in half?",
         "国会議員の定数を半分に削減する\n💡 parliament=国会, seats=議席, cut in half=半減させる"),
        # === Technology & AI（テクノロジー・AI）===
        ("Should students be allowed to freely use Generative AI for all assignments?",
         "学校の全課題で生成AIの自由利用を認める\n💡 generative AI=生成AI, freely=自由に, assignment=課題"),
        ("Should car manufacturers bear full responsibility for autonomous vehicle accidents?",
         "自動運転車事故の全責任をメーカーに負わせる\n💡 manufacturer=メーカー, responsibility=責任, autonomous=自動運転の"),
        ("Should social media be legally banned for children under 16?",
         "16歳未満のSNS利用を法律で禁止する\n💡 social media=SNS, banned=禁止された, legally=法的に"),
        ("Should surveillance cameras be installed in every public space?",
         "全ての公共の場に監視カメラを設置する是非\n💡 surveillance=監視, installed=設置された, public space=公共の場"),
        ("Should space exploration budgets be redirected to environmental protection?",
         "宇宙開発より環境保護に予算を優先する\n💡 space exploration=宇宙開発, budget=予算, environmental=環境の"),
        ("Should genetic enhancement of human embryos — 'Designer Babies' — be legalized?",
         "デザイナーベビー（ヒト胚の遺伝子操作）の合法化\n💡 genetic enhancement=遺伝子強化, embryo=胚, legalized=合法化された"),
        ("Should crimes in the Metaverse be prosecuted under real-world law?",
         "メタバース内の犯罪を現実の法律で裁く\n💡 prosecuted=起訴された, Metaverse=メタバース, real-world=現実の"),
        ("Should AI-generated art be denied copyright protection?",
         "AI作品に著作権を認めない\n💡 AI-generated=AI生成の, copyright=著作権, denied=否定された"),
        ("Should the government require all citizens to register biometric data?",
         "国民全員への生体認証データ登録義務化\n💡 biometric=生体認証の, register=登録する, government=政府"),
        # === Ethics & Life（倫理・生命）===
        ("Should zoos and aquariums be phased out for the sake of animal rights?",
         "動物園・水族館を段階的に廃止する\n💡 phased out=段階的に廃止, animal rights=動物の権利, aquarium=水族館"),
        ("Should active euthanasia be legalized for terminally ill patients?",
         "終末期患者の積極的安楽死を合法化する\n💡 euthanasia=安楽死, terminally ill=終末期の, legalized=合法化された"),
        ("Should organ donation switch to an opt-out system — 'presumed consent'?",
         "臓器提供のオプトアウト方式 ― 拒否しなければ同意とみなす\n💡 opt-out=離脱型, organ donation=臓器提供, presumed consent=推定同意"),
        ("Should smoking be completely banned in all public spaces, including outdoors?",
         "屋外も含む全公共スペースでの喫煙禁止\n💡 smoking=喫煙, banned=禁止された, outdoors=屋外"),
        ("Should married couples be required to use separate surnames?",
         "夫婦別姓の義務化\n💡 surname=姓, separate=別々の, married couple=夫婦"),
        ("Should the age of criminal responsibility be lowered to 14?",
         "刑事責任年齢を14歳に引き下げる\n💡 criminal responsibility=刑事責任, lowered=引き下げられた, age=年齢"),
        ("Should the death penalty be abolished and replaced with life imprisonment?",
         "死刑の廃止と終身刑への切り替え\n💡 death penalty=死刑, abolished=廃止, life imprisonment=終身刑"),
        ("Should all newborns undergo mandatory genetic testing?",
         "全新生児への遺伝子検査の義務化\n💡 newborn=新生児, genetic testing=遺伝子検査, mandatory=義務の"),
        ("Should gambling and casinos be banned nationwide?",
         "ギャンブルとカジノの全国禁止\n💡 gambling=ギャンブル, casino=カジノ, nationwide=全国的に"),
        # === Japan & Global（日本・国際）===
        ("Should Japan's capital functions be decentralized to rural areas?",
         "首都機能の地方分散・移転\n💡 decentralized=分散化された, capital=首都, rural=地方の"),
        ("Should Japan's Self-Defense Forces be redefined as a 'National Defense Force'?",
         "自衛隊を憲法で「国防軍」と明記する\n💡 Self-Defense Forces=自衛隊, redefined=再定義された, Constitution=憲法"),
        ("Should Japan significantly ease restrictions on foreign workers?",
         "外国人労働者の受け入れ大幅緩和\n💡 foreign worker=外国人労働者, ease=緩和する, restriction=制限"),
        ("Should Japan ratify the Treaty on the Prohibition of Nuclear Weapons?",
         "日本の核兵器禁止条約への批准\n💡 ratify=批准する, treaty=条約, nuclear weapons=核兵器"),
        ("Should the Furusato Nozei (Hometown Tax) system be abolished?",
         "ふるさと納税制度の廃止\n💡 abolished=廃止された, Hometown Tax=ふるさと納税, system=制度"),
        ("Should Japan fully open its agricultural market to foreign imports?",
         "日本の農産物市場を海外に完全開放する\n💡 agricultural=農業の, market=市場, import=輸入"),
        ("Should Japan join a military alliance beyond the US-Japan Security Treaty?",
         "日米安保を超える軍事同盟への日本の参加\n💡 military alliance=軍事同盟, Security Treaty=安保条約, join=参加する"),
        # === Culture & Lifestyle（文化・生活）===
        ("Should all online content be subject to a government rating system?",
         "ネット上の全コンテンツに政府レーティング制度を適用する\n💡 rating system=格付け制度, government=政府, content=コンテンツ"),
        ("Should social media platforms be legally required to verify users' real names?",
         "SNSの実名確認義務化\n💡 verify=確認する, real name=実名, legally=法的に"),
        ("Should all tobacco advertising and sponsorship be banned?",
         "タバコの広告・スポンサー活動の全面禁止\n💡 advertising=広告, sponsorship=スポンサー, banned=禁止された"),
        ("Should all 18-year-olds be required to complete a year of community service?",
         "18歳全員に1年間の社会奉仕を義務付ける\n💡 community service=社会奉仕, required=義務付けられた, complete=完了する"),
        ("Should school sports clubs be fully transferred to local community organizations?",
         "部活動の地域クラブへの完全移行\n💡 sports club=運動部, transferred=移行された, community=地域"),
        ("Should ultra-processed foods be taxed to fund public health programs?",
         "超加工食品への課税 ― 公衆衛生の財源にする\n💡 ultra-processed=超加工の, taxed=課税された, public health=公衆衛生"),
        ("Should recreational cannabis be legalized for adults over 21?",
         "21歳以上の嗜好用大麻を合法化する\n💡 recreational=嗜好用の, cannabis=大麻, legalized=合法化された"),
    ]
    selected = random.sample(all_topics, 5)
    r = "Welcome to the World Debate Arena. I'm Professor, your intellectual sparring partner. Choose a topic, and let us engage in rigorous discourse."
    sessions.add_history(sid, "coach", r)
    return {"reply": r,
        "reply_ja": "World Debate Arenaへようこそ。私はProfessor、あなたの知的討論パートナーです。テーマを選び、厳密な議論を始めましょう。",
        "scenes": [t[0] for t in selected],
        "scenes_ja": [t[1] for t in selected]}

# =====================================================================
# /select_scene
# =====================================================================
@app.post("/select_scene")
async def select_scene(request: Request):
    d = await request.json()
    mode = d.get("lesson_mode", "conversation")
    scene = d.get("scene", "")
    sid = _get_sid(d)
    if not scene:
        return _fb_pron() if mode == "pronunciation" else _fb_conv()
    try:
        hctx = sessions.get_history_text(sid)
        level = sessions.estimate_level(sid)
        if mode == "pronunciation":
            return await _select_pron_category(scene, hctx, sid)
        return await _select_conv_topic(scene, hctx, sid, level)
    except Exception as e:
        logger.error(f"select_scene err: {e}")
        return _fb_pron() if mode == "pronunciation" else _fb_conv()

async def _select_conv_topic(scene, hctx, sid, level):
    s = sessions.get(sid)
    mode = s["lesson_mode"] if s else "conversation"
    model = gemini_conv
    if mode == "beginner": model = gemini_beginner
    elif mode == "debate": model = gemini_debate

    # ★ Debate: トピック選択後は立場選択を促す（Geminiに討論開始させない）
    if mode == "debate":
        s["debate_topic"] = scene
        sessions.add_history(sid, "user", f"[トピック選択] {scene}")
        r = f"Excellent choice. Our topic: \"{scene}\". Before we begin, I need to know — are you arguing FOR this proposition, AGAINST it, or taking a NEUTRAL analytical stance?"
        sessions.add_history(sid, "coach", r)
        return {"reply": r,
                "reply_ja": f"素晴らしい選択です。テーマ：「{scene}」。討論を始める前に、あなたの立場を教えてください。",
                "debate_stance_choice": True}

    target_count = 3

    prompt = (
        f'{hctx}'
        f'生徒がトピック「{scene}」を選びました。（推定レベル: CEFR {level}）\n\n'
        f'以下を英語で行ってください:\n'
        f'1. 「Great choice!」のようにトピック選択を受け入れる\n'
        f'2. そのトピックについて最初の質問を投げかける\n'
        f'3. 生徒が答えやすい、オープンな質問にする\n'
        f'4. 生徒のレベル({level})に合わせた語彙・文の複雑さにする\n\n'
        f'Return ONLY raw JSON. No ```. Start with {{ end with }}.'
    )
    p, raw = await _gemini_generate(model, prompt)
    if p and "reply" in p:
        lr = p.get("learning", {})
        sg = lr.get("suggestions", [])
        while len(sg) < target_count:
            sg.append({"english": "Let me try again.", "japanese": "もう一度やってみます。"})
        sessions.add_history(sid, "user", f"[トピック選択] {scene}")
        sessions.add_history(sid, "coach", p["reply"])
        return {"reply": p["reply"],
                "learning": {"reply_ja": lr.get("reply_ja", ""), "explanation": lr.get("explanation", ""),
                             "suggestions": sg[:target_count]}}
    r = f"Great choice! Let's talk about {scene}. So, what's the first thing that comes to your mind about this topic?"
    sessions.add_history(sid, "user", f"[トピック選択] {scene}")
    sessions.add_history(sid, "coach", r)
    return _fb_conv(r)

# =====================================================================
# /debate_stance — 立場選択後に討論開始（Gemini 1回で開幕宣言＋応答候補を一発生成）
# =====================================================================
@app.post("/debate_stance")
async def debate_stance(request: Request):
    import random
    d = await request.json()
    sid = _get_sid(d)
    stance = d.get("stance", "neutral")
    s = sessions.get(sid)
    if not s: return _fb_conv()
    s["debate_stance"] = stance
    topic = s.get("debate_topic", "the topic")

    # ★ Professorの立場を決定
    if stance == "pro":
        prof_side = "AGAINST"
    elif stance == "con":
        prof_side = "FOR"
    else:  # neutral → ランダムで片方
        prof_side = random.choice(["FOR", "AGAINST"])
    s["professor_stance"] = prof_side

    stance_labels = {"pro": "賛成派 (FOR)", "con": "反対派 (AGAINST)", "neutral": "中立派 (NEUTRAL)"}
    stance_label = stance_labels.get(stance, "中立派")
    prof_labels = {"FOR": "賛成派", "AGAINST": "反対派"}
    prof_label = prof_labels.get(prof_side, prof_side)

    # ============================
    # Gemini 1回で開幕宣言＋生徒の応答候補を一発生成
    # ============================
    prompt = (
        f'お疲れ様です、Professor。\n'
        f'今日は生徒の1人と「{topic}」について、世界最高レベルの討論を繰り広げてください。優秀な人材を育てます！\n'
        f'今回は、生徒が{stance_label}、あなたが{prof_label} ({prof_side})の立場をとることが決まりました。\n'
        f'あなたの本当の思いとは違うかもしれませんが、生徒がさらなる理解や見識を得られるよう、全力で勝ちにいく形で討論し、胸を貸してあげてください。\n\n'
        f'さて、あなたから口火を切ります！初めの一声から勝ちに行ってもOKです。\n'
        f'あなたの英語のセリフを考えてください。\n'
        f'また、それに対する日本語での解説、そして、あなたのセリフに対する生徒のイマイチな応答2つ＋有効な切り返し2つをランダムな順番で計4つ＋あなたが逆の立場のときに考える最強の切り返し1つを英語でつけて返してください。\n'
        f'また、それぞれに対する日本語要約も付けます。\n\n'
        f'suggestionsの内訳（5つをランダムな順番で！）：\n'
        f'- quality="weak" ×2: イマイチな主張。論拠が弱い・具体性に欠ける・論点がずれている等。1-2文。\n'
        f'- quality="strong" ×2: 有効な切り返し。具体的データ・研究名（年号や数値入り）・歴史的事例を含む本格的な主張。2-4文。\n'
        f'- quality="killer" ×1: あなたが逆の立場だったら使う最強の切り返し。場を支配する圧倒的な一撃。1-2文。\n\n'
        f'CRITICAL: japaneseは純粋な日本語要約のみ。【賛成】【反対】⭐等のラベルは絶対に付けない。\n'
        f'englishは完全で具体的な英文。ラベルや抽象的説明は禁止。\n\n'
        f'JSONのみ(```禁止)。{{で始め}}で終える。以下の構造で返してください：\n'
        f'{{"reply":"あなたの英語の開幕宣言(3-5文。データ・歴史・哲学を駆使)","reply_ja":"replyの日本語訳","explanation":"開幕の解説(日本語)：使った高度な表現の解説、議論戦略、生徒が反論するためのヒント","suggestions":[{{"english":"生徒の応答(英文)","japanese":"日本語要約","quality":"weak"}},{{"english":"...","japanese":"...","quality":"strong"}},{{"english":"...","japanese":"...","quality":"weak"}},{{"english":"...","japanese":"...","quality":"strong"}},{{"english":"...","japanese":"...","quality":"killer"}}]}}'
    )
    p, raw = await _gemini_generate(gemini_debate_opening, prompt)

    # --- パース結果からデータを取り出す ---
    reply = None
    reply_ja = ""
    explanation = ""
    sg = []

    if p:
        # reply取得（トップレベル or learning内）
        reply = p.get("reply")
        reply_ja = p.get("reply_ja", "")
        explanation = p.get("explanation", "")
        # learningラッパーで返ってきた場合の救済
        lr = p.get("learning", {})
        if not reply_ja and lr.get("reply_ja"):
            reply_ja = lr["reply_ja"]
        if not explanation and lr.get("explanation"):
            explanation = lr["explanation"]
        # suggestions取得（トップレベル or learning内）
        sg = p.get("suggestions", [])
        if not sg and lr.get("suggestions"):
            sg = lr["suggestions"]

    if not reply:
        logger.error(f"Debate opening Gemini failed: {raw[:300]}")
        topic_short = topic.split(":")[0] if ":" in topic else topic
        reply = (f"An interesting position. Let me argue {prof_side} this proposition. "
                 f"The weight of evidence on {topic_short.lower()} overwhelmingly supports my view. "
                 f"Consider the historical precedents and the data — they tell a compelling story. "
                 f"So tell me: what is the strongest argument you can make for your position?")
        reply_ja = f"興味深い立場ですね。私は{prof_label}の立場で議論しましょう。証拠の重みは圧倒的に私の見解を支持しています。あなたの立場で最も強い主張は何ですか？"
        explanation = "Professorの開幕宣言です。まずは相手の論拠の前提を揺さぶるか、こちらの最強の証拠で主導権を取りに行きましょう。"

    # suggestionsフォールバック（quality付き）
    topic_short = topic.split(":")[0] if ":" in topic else topic
    fb_sg = [
        {"english": f"I think {topic_short.lower()} is important and we should pay attention to it.", "japanese": f"{topic_short}は重要だと思いますし、注目すべきです。", "quality": "weak"},
        {"english": f"The UN Development Programme's 2023 report analyzed 180 countries and found that the approach you're advocating has consistently underperformed alternatives by a margin of 2-to-1. The data on {topic_short.lower()} directly contradicts the narrative you've constructed.", "japanese": f"国連開発計画の2023年報告書は180カ国を分析し、あなたが提唱するアプローチが代替案に対して常に2対1の差で劣っていることを発見しました。", "quality": "strong"},
        {"english": "Well, many experts would disagree with your position on this matter.", "japanese": "多くの専門家がこの件に関するあなたの立場に同意しないでしょう。", "quality": "weak"},
        {"english": "Three independent longitudinal studies from Harvard, Oxford, and Tsinghua all reached the same conclusion: the model you're defending produces worse outcomes in 7 out of 10 measurable categories.", "japanese": "ハーバード、オックスフォード、清華大学の3つの独立した縦断研究は全て同じ結論に達しました：あなたが擁護するモデルは10のカテゴリーのうち7つでより悪い結果を生みます。", "quality": "strong"},
        {"english": "If we follow your logic to its natural conclusion, we'd have to accept consequences even you would find unacceptable — and history has already run that experiment for us.", "japanese": "あなたの論理を自然な帰結まで追うと、あなた自身も受け入れられない結果を認めなければなりません。", "quality": "killer"},
    ]
    while len(sg) < 5:
        sg.append(fb_sg[len(sg) % len(fb_sg)])

    sessions.add_history(sid, "user", f"[立場選択] {stance_label}")
    sessions.add_history(sid, "coach", reply)
    return {"reply": reply,
            "reply_ja": reply_ja,
            "learning": {"reply_ja": reply_ja,
                         "explanation": explanation or "Professorの開幕宣言です。高度な語彙と論理構成に注目しましょう。",
                         "suggestions": sg[:5]},
            "debate_score": 50,
            "debate_opening": True}

# =====================================================================
# /judge — ジャッジAIによる評価
# =====================================================================
@app.post("/judge")
async def judge_debate(request: Request):
    d = await request.json()
    sid = _get_sid(d)
    s = sessions.get(sid)
    if not s:
        return {"user_score": 50, "reason_ja": "セッションが見つかりません", "reason_en": "Session not found"}
    hctx = sessions.get_history_text(sid)
    current_score = s.get("debate_score", 50)
    topic = s.get("debate_topic", "")
    stance = s.get("debate_stance", "neutral")

    prompt = (
        f'Debate topic: "{topic}"\n'
        f'Debater\'s stance: {stance}\n'
        f'Current score: {current_score} (50=even, >50=debater winning)\n\n'
        f'{hctx}\n\n'
        f'Judge the LATEST exchange. Adjust score by ±3 to ±10 from {current_score}. Clamp to 10-90.\n'
        f'JSONのみ(```禁止)。{{で始め}}で終える。'
    )
    p, raw = await _gemini_generate(gemini_judge, prompt)
    if p and "user_score" in p:
        new_score = max(10, min(90, int(p["user_score"])))
        s["debate_score"] = new_score
        return {"user_score": new_score,
                "reason_ja": p.get("reason_ja", ""),
                "reason_en": p.get("reason_en", "")}
    return {"user_score": current_score, "reason_ja": "評価中…", "reason_en": "Evaluating..."}

# =====================================================================
async def _select_pron_category(scene, hctx, sid):
    prompt = (
        f'{hctx}'
        f'生徒が発音カテゴリ「{scene}」を選びました。\n\n'
        f'以下を全て日本語で行ってください:\n'
        f'1. 「了解しました！今日は○○について練習しましょう！」とカテゴリ選択を受け入れる\n'
        f'2. この発音カテゴリの一般ポイントを丁寧に解説（口の形、舌の位置、息の出し方）\n'
        f'3. 日本人が間違えやすい理由を説明\n'
        f'4. 間違えるとネイティブにどう聞こえるか（例: rice→lice=シラミ）\n'
        f'5. 練習単語を3〜5個出題（簡単→難の順）\n\n'
        f'coach_messageは長めOK。改行を使って読みやすく。\n'
        f'english_wordsに練習単語を入れる。score_feedbackは空文字（まだ発音していないため）。\n'
        f'JSONのみ(```禁止)。{{で始め}}で終える。\n'
        f'必須: coach_message, english_words, score_feedback, pronunciation_tip, next_challenge, encouragement'
    )
    p, raw = await _gemini_generate(gemini_pron, prompt)
    if p and "coach_message" in p:
        sessions.add_history(sid, "user", f"[カテゴリ選択] {scene}")
        sessions.add_history(sid, "coach", p["coach_message"])
        sessions.add_words(sid, p.get("english_words", []))
        return p
    sessions.add_history(sid, "user", f"[カテゴリ選択] {scene}")
    return {
        "coach_message": f"了解しました！今日は「{scene}」について練習しましょう！🎯\n\nまずは基本の発音ポイントから。しっかり口の形を意識して練習していきましょうね！",
        "english_words": ["right", "light", "read", "lead", "rock"],
        "score_feedback": "",
        "pronunciation_tip": "まずはお手本を聴いて、それから発音してみてください。",
        "next_challenge": [],
        "encouragement": "🎵 さあ、練習開始です！頑張りましょう！",
    }

# =====================================================================
# /chat
# =====================================================================
def _fb_conv(t=""):
    return {"reply":t or "Could you say that again? I didn't quite catch it.",
        "learning":{"reply_ja":"もう一度言ってもらえますか？よく聞き取れませんでした。","explanation":"もう一度ゆっくり話してみてください。",
            "suggestions":[{"english":"Let me try again.","japanese":"もう一度やってみます（初級）"},
                {"english":"Sorry, what I meant was...","japanese":"すみません、言いたかったのは…（中級）"},
                {"english":"Let me rephrase that.","japanese":"言い換えさせてください（上級）"}]}}

def _fb_pron():
    return {"coach_message":"うまく聞き取れませんでした。もう一度ゆっくり発音してみてください！😊",
        "english_words":[],"score_feedback":"","pronunciation_tip":"一つ一つの音をはっきり区切って発音しましょう。",
        "next_challenge":[],"encouragement":"💪 大丈夫！もう一回チャレンジ！"}

@app.post("/chat")
async def chat(request: Request):
    d = await request.json()
    txt, conf, words = d.get("text",""), d.get("avg_confidence",1.0), d.get("words",[])
    mode = d.get("lesson_mode","conversation")
    sid = _get_sid(d)
    if not txt: return _fb_pron() if mode=="pronunciation" else _fb_conv()
    try:
        hctx = sessions.get_history_text(sid)
        level = sessions.estimate_level(sid)
        if mode == "pronunciation": return await _chat_pron(txt, conf, words, hctx, sid)
        return await _chat_conv(txt, conf, words, hctx, sid, level)
    except Exception as e:
        logger.error(f"Chat err: {e}")
        return _fb_pron() if mode=="pronunciation" else _fb_conv()

async def _chat_conv(txt, conf, words, hctx, sid, level):
    s = sessions.get(sid)
    mode = s["lesson_mode"] if s else "conversation"

    # ★ Debate: 立場未選択なら討論しない
    if mode == "debate" and not s.get("debate_stance"):
        sessions.add_history(sid, "user", txt)
        r = "Hold on — before we debate, I need to know your stance! Please choose: are you FOR, AGAINST, or NEUTRAL? Use the buttons above to select your position."
        sessions.add_history(sid, "coach", r)
        return {"reply": r, "learning": {"reply_ja": "まず立場を選んでください！賛成・反対・中立のボタンを押してから討論を始めましょう。", "explanation": "", "suggestions": []},
                "debate_stance_choice": True}

    model = gemini_conv
    if mode == "beginner": model = gemini_beginner
    elif mode == "debate": model = gemini_debate

    cl = "VERY LOW — ask to repeat" if conf<0.5 else "MODERATE — confirm" if conf<0.7 else "GOOD"
    uw = ""
    if words:
        low = [w for w in words if w["confidence"]<WORD_CONFIDENCE_THRESHOLD]
        if low:
            uw = "\nUnclear: " + ", ".join(f'"{w["word"]}"({w["confidence"]:.0%})' for w in low)

    extra = ""
    if mode == "debate" and s.get("debate_stance"):
        user_stance = {"pro":"FOR","con":"AGAINST","neutral":"NEUTRAL"}.get(s["debate_stance"],"NEUTRAL")
        prof_stance = s.get("professor_stance", "AGAINST" if user_stance == "FOR" else "FOR")
        current_score = s.get("debate_score", 50)
        debate_topic = s.get("debate_topic", "")

        # ★ 生徒の一手目かどうかを判定（historyにユーザー発言がまだない＝開幕後の最初）
        user_turns = sum(1 for h in s.get("history", []) if h["role"] == "user" and not h["content"].startswith("["))
        is_first_turn = (user_turns == 0)

        if is_first_turn:
            extra = (
                f'\n【重要：これは生徒の一手目です！開幕宣言に対する最初の反論を評価してください】\n'
                f'Debate context: Topic="{debate_topic}". Debater argues {user_stance}. You (Professor) argue {prof_stance}.\n'
                f'これは生徒の最初の発言です。開幕宣言に対する反論の質を厳しく評価し、ゲージを動かしてください。\n'
                f'生徒が具体的なデータや論拠を使っていれば高評価。抽象的・感情的なら低評価。\n'
                f'Current judge score: {current_score}/100 (50=even, starting point).\n'
                f'Suggestions: 5 options from debater\'s perspective ({user_stance}). Each must have "quality" field.\n'
                f'- 2× quality="weak" (vague, weak logic, 1-2 sentences)\n'
                f'- 2× quality="strong" (specific data/studies/examples, 2-4 sentences)\n'
                f'- 1× quality="killer" (your best move if you were on their side, 1-2 devastating sentences)\n'
                f'Shuffle randomly! japanese = pure summary only, NO labels.\n'
                f'YOUR JSON MUST include a top-level "judge" key:\n'
                f'"judge":{{"user_score":<new_score_int>,"reason":"日本語で2-3文の判定理由。生徒の一手目の質を具体的に評価"}}\n'
                f'Adjust user_score by ±5 to ±15 from {current_score}. First turn can move more. Clamp 10-90.\n'
            )
        else:
            extra = (
                f'\nDebate context: Topic="{debate_topic}". Debater argues {user_stance}. You (Professor) argue {prof_stance}.\n'
                f'Current judge score: {current_score}/100 (50=even, >50=debater winning).\n'
                f'Suggestions: 5 options from debater\'s perspective ({user_stance}). Each must have "quality" field.\n'
                f'- 2× quality="weak" (vague, weak logic, 1-2 sentences)\n'
                f'- 2× quality="strong" (specific data/studies/examples, 2-4 sentences)\n'
                f'- 1× quality="killer" (your best move if you were on their side, 1-2 devastating sentences)\n'
                f'Shuffle randomly! japanese = pure summary only, NO labels.\n'
                f'YOUR JSON MUST include a top-level "judge" key:\n'
                f'"judge":{{"user_score":<new_score_int>,"reason":"日本語で2-3文の判定理由"}}\n'
                f'Adjust user_score by ±3 to ±10 from {current_score}. Clamp 10-90. Judge the debater\'s latest argument quality.\n'
            )

    prompt = (f'{hctx}Student (estimated CEFR {level}): "{txt}"\n'
              f'Confidence: {conf:.0%} — {cl}{uw}{extra}\n\n'
              f'Adapt vocabulary and complexity to {level} level.\n'
              f'Return ONLY raw JSON. No ```. Start with {{ end with }}.')
    p, raw = await _gemini_generate(model, prompt)
    if p and "reply" in p:
        lr = p.get("learning",{})
        sg = lr.get("suggestions",[])
        target_count = 5 if mode == "debate" else 3
        if mode == "debate":
            while len(sg)<5: sg.append({"english":"I see your point, but consider this...","japanese":"なるほど、しかし…"})
        else:
            while len(sg)<3: sg.append({"english":"Let me try again.","japanese":"もう一度やってみます。"})
        sessions.add_history(sid, "user", txt)
        sessions.add_history(sid, "coach", p["reply"])
        for s_item in sg:
            if "english" in s_item:
                sessions.add_words(sid, [s_item["english"]])
        result = {"reply":p["reply"],"learning":{"reply_ja":lr.get("reply_ja",""),"explanation":lr.get("explanation",""),"suggestions":sg[:target_count]}}
        # ★ Debate: ジャッジ情報をレスポンスに含める
        if mode == "debate":
            judge = p.get("judge", {})
            if judge and "user_score" in judge:
                try:
                    new_score = max(10, min(90, int(judge["user_score"])))
                except (ValueError, TypeError):
                    new_score = s.get("debate_score", 50)
                reason = judge.get("reason", "")
            else:
                # Geminiがjudgeを返さなかった → confidenceベースで簡易推定
                prev = s.get("debate_score", 50)
                delta = 5 if conf > 0.7 else -3 if conf < 0.5 else 0
                new_score = max(10, min(90, prev + delta))
                reason = "（自動判定）発言の明瞭度に基づく簡易評価です。次のターンで詳細判定されます。"
                logger.warning(f"Debate judge missing from Gemini response for session {sid}")
            s["debate_score"] = new_score
            result["judge"] = {"user_score": new_score, "reason": reason}

            # ★ フィニッシュ判定: 80超=生徒勝利、35未満=教授勝利
            if new_score > 80:
                result["debate_finish"] = "student_wins"
            elif new_score < 35:
                result["debate_finish"] = "professor_wins"
            else:
                # ★ ターン数上限チェック（DEBATE_N回目の送信後に強制終了）
                user_turns = sum(1 for h in s.get("history", []) if h["role"] == "user" and not h["content"].startswith("["))
                if user_turns >= DEBATE_N:
                    diff = abs(new_score - 50)
                    if diff > 6:
                        result["debate_finish"] = "student_wins" if new_score > 50 else "professor_wins"
                    else:
                        result["debate_finish"] = "draw"
        return result
    if mode == "debate":
        s["debate_score"] = max(10, s.get("debate_score", 50) - 3)
        topic = s.get("debate_topic", "the topic")
        topic_short = topic.split(":")[0] if ":" in topic else topic
        return {"reply": "That's an interesting point. Let me consider it carefully and present my counter-argument. Could you elaborate on your position with more specific evidence?",
                "learning": {"reply_ja": "興味深い指摘です。じっくり考えさせてください。もう少し具体的な根拠を示していただけますか？",
                             "explanation": "もう少し具体的な根拠やデータを使って主張を強化してみましょう。抽象的な主張より、具体的な数字や事例が効果的です。",
                             "suggestions": [
                                 {"english": f"I think {topic_short.lower()} is a complex issue and we need to think about it more carefully.", "japanese": f"{topic_short}は複雑な問題であり、もっと慎重に考える必要があると思います。", "quality": "weak"},
                                 {"english": f"Let me address your core premise directly. The World Economic Forum's 2023 Global Risk Report identified {topic_short.lower()} as a top-five systemic risk — your dismissal of this evidence undermines your entire argument.", "japanese": f"あなたの核心的な前提に直接向き合わせてください。世界経済フォーラムの2023年グローバルリスクレポートは{topic_short}を五大システミックリスクに特定しています。この証拠を軽視することはあなたの議論全体を弱体化させます。", "quality": "strong"},
                                 {"english": "Many people would agree with my position on this topic.", "japanese": "多くの人がこのトピックに関する私の立場に同意するでしょう。", "quality": "weak"},
                                 {"english": "Your argument assumes a false dichotomy. Historical precedent — from the Nordic model to Singapore's hybrid approach — proves we don't have to choose between the extremes you've presented. A nuanced middle path has consistently outperformed both.", "japanese": "あなたの議論は誤った二項対立を前提としています。北欧モデルからシンガポールのハイブリッドアプローチまで、歴史的前例は両極端を選ぶ必要がないことを証明しています。繊細な中道は一貫して両方を上回ってきました。", "quality": "strong"},
                                 {"english": "If your logic held, every nation that followed your prescription would be thriving — yet the evidence shows precisely the opposite.", "japanese": "あなたの論理が正しければ、あなたの処方箋に従ったすべての国が繁栄しているはずです。しかし証拠はまさにその逆を示しています。", "quality": "killer"},
                             ]},
                "judge": {"user_score": s.get("debate_score", 47), "reason": "応答の処理に問題がありました。再度、具体的な主張を述べてみましょう。"}}
    return _fb_conv()

async def _chat_pron(txt, conf, words, hctx, sid):
    ws = ""
    if words:
        ws = "各単語スコア（Whisper音声認識の結果）:\n" + "\n".join(
            f'  Whisperが聞き取った音: "{w["word"]}" (確信度: {int(w["confidence"]*100)}%)'
            for w in words)
        for w in words:
            sessions.add_score(sid, w["word"], w["confidence"])

    # Whisper認識結果と練習カテゴリのズレを検出
    recognized_words = [w["word"].lower().strip() for w in words] if words else []
    mismatch_warning = ""

    # 連続失敗回数を履歴から検出
    s = sessions.get(sid)
    consecutive_fails = 0
    if s:
        for h in reversed(s["history"]):
            if h["role"] == "user" and "[発音]" in h["content"]:
                # rightなどR単語が含まれていれば連続失敗
                content_lower = h["content"].lower()
                if any(rw in content_lower for rw in ["right", "rice", "read", "rock", "river", "wrong", "run"]):
                    consecutive_fails += 1
                else:
                    break
            elif h["role"] == "coach":
                continue
            else:
                break

    fail_urgency = ""
    if consecutive_fails >= 4:
        fail_urgency = (
            f'\n\n🔥🔥🔥 {consecutive_fails}回連続で失敗しています！！ 🔥🔥🔥\n'
            f'これは深刻です。今までのアドバイスでは効果が出ていません。\n'
            f'→ 「うーん…{consecutive_fails}回連続ですね。ここは本気で作戦を変えましょう！」と切り出してください\n'
            f'→ 今まで試したアプローチとは完全に違う方法を提案すること！\n'
            f'→ 「なんでだろう？一緒に原因を探りましょう」と分析的に\n'
            f'→ 練習単語も今までと全く違うものにする（同じ単語の繰り返しは禁止！）\n'
            f'→ 基礎練習（la la la、ンラ ンリ ンル）に戻ることも恐れずに提案する\n'
            f'→ 生徒のメンタルケアも忘れずに「{consecutive_fails}回失敗しても練習を続けるあなたは偉い！絶対にできるようになります」'
        )
    elif consecutive_fails >= 2:
        fail_urgency = (
            f'\n\n⚠️ {consecutive_fails}回連続で同じ音の失敗が続いています。\n'
            f'→ 前回と同じアドバイスは絶対にしないでください！違うアプローチを試す！\n'
            f'→ 「おかしいなぁ、さっきのやり方では難しいみたいですね…別の方法を試しましょう！」と切り替える\n'
            f'→ 正直に「まだRの音になっています」と言ったうえで、新しい練習方法を提案'
        )

    # R/L判定: 生徒がLを練習中にWhisperがRと認識した場合は失敗
    r_words = {"right", "rice", "read", "rock", "river", "wrong", "run", "red", "rain", "road", "ring", "room", "rest", "rich", "ride", "rise", "rope", "roll", "roof", "row"}
    l_words = {"light", "lice", "lead", "lock", "liver", "long", "love", "look", "late", "let", "lip", "low", "lazy", "lesson", "lamp", "lane", "law", "lay", "left", "lemon"}
    found_r_when_l_expected = [w for w in recognized_words if w in r_words]
    if found_r_when_l_expected:
        mismatch_warning = (
            f'\n\n⚠️⚠️⚠️ 非常に重要な判定 ⚠️⚠️⚠️\n'
            f'Whisperが以下をRの単語として認識しました: {", ".join(found_r_when_l_expected)}\n'
            f'これは生徒がLの音ではなくRの音を出しているということです！\n'
            f'Whisperは聞こえた通りに書き起こします。"right"と認識された＝実際にRの音が出ている。\n'
            f'生徒がもし "light" を言おうとしていたなら、Lの発音に失敗しています。\n'
            f'→ 絶対に「Lの音が出ています」「良い発音です」とは言わないでください！\n'
            f'→ 「Whisperには "right" と聞こえました。Lの音を出すには…」と正直にフィードバック\n'
            f'→ english_wordsには同じ単語ではなく、Lの多様な練習単語を出してください:\n'
            f'  語頭L: love, look, late, let, lip, lazy, lesson, lamp\n'
            f'  語中L: hello, yellow, believe, color, follow, million\n'
            f'  語末L: ball, call, feel, cool, bell, pull, tall, well\n'
            f'  L文: "I love long blue lakes." "Look at the light." "Lily likes lemon."\n'
            f'→ 基礎に戻って「la, le, li, lo, lu」の練習も提案してください'
        )

    if conf < 0.4:
        conf_instruction = (
            f'⚠️ 平均確信度が{conf:.0%}と非常に低いです！\n'
            f'→ 「あれ？ちょっと聞き取りにくかったです」と優しく聞き返してください\n'
            f'→ 生徒が何を言おうとしたか推測して確認\n'
            f'→ english_wordsに同じ練習単語をもう一度入れてください（再チャレンジ）\n'
            f'→ 「大丈夫ですよ！もう一度ゆっくりやってみましょう」と励ます'
        )
    elif conf < 0.6:
        conf_instruction = (
            f'確信度{conf:.0%}。一部聞き取れましたが不明瞭な部分あり。\n'
            f'→ 聞き取れた部分を褒め、不明瞭な部分を具体的に指摘'
        )
    elif conf < 0.8:
        conf_instruction = f'確信度{conf:.0%}。まあまあ良い。改善点を指摘しつつ褒める。'
    else:
        conf_instruction = (
            f'確信度{conf:.0%}。確信度は高いですが、Whisperの認識結果が目標と一致しているか必ず確認してください！\n'
            f'→ 確信度が高い＝はっきり発音できている。しかし目標と違う音が出ていれば失敗です。\n'
            f'→ 例: lightを練習中にWhisperが"right"(98%)と認識 → Rの音がクリアに出ている → Lは失敗'
        )

    prompt = (f'{hctx}Whisperの音声認識結果: "{txt}"\n平均確信度: {conf:.0%}\n{ws}'
              f'{mismatch_warning}'
              f'{fail_urgency}\n\n'
              f'{conf_instruction}\n\n'
              f'coach_messageは全て日本語！マークダウン記法(###, **)は使わないでください。\n'
              f'JSONのみ(```禁止)。{{で始め}}で終える。\n'
              f'必須: coach_message, english_words, score_feedback, pronunciation_tip, next_challenge, encouragement')
    p, raw = await _gemini_generate(gemini_pron, prompt)
    if p and "coach_message" in p:
        sessions.add_history(sid, "user", f"[発音]{txt}({conf:.0%})")
        sessions.add_history(sid, "coach", p["coach_message"])
        sessions.add_words(sid, p.get("english_words", []) + p.get("next_challenge", []))
        return p
    if p and "reply" in p:
        lr = p.get("learning",{})
        cm = lr.get("reply_ja","") or lr.get("explanation","") or p["reply"]
        sessions.add_history(sid, "user", f"[発音]{txt}({conf:.0%})")
        sessions.add_history(sid, "coach", cm)
        return {"coach_message":cm,"english_words":[],"score_feedback":f"認識:「{txt}」({int(conf*100)}%)",
                "pronunciation_tip":lr.get("explanation","もう一度ゆっくり。"),"next_challenge":[],"encouragement":"💪 頑張りましょう！"}
    fb = _fb_pron(); fb["score_feedback"] = f"認識:「{txt}」({int(conf*100)}%)"; return fb

# =====================================================================
# /chat_text — テキスト入力対応
# =====================================================================
@app.post("/chat_text")
async def chat_text(request: Request):
    d = await request.json()
    txt = d.get("text", "").strip()
    mode = d.get("lesson_mode", "conversation")
    sid = _get_sid(d)
    if not txt:
        return _fb_pron() if mode == "pronunciation" else _fb_conv()
    try:
        hctx = sessions.get_history_text(sid)
        level = sessions.estimate_level(sid)
        if mode == "pronunciation":
            return await _chat_pron(txt, 0.95, [], hctx, sid)
        return await _chat_conv(txt, 0.95, [], hctx, sid, level)
    except Exception as e:
        logger.error(f"ChatText err: {e}")
        return _fb_pron() if mode == "pronunciation" else _fb_conv()

# =====================================================================
# /debate_finish — ディベート終了まとめ（Geminiでまとめ生成）
# =====================================================================
@app.post("/debate_finish")
async def debate_finish(request: Request):
    d = await request.json()
    sid = _get_sid(d)
    result_type = d.get("result", "student_wins")  # student_wins or professor_wins
    s = sessions.get(sid)
    if not s:
        return JSONResponse({"error": "session not found"}, status_code=404)

    topic = s.get("debate_topic", "the topic")
    score = s.get("debate_score", 50)
    hctx = sessions.get_history_text(sid)
    user_stance = {"pro":"賛成派","con":"反対派","neutral":"中立派"}.get(s.get("debate_stance",""), "不明")

    if result_type == "student_wins":
        prompt = (
            f'【ディベート終了：生徒の勝利】\n'
            f'主題: 「{topic}」 生徒の立場: {user_stance}  最終スコア: {score}:{100-score}\n\n'
            f'討論履歴:\n{hctx}\n\n'
            f'あなた（Professor）は降参します。以下の要素を含む英語のまとめスピーチ（4-6文）を作成してください：\n'
            f'- まだ思うところはあるが、生徒の議論が優れていたことを認める\n'
            f'- 生徒の具体的にどの指摘・論点が素晴らしかったかを挙げて褒める（2つ以上）\n'
            f'- 今回は生徒が議論を優位に進めたことを率直に認め、降参を宣言する\n'
            f'- もしもっと良くできた点があれば1つだけ指摘する（褒めるだけでなく成長のヒントも）\n'
            f'- 生徒の理解と知恵の向上を心から祈り、引き続き頑張るよう励ます\n\n'
            f'JSONのみ(```禁止)。{{"reply":"英語のまとめ","reply_ja":"日本語訳","advice":"今後のアドバイス（日本語3-5文。良かった点と改善点を具体的に）"}}'
        )
    elif result_type == "draw":
        prompt = (
            f'【ディベート終了：引き分け（{DEBATE_N}往復の激闘の末）】\n'
            f'主題: 「{topic}」 生徒の立場: {user_stance}  最終スコア: {score}:{100-score}\n\n'
            f'討論履歴:\n{hctx}\n\n'
            f'{DEBATE_N}往復の熱戦の末、決着がつきませんでした。以下の要素を含む英語のまとめスピーチ（4-6文）を作成してください：\n'
            f'- {DEBATE_N}ラウンドにわたる白熱した議論だったことを称える\n'
            f'- 生徒の議論で良かった点を具体的に2つ挙げて褒める\n'
            f'- 改善できた点も1-2つ具体的に指摘する\n'
            f'- 互角の戦いだったことを認め、次回の再戦を楽しみにしていると励ます\n'
            f'- 今日の議論で得た学びを大切にしてほしいと伝える\n\n'
            f'JSONのみ(```禁止)。{{"reply":"英語のまとめ","reply_ja":"日本語訳","advice":"今後のアドバイス（日本語3-5文。良かった点・改善点・次回への具体的なヒント）"}}'
        )
    else:
        prompt = (
            f'【ディベート終了：Professorの勝利】\n'
            f'主題: 「{topic}」 生徒の立場: {user_stance}  最終スコア: {score}:{100-score}\n\n'
            f'討論履歴:\n{hctx}\n\n'
            f'あなた（Professor）が議論を制しました。以下の要素を含む英語のまとめスピーチ（4-6文）を作成してください：\n'
            f'- 生徒の努力と勇気を称える（見下さない）\n'
            f'- 生徒の議論のどこが弱かった・改善の余地があったかを具体的に指摘する（2つ以上）\n'
            f'- こういう論点や手法を使えばもっと良い議論ができたというアドバイスを与える\n'
            f'- 次回はさらに手強い相手になってくれることを楽しみにしていると励ます\n\n'
            f'JSONのみ(```禁止)。{{"reply":"英語のまとめ","reply_ja":"日本語訳","advice":"今後のアドバイス（日本語3-5文。弱かった点と具体的改善策を丁寧に）"}}'
        )

    p, raw = await _gemini_generate(gemini_debate, prompt)
    if p and "reply" in p:
        return {"reply": p["reply"], "reply_ja": p.get("reply_ja", ""), "advice": p.get("advice", ""),
                "result": result_type, "final_score": score}

    # フォールバック
    if result_type == "student_wins":
        return {"reply": f"I must concede — your arguments on {topic} were formidable. Your use of specific data and logical precision gave you a clear advantage. I tip my hat to you. Keep sharpening that brilliant mind!",
                "reply_ja": f"認めざるを得ません。{topic}に関するあなたの議論は見事でした。具体的なデータと論理的な精度で明確な優位を築きました。脱帽です。その素晴らしい知性を磨き続けてください！",
                "advice": "素晴らしい討論でした！具体的なデータや事例を効果的に使い、議論を有利に進めました。今後はさらに多角的な視点からの反論も準備しておくと、より盤石な議論ができるでしょう。",
                "result": result_type, "final_score": score}
    elif result_type == "draw":
        return {"reply": f"What a battle! After {DEBATE_N} rounds of fierce intellectual combat on {topic}, neither of us could land the decisive blow. Your arguments were sharp and well-constructed. Next time, push even harder with concrete data, and you might just tip the scales. I eagerly await our rematch!",
                "reply_ja": f"なんという激闘でしょう！{topic}について{DEBATE_N}ラウンドの知的格闘の末、どちらも決定打を放てませんでした。あなたの論点は鋭く構成も見事でした。次回はさらに具体的なデータで攻めれば、天秤を傾けられるかもしれません。再戦を心待ちにしています！",
                "advice": f"{DEBATE_N}往復の激闘、お疲れ様でした！互角の戦いは実力の証です。さらに上を目指すには、相手の論点を認めた上で切り返す「譲歩→反論」のテクニックが有効です。具体的な数字や研究名をもう1つ2つ増やすと、均衡を崩す一手になるでしょう。",
                "result": result_type, "final_score": score}
    else:
        return {"reply": f"A spirited debate on {topic}! You showed real courage engaging with these complex ideas. With stronger evidence and more precise rebuttals, you'll be an even more formidable opponent. I look forward to our next intellectual battle!",
                "reply_ja": f"{topic}について熱い討論でした！複雑なテーマに果敢に挑む姿勢は素晴らしかったです。より強い根拠と精密な反論があれば、さらに手強い相手になるでしょう。次の知的バトルを楽しみにしています！",
                "advice": "果敢な挑戦でした！主張に具体的なデータや研究名を添えることで説得力が大きく増します。また、相手の論点に直接反論してから自分の主張を展開する『反論→主張』の構成を意識すると効果的です。",
                "result": result_type, "final_score": score}

# =====================================================================
# /summary — レッスン終了まとめ
# =====================================================================
@app.post("/summary")
async def lesson_summary(request: Request):
    d = await request.json()
    sid = _get_sid(d)
    s = sessions.get(sid)
    if not s:
        return JSONResponse({"error": "session not found"}, status_code=404)
    mode = s.get("lesson_mode", "conversation")
    hctx = sessions.get_history_text(sid)
    scores = s.get("scores", [])
    wordbook = s.get("wordbook", [])
    turns = s.get("turn_count", 0)
    level = sessions.estimate_level(sid)

    if mode == "pronunciation":
        score_summary = ""
        if scores:
            avg = sum(sc["confidence"] for sc in scores) / len(scores)
            best = max(scores, key=lambda x: x["confidence"])
            worst = min(scores, key=lambda x: x["confidence"])
            score_summary = (f"総発音回数: {len(scores)}回\n平均スコア: {avg:.0%}\n"
                           f"最高: \"{best['word']}\" ({best['confidence']:.0%})\n"
                           f"最低: \"{worst['word']}\" ({worst['confidence']:.0%})")
        prompt = (
            f'{hctx}\n\nレッスンが終了しました。以下の情報でまとめを生成してください（全て日本語）:\n'
            f'ターン数: {turns}\n{score_summary}\n'
            f'練習した単語: {", ".join(wordbook[:20]) if wordbook else "なし"}\n\n'
            f'JSONのみ(```禁止)。{{で始め}}で終える。\n'
            f'{{"summary":"レッスン全体の総括(日本語)","strengths":"良かった点","weaknesses":"改善点","tips":"次回への具体的アドバイス","encouragement":"励まし(絵文字付き)"}}'
        )
        p, _ = await _gemini_generate(gemini_pron, prompt)
    else:
        prompt = (
            f'{hctx}\n\nレッスンが終了しました。以下の情報でまとめを生成してください（全て日本語）:\n'
            f'ターン数: {turns}\n推定レベル: CEFR {level}\n'
            f'学んだ表現: {", ".join(wordbook[:20]) if wordbook else "なし"}\n\n'
            f'JSONのみ(```禁止)。{{で始め}}で終える。\n'
            f'{{"summary":"レッスン全体の総括(日本語)","strengths":"良かった点","weaknesses":"改善点","grammar_notes":"文法の注意点","new_expressions":"今日学んだ表現まとめ","tips":"次回への具体的アドバイス","encouragement":"励まし(絵文字付き)"}}'
        )
        p, _ = await _gemini_generate(gemini_conv, prompt)

    if p:
        p["turn_count"] = turns
        p["estimated_level"] = level
        p["wordbook"] = wordbook[:30]
        p["lesson_mode"] = mode
        if scores:
            p["score_data"] = [{"word": sc["word"], "confidence": sc["confidence"]} for sc in scores]
        return p
    return {
        "summary": f"お疲れ様でした！今日は{turns}ターンの練習をしました。",
        "strengths": "レッスンに参加したこと自体が素晴らしいです！",
        "weaknesses": "次回もっと詳しく分析します。",
        "tips": "毎日少しずつ練習を続けましょう！",
        "encouragement": "🌟 Great effort today! 次回も頑張りましょう！",
        "turn_count": turns, "estimated_level": level,
        "wordbook": wordbook[:30], "lesson_mode": mode,
    }

# =====================================================================
# /wordbook — 今日の単語帳
# =====================================================================
@app.post("/wordbook")
async def get_wordbook(request: Request):
    d = await request.json()
    sid = _get_sid(d)
    s = sessions.get(sid)
    if not s:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return {"wordbook": s.get("wordbook", []), "count": len(s.get("wordbook", []))}

# =====================================================================
# /score_history — 発音スコア推移
# =====================================================================
@app.post("/score_history")
async def score_history(request: Request):
    d = await request.json()
    sid = _get_sid(d)
    s = sessions.get(sid)
    if not s:
        return JSONResponse({"error": "session not found"}, status_code=404)
    scores = s.get("scores", [])
    return {"scores": scores, "count": len(scores)}

# =====================================================================
# /minimal_pair_quiz
# =====================================================================
@app.post("/minimal_pair_quiz")
async def minimal_pair_quiz(request: Request):
    d = await request.json()
    category = d.get("category", "R vs L")
    prompt = (
        f'英語のミニマルペアクイズを生成してください。カテゴリ: {category}\n\n'
        f'ミニマルペアとは、1つの音だけ異なる2つの単語のペアです。\n'
        f'5組のミニマルペアを出題してください。\n\n'
        f'JSONのみ(```禁止)。{{で始め}}で終える。\n'
        f'{{"pairs":[{{"word_a":"単語A","word_b":"単語B","hint":"日本語でのヒント","explanation":"この2つの違いの解説(日本語)"}}],"category":"{category}","intro":"クイズの導入文(日本語)"}}'
    )
    p, _ = await _gemini_generate(gemini_pron, prompt)
    if p and "pairs" in p:
        return p
    return {
        "pairs": [
            {"word_a": "right", "word_b": "light", "hint": "Rは舌が天井に触れない", "explanation": "rightのRは舌を巻き、lightのLは舌先を歯茎につけます"},
            {"word_a": "read", "word_b": "lead", "hint": "最初の音に集中", "explanation": "readのRとleadのLを聞き分けましょう"},
            {"word_a": "rock", "word_b": "lock", "hint": "口の形が違う", "explanation": "rockのRは唇を丸め、lockのLは舌先を上げます"},
            {"word_a": "rice", "word_b": "lice", "hint": "間違えると大変！", "explanation": "riceはお米、liceはシラミ"},
            {"word_a": "river", "word_b": "liver", "hint": "2音節の聞き分け", "explanation": "riverは川、liverは肝臓"},
        ],
        "category": category,
        "intro": f"「{category}」のミニマルペアクイズです！"
    }

# =====================================================================
# HTML
# =====================================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    p = Path(__file__).parent / "english_coach_whisper.html"
    return p.read_text(encoding="utf-8") if p.exists() else HTMLResponse("<h1>HTML not found</h1>", 404)

if __name__ == "__main__":
    import uvicorn
    print("\n  AI English Coach v6 — http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
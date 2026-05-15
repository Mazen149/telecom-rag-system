import os
import re
import json
import hashlib
import math
from collections import Counter
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

class TelecomRAG:
    def __init__(self, data_path="data", embedding_model="BAAI/bge-m3", cache_dir="cache", cache_version="v1"):
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=self.groq_api_key)
        self.data_path = data_path
        self.embedding_model_name = embedding_model
        self.cache_dir = cache_dir
        self.cache_version = cache_version
        self.cache_namespace = self._build_cache_namespace()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print("Initializing TelecomRAG...")
        self.model = SentenceTransformer(self.embedding_model_name)
        
        self.all_chunks = []
        self.metadata = []
        self.index = None
        self._doc_tf = []
        self._doc_len = []
        self._df = {}
        self._avg_doc_len = 0.0
        self._lexical_ready = False
        
        self._prepare_system()

    def _build_cache_namespace(self):
        raw = f"{os.path.abspath(self.data_path)}::{self.embedding_model_name}::{self.cache_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _cache_paths(self):
        base = os.path.join(self.cache_dir, self.cache_namespace)
        return {
            "meta": f"{base}.meta.json",
            "chunks": f"{base}.chunks.json",
            "index": f"{base}.index"
        }

    def _compute_corpus_fingerprint(self):
        if not os.path.exists(self.data_path):
            return ""

        items = []
        for file in sorted(os.listdir(self.data_path)):
            if not file.endswith(".md"):
                continue
            file_path = os.path.join(self.data_path, file)
            try:
                stat = os.stat(file_path)
            except OSError:
                continue
            items.append({
                "path": file,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns
            })

        raw = json.dumps(items, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _try_load_cache(self, fingerprint):
        paths = self._cache_paths()
        if not (os.path.exists(paths["meta"]) and os.path.exists(paths["chunks"]) and os.path.exists(paths["index"])):
            return False

        try:
            with open(paths["meta"], "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") != fingerprint:
                return False
            if meta.get("embedding_model") != self.embedding_model_name:
                return False
            if meta.get("cache_version") != self.cache_version:
                return False

            with open(paths["chunks"], "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.all_chunks = payload.get("chunks", [])
            self.metadata = payload.get("metadata", [])
            self.index = faiss.read_index(paths["index"])

            if self.index is None or not self.all_chunks:
                return False

            print(f"Loaded FAISS cache with {self.index.ntotal} vectors.")
            return True
        except Exception as e:
            print(f"Cache load failed: {e}")
            self.all_chunks = []
            self.metadata = []
            self.index = None
            return False

    def _save_cache(self, fingerprint):
        if not fingerprint or self.index is None or not self.all_chunks:
            return

        paths = self._cache_paths()
        meta = {
            "fingerprint": fingerprint,
            "embedding_model": self.embedding_model_name,
            "cache_version": self.cache_version,
            "data_path": self.data_path
        }

        try:
            with open(paths["meta"], "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            with open(paths["chunks"], "w", encoding="utf-8") as f:
                json.dump({"chunks": self.all_chunks, "metadata": self.metadata}, f)
            faiss.write_index(self.index, paths["index"])
            print("Cache saved.")
        except Exception as e:
            print(f"Cache save failed: {e}")

    def _prepare_system(self):
        """Loads data, chunks it, creates embeddings, and builds the FAISS index."""
        if not os.path.exists(self.data_path):
            print(f"Warning: Data path {self.data_path} not found.")
            return

        fingerprint = self._compute_corpus_fingerprint()
        if fingerprint and self._try_load_cache(fingerprint):
            self._build_lexical_index()
            return

        # 1. Load and chunk documents
        for file in os.listdir(self.data_path):
            if file.endswith(".md"):
                file_path = os.path.join(self.data_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                doc_chunks = self.chunk_text(text)
                for chunk in doc_chunks:
                    self.all_chunks.append(chunk)
                    self.metadata.append({"source": file})

        if not self.all_chunks:
            print("No data found to index.")
            return

        # 2. Create embeddings
        print(f"Creating embeddings for {len(self.all_chunks)} chunks...")
        embeddings = self.model.encode(
            self.all_chunks,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings = np.array(embeddings).astype("float32")

        # 3. Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors.")
        self._save_cache(fingerprint)
        self._build_lexical_index()

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def _build_lexical_index(self):
        self._doc_tf = []
        self._doc_len = []
        df = Counter()

        for chunk in self.all_chunks:
            tokens = self._tokenize(chunk)
            tf = Counter(tokens)
            self._doc_tf.append(tf)
            self._doc_len.append(len(tokens))
            df.update(set(tf.keys()))

        self._df = dict(df)
        total_len = sum(self._doc_len)
        self._avg_doc_len = (total_len / len(self._doc_len)) if self._doc_len else 0.0
        self._lexical_ready = True

    def _get_rewrite_topics(self, limit=25):
        if not self.metadata:
            return []

        topics = []
        seen = set()
        for meta in self.metadata:
            source = meta.get("source", "")
            if not source:
                continue
            title = os.path.splitext(source)[0]
            title = title.replace("_", " ").replace("-", " ").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            topics.append(title)
            if len(topics) >= limit:
                break

        return topics

    def _rewrite_query_llm(self, query):
        original = query.strip()
        if not original:
            return original

        topics = self._get_rewrite_topics()
        topics_text = ", ".join(topics) if topics else "NileTel telecom support"

        system_prompt = (
            "You are a query rewriting assistant for a telecom support RAG. "
            "Rewrite the user query to improve retrieval. Keep the same language "
            "and intent, add relevant telecom keywords if implied, and keep it short. "
            "Return only the rewritten query, no quotes, no explanations."
        )

        user_prompt = (
            f"User query: {original}\n"
            f"Available topics: {topics_text}\n"
            "Rewrite the query for better search."
        )

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=80
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten else original
        except Exception as e:
            print(f"LLM query rewrite failed: {e}")
            return original

    def _answer_implies_action(self, answer_text):
        if not answer_text:
            return False

        text = answer_text.lower()
        text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
        text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        text = text.replace("ى", "ي")
        text = re.sub(r"\s+", " ", text).strip()

        phrase_hits = [
            "هنبعت مهندس", "هنبعت فني", "هنرسل مهندس", "هندب مهندس",
            "هنفتح تذكرة", "هنعمل تذكرة", "هنرفع تذكرة", "هرفع تذكرة",
            "هنسجل شكوى", "هنرفع شكوى", "هنصعد التذكرة", "هصعد التذكرة",
            "ticket created", "open a ticket", "raise a ticket", "create a ticket",
            "dispatch an engineer", "send an engineer", "engineer will be dispatched"
        ]

        if any(phrase in text for phrase in phrase_hits):
            return True

        ticket_regex = r"(هن|سوف|سيتم)\s+(فتح|رفع|عمل|انشاء|تسجيل|تصعيد)\s+(تذكره|تذكرة|شكوى)"
        engineer_regex = r"(هن|سوف|سيتم)\s+(ارسال|بعث|توجيه|ندب)\s+(مهندس|فني)"

        if re.search(ticket_regex, text) or re.search(engineer_regex, text):
            return True

        return False

    def _ticket_fixed_message(self):
        return "تمام يا فندم، تم تسجيل طلبك لفتح تذكرة وسيتم التواصل معك في أقرب وقت ممكن."

    def _bm25_scores(self, query_tokens, k1=1.5, b=0.75):
        if not self._lexical_ready:
            self._build_lexical_index()

        if not query_tokens or not self._doc_tf:
            return []

        num_docs = len(self._doc_tf)
        avgdl = self._avg_doc_len if self._avg_doc_len > 0 else 1.0
        q_tf = Counter(query_tokens)
        scores = [0.0] * num_docs

        for term, qf in q_tf.items():
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._doc_tf):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + k1 * (1 - b + b * (self._doc_len[i] / avgdl))
                scores[i] += idf * (f * (k1 + 1) / denom) * qf

        return scores

    def _match_filters(self, meta, filters):
        if not filters:
            return True
        for key, value in filters.items():
            if key not in meta:
                return False
            if isinstance(value, (list, tuple, set)):
                if meta.get(key) not in value:
                    return False
            else:
                if meta.get(key) != value:
                    return False
        return True

    def _apply_meta_filter(self, results, filters):
        if not filters:
            return results
        filtered = []
        for res in results:
            idx = res.get("idx")
            if idx is None or idx >= len(self.metadata):
                continue
            if self._match_filters(self.metadata[idx], filters):
                filtered.append(res)
        return filtered

    def chunk_text(self, text, max_len=700, overlap=120):
        """Splits text into paragraph-based chunks with overlap."""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            projected_len = len(current_chunk) + len(para) + (2 if current_chunk else 0)
            if projected_len > max_len:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    if overlap > 0:
                        tail = current_chunk[-overlap:]
                        current_chunk = tail + ("\n\n" if tail else "") + para
                    else:
                        current_chunk = para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def retrieve(self, query, top_k=6, filters=None, semantic_k=None, keyword_k=None, rrf_k=60):
        """Retrieves relevant chunks using hybrid (semantic + keyword) search with RRF fusion."""
        if self.index is None or not self.all_chunks:
            return []

        if semantic_k is None:
            semantic_k = max(top_k * 3, top_k)
        if keyword_k is None:
            keyword_k = max(top_k * 3, top_k)

        semantic_results = []
        rewritten_query = self._rewrite_query_llm(query)
        if rewritten_query and rewritten_query != query:
            query_text = f"{query} {rewritten_query}"
        else:
            query_text = query

        query_emb = self.model.encode([query_text], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype("float32")
        distances, indices = self.index.search(query_emb, semantic_k)
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.all_chunks) and score > 0.4:
                semantic_results.append({
                    "idx": int(idx),
                    "text": self.all_chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "score": float(score)
                })

        query_tokens = self._tokenize(query_text)
        keyword_results = []
        scores = self._bm25_scores(query_tokens)
        if scores:
            ranked = sorted(
                ((i, s) for i, s in enumerate(scores) if s > 0.0),
                key=lambda x: x[1],
                reverse=True
            )[:keyword_k]
            for idx, score in ranked:
                keyword_results.append({
                    "idx": int(idx),
                    "text": self.all_chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "score": float(score)
                })

        semantic_results = self._apply_meta_filter(semantic_results, filters)
        keyword_results = self._apply_meta_filter(keyword_results, filters)

        combined_scores = {}
        for rank, res in enumerate(semantic_results, start=1):
            idx = res["idx"]
            combined_scores[idx] = combined_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        for rank, res in enumerate(keyword_results, start=1):
            idx = res["idx"]
            combined_scores[idx] = combined_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        ranked_idx = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked_idx:
            if idx < len(self.all_chunks):
                results.append({
                    "text": self.all_chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "score": float(score)
                })

        return results

    def route_query(self, query):
        """
        Decides how to handle the user query before sending it to the LLM.
        Uses a hybrid approach: rule-based first, then LLM for complex cases.
        """
        q = query.strip()
        q_norm = re.sub(r'[^\w\s]', '', q) # remove punctuation
        q_norm = re.sub(r'\s+', ' ', q_norm).strip()
        
        # 1. Rule-based routing
        greetings = [
            "ازيك", "ازيكك", "عامل ايه", "عامل اي", "اخبارك", "أخبارك",
            "عامله ايه", "عامله اي", "كيفك", "كيف الحال", "شو اخبارك",
            "مرحبا", "اهلا", "أهلا", "اهلا بيك", "أهلاً", "هلا", "هلا والله",
            "اهلين", "أهلين", "يا هلا", "نورت",
            "السلام عليكم", "وعليكم السلام", "سلام", "سلام عليكم",
            "السلام", "سلامو", "سلامات",
            "صباح الخير", "صباح النور", "صباحو",
            "مساء الخير", "مساء النور",
            "hello", "hi", "hey", "hey there", "hola",
            "good morning", "good evening", "good afternoon",
            "yo", "sup", "what's up", "whats up"
        ]
        
        tokens = q_norm.split()
        is_greeting_exact = any(g == q_norm for g in greetings)
        is_greeting_short = len(tokens) <= 2 and any(token in greetings for token in tokens)
        if is_greeting_exact or is_greeting_short:
            return "greeting"
        
        out_of_scope = [
            "سياسة", "سياسي", "حكومة", "رئيس", "وزير", "انتخابات", "برلمان",
            "رياضة", "كورة", "كرة", "ماتش", "مباراة", "لاعب", "مدرب",
            "دوري", "كاس", "بطولة", "هدف", "جون",
            "اكل", "أكل", "طبخ", "وصفة", "وصفات", "مطعم", "اكلات",
            "فطار", "غدا", "عشا", "عشاء", "حلويات", "مشروب",
            "مسلسل", "فيلم", "افلام", "أفلام", "ممثل", "ممثلة",
            "اغاني", "أغاني", "اغنية", "أغنية", "مغني", "موسيقى",
            "اقتصاد", "بورصة", "دولار", "ذهب", "اسهم", "أسهم",
            "عملة", "بنك", "تجارة", "استثمار",
            "طقس", "جو", "حر", "برد", "مطر",
            "سفر", "رحلة", "فندق", "طيران",
            "لعبة", "العاب", "ألعاب", "جيم", "game"
        ]
        out_of_scope_hits = [token for token in tokens if token in out_of_scope]
        out_of_scope_ratio = (len(out_of_scope_hits) / len(tokens)) if tokens else 0.0
        is_out_of_scope_short = len(tokens) <= 3 and len(out_of_scope_hits) > 0
        is_out_of_scope_heavy = out_of_scope_ratio >= 0.5
        if is_out_of_scope_short or is_out_of_scope_heavy:
            return "out_of_scope"

        ticket_keywords = [
            "تصعيد", "مهندس", "تذكرة", "شكوى", "فني", "مندوب",
            "صيانة", "عطل", "بايظ", "مقطوع",
            "عمل تذكرة", "رفع تذكرة", "ابعت مهندس",
            "ابعت فني", "عايز اشتكي", "سجل شكوى",
            "النت فاصل", "النت قاطع"
        ]

        ticket_phrases = [k for k in ticket_keywords if " " in k]
        ticket_terms = [k for k in ticket_keywords if " " not in k]

        if any(phrase in q_norm for phrase in ticket_phrases):
            return "ticket"

        ticket_term_hits = [token for token in tokens if token in ticket_terms]
        is_ticket_short = len(tokens) <= 3 and len(ticket_term_hits) > 0
        if is_ticket_short:
            return "ticket"

        # 2. LLM-based routing for ambiguous cases
        try:
            routing_prompt = f"""أنت نظام توجيه ذكي لطلبات عملاء شركة اتصالات.
                                صنف الرسالة التالية إلى واحدة من الفئات الثلاث فقط:
                                - "chat": إذا كانت الرسالة سؤال عادي عن خدمات الاتصالات، باقات، إنترنت، فواتير، أو دردشة عامة.
                                - "out_of_scope": إذا كانت الرسالة خارج نطاق الاتصالات تماماً (مثل سياسة، رياضة، فن، أسئلة طبية).
                                - "ticket": إذا كان المستخدم يطلب صراحة تدخل بشري، إرسال فني/مهندس، تصعيد مشكلة، أو فتح تذكرة عطل.

                                الرسالة: "{query}"

                                الفئة (كلمة واحدة فقط):"""
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": routing_prompt}],
                temperature=0.0,
                max_tokens=10
            )
            category = response.choices[0].message.content.strip().lower()
            if "out_of_scope" in category:
                return "out_of_scope"
            elif "ticket" in category:
                return "ticket"
            else:
                return "chat"
        except Exception as e:
            print(f"LLM routing failed: {e}")
            return "chat"

    def generate_answer(self, query, retrieved_results, route="chat"):
        """Takes the user query and the retrieved chunks, sends it to Groq, and returns the final answer."""
        if not retrieved_results:
            if route == "ticket":
                return {
                    "answer": self._ticket_fixed_message(),
                    "needs_action": "YES",
                    "sources": []
                }
            return {
                "answer": "مش متأكد من البيانات المتاحة يا فندم بخصوص الموضوع ده.",
                "needs_action": "NO",
                "sources": []
            }

        context = "\n\n".join([
            f"Source: {res['source']}\n{res['text']}" 
            for res in retrieved_results
        ])

        system_prompt = """أنت مساعد دعم عملاء محترف في شركة NileTel للاتصالات.

                            قواعد صارمة:
                            - أجب باللهجة المصرية الطبيعية وبلباقة (يا فندم، تمام، هنحلها، تحت أمرك...).
                            - استخدم فقط المعلومات الموجودة في السياق. ممنوع التأليف أو الاستنتاج من خارج السياق.
                            - لا تختلق أرقام تذاكر أو تفاصيل وهمية على الإطلاق.
                            - مهم جداً: حافظ على تناسق النص من اليمين لليسار. عند استخدام كلمات إنجليزية داخل النص العربي، اكتبها بشكل سليم بحيث لا يختل ترتيب القراءة. 
                            - قدم إجابات مفيدة وواضحة، واستخدم النقاط (Bullet points) لتنظيم الخطوات أو التفاصيل.
                            - يجب أن يكون الرد بالهيكل التالي فقط (دون أي إضافات):
                            answer: [ضع إجابتك المنسقة هنا]
                            needs_action: [YES أو NO]

                            حيث تكون needs_action كالتالي:
                            - إذا طلب المستخدم صراحة أو ضمناً إنشاء تذكرة، رفع تذكرة، إرسال مهندس، تصعيد المشكلة، أو طلب تدخل بشري → YES
                            - إذا كان السؤال استفسار معلوماتي فقط أو محادثة عادية → NO"""

        user_prompt = f"""السياق المتاح (استخدمه فقط):
                            {context}

                            السؤال: {query}"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        text = response.choices[0].message.content.strip()
        
        # Robust parsing for needs_action
        match = re.search(r"needs[_\s]*action\s*:\s*([a-zA-Zأ-ي]+)", text, flags=re.IGNORECASE)
        if match:
            val = match.group(1).upper()
            needs_action = "YES" if val in ["YES", "نعم"] else "NO"
        else:
            needs_action = "NO"
            
        # Clean the answer text
        clean_answer = re.sub(r"needs[_\s]*action\s*:.*", "", text, flags=re.IGNORECASE)
        clean_answer = re.sub(r"^(answer|الإجابة)\s*:\s*", "", clean_answer, flags=re.IGNORECASE).strip()
        
        # Fallbacks: if route is ticket, force action. Or if answer implies action
        if route == "ticket" or self._answer_implies_action(clean_answer):
            needs_action = "YES"

        if needs_action == "YES":
            return {
                "answer": self._ticket_fixed_message(),
                "needs_action": "YES",
                "sources": []
            }

        sources = list(set([res["source"] for res in retrieved_results]))

        return {
            "answer": clean_answer,
            "needs_action": needs_action,
            "sources": sources
        }

    def run_rag_pipeline(self, query: str):
        """Full RAG pipeline flow."""
        print(f"\nProcessing query: {query}")
        
        route = self.route_query(query)

        if route == "out_of_scope":
            return {
                "answer": "عذراً يا فندم، أنا متخصص في خدمات NileTel للاتصالات فقط. منقدرش نساعد في مواضيع تانية خارج التخصص ده.",
                "needs_action": "NO",
                "sources": []
            }

        if route == "greeting":
            return {
                "answer": "أهلاً بيك يا فندم في NileTel، أقدر أساعدك إزاي النهارده؟",
                "needs_action": "NO",
                "sources": []
            }

        # Normal flow
        results = self.retrieve(query, top_k=6)
        response = self.generate_answer(query, results, route=route)
        
        return response

if __name__ == "__main__":
    # Test the class
    rag = TelecomRAG()
    test_queries = [
        "ازاي أحل مشكلة 5G throttling؟",
        "النت مقطوع تماماً في المنصورة، اعمل تذكرة",
        "ازيك"
    ]

    for q in test_queries:
        print(f"\n--- Testing: {q} ---")
        res = rag.run_rag_pipeline(q)
        print(f"Answer: {res['answer']}")
        print(f"Needs Action: {res['needs_action']}")
        print(f"Sources: {res['sources']}")
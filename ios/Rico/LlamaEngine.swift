import Foundation
import llama

/// Обёртка над llama.cpp: загрузка GGUF, шаблон чата, потоковая генерация, переиспользование KV-кэша.
final class LlamaEngine {
    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var vocab: OpaquePointer?
    private var tmpl: String?
    private var cached: [llama_token] = []
    private(set) var loadedName: String?
    var onProgress: ((Float) -> Void)?
    private static var backendReady = false

    private static let stops = ["<turn|>", "<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end|>"]

    /// Грамматика JSON из llama.cpp (grammars/json.gbnf): ответ гарантированно будет JSON-объектом.
    private static let jsonGBNF = #"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws
ws ::= | " " | "\n" [ \t]{0,20}
"""#

    private func fail(_ msg: String) -> NSError {
        NSError(domain: "rico.llm", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }

    func unload() {
        if let c = ctx { llama_free(c) }
        if let m = model { llama_model_free(m) }
        ctx = nil; model = nil; vocab = nil; tmpl = nil; cached = []; loadedName = nil
    }

    func load(path: String, name: String) throws {
        unload()
        if !LlamaEngine.backendReady { llama_backend_init(); LlamaEngine.backendReady = true }
        var mp = llama_model_default_params()
        mp.n_gpu_layers = 999
        mp.progress_callback = { p, ud in
            if let ud = ud { Unmanaged<LlamaEngine>.fromOpaque(ud).takeUnretainedValue().onProgress?(p) }
            return true
        }
        mp.progress_callback_user_data = Unmanaged.passUnretained(self).toOpaque()
        var m = llama_model_load_from_file(path, mp)
        if m == nil {
            mp.n_gpu_layers = 0
            m = llama_model_load_from_file(path, mp)
        }
        guard let mdl = m else {
            throw fail("Не удалось загрузить модель. Файл повреждён, формат не поддерживается или модели не хватает памяти. Попробуй версию Q4_K_M поменьше.")
        }
        var cp = llama_context_default_params()
        cp.n_ctx = 4096
        cp.n_batch = 512
        cp.n_ubatch = 512
        let th = Int32(max(2, min(6, ProcessInfo.processInfo.activeProcessorCount - 2)))
        cp.n_threads = th
        cp.n_threads_batch = th
        var c = llama_init_from_model(mdl, cp)
        if c == nil { cp.n_ctx = 2048; c = llama_init_from_model(mdl, cp) }
        guard let cx = c else {
            llama_model_free(mdl)
            throw fail("Модель загрузилась, но не хватило памяти для её работы. Нужна модель поменьше.")
        }
        model = mdl
        ctx = cx
        vocab = llama_model_get_vocab(mdl)
        if let t = llama_model_chat_template(mdl, nil) { tmpl = String(cString: t) }
        loadedName = name
    }

    // MARK: - Токены

    private func tokenize(_ text: String, addSpecial: Bool) -> [llama_token] {
        guard let vocab = vocab else { return [] }
        let n = Int32(text.utf8.count)
        var toks = [llama_token](repeating: 0, count: Int(n) + 16)
        var r = llama_tokenize(vocab, text, n, &toks, Int32(toks.count), addSpecial, true)
        if r < 0 {
            toks = [llama_token](repeating: 0, count: Int(-r))
            r = llama_tokenize(vocab, text, n, &toks, Int32(toks.count), addSpecial, true)
        }
        return r > 0 ? Array(toks[0..<Int(r)]) : []
    }

    private func piece(_ t: llama_token) -> [UInt8] {
        guard let vocab = vocab else { return [] }
        var buf = [CChar](repeating: 0, count: 64)
        var n = llama_token_to_piece(vocab, t, &buf, Int32(buf.count), 0, false)
        if n < 0 {
            buf = [CChar](repeating: 0, count: Int(-n))
            n = llama_token_to_piece(vocab, t, &buf, Int32(buf.count), 0, false)
        }
        return n > 0 ? buf[0..<Int(n)].map { UInt8(bitPattern: $0) } : []
    }

    private func isSingleSpecial(_ s: String) -> Bool { tokenize(s, addSpecial: false).count == 1 }

    // MARK: - Шаблон чата

    private func applyTemplate(_ t: String, _ msgs: [(String, String)]) -> String? {
        var ptrs: [UnsafeMutablePointer<CChar>] = []
        defer { ptrs.forEach { free($0) } }
        var chat: [llama_chat_message] = []
        for (r, c) in msgs {
            guard let rp = strdup(r), let cp = strdup(c) else { return nil }
            ptrs.append(rp); ptrs.append(cp)
            chat.append(llama_chat_message(role: UnsafePointer(rp), content: UnsafePointer(cp)))
        }
        var cap = msgs.reduce(1024) { $0 + $1.0.utf8.count + $1.1.utf8.count } * 2
        for _ in 0..<2 {
            var buf = [CChar](repeating: 0, count: cap)
            let n: Int32 = chat.withUnsafeBufferPointer { cb in
                buf.withUnsafeMutableBufferPointer { bb in
                    t.withCString { llama_chat_apply_template($0, cb.baseAddress, cb.count, true, bb.baseAddress, Int32(cap)) }
                }
            }
            if n < 0 { return nil }
            if Int(n) <= cap { return String(decoding: buf[0..<Int(n)].map { UInt8(bitPattern: $0) }, as: UTF8.self) }
            cap = Int(n) + 16
        }
        return nil
    }

    private func buildPrompt(_ msgs: [(String, String)]) -> String {
        // Gemma 4 размечает реплики токенами <|turn> … <turn|>
        if (tmpl?.contains("<|turn>") ?? false) || (isSingleSpecial("<|turn>") && isSingleSpecial("<turn|>")) {
            var s = ""
            for (r, c) in msgs {
                let role = r == "assistant" ? "model" : r
                s += "<|turn>\(role)\n\(c)<turn|>\n"
            }
            return s + "<|turn>model\n"
        }
        if let t = tmpl, let p = applyTemplate(t, msgs) { return p }
        if isSingleSpecial("<start_of_turn>"), let p = applyTemplate("gemma", msgs) { return p }
        if let p = applyTemplate("chatml", msgs) { return p }
        return msgs.map { "\($0.0): \($0.1)" }.joined(separator: "\n") + "\nassistant: "
    }

    // MARK: - Генерация

    func generate(messages: [(String, String)], maxTokens: Int, temp: Float, json: Bool,
                  cancel: () -> Bool, emit: (String) -> Void) throws -> Double {
        guard let ctx = ctx, let vocab = vocab else { throw fail("Модель не загружена") }
        var input = tokenize(buildPrompt(messages), addSpecial: true)
        guard !input.isEmpty else { throw fail("Пустой запрос") }
        let nctx = Int(llama_n_ctx(ctx))
        let limit = max(64, nctx - maxTokens - 16)
        if input.count > limit { input = Array(input.prefix(1)) + Array(input.suffix(limit - 1)) }

        // Переиспользуем общий префикс прошлого запроса (характер Рико и т.п. не пересчитываются)
        var common = 0
        while common < cached.count && common < input.count && cached[common] == input[common] { common += 1 }
        if common >= input.count { common = input.count - 1 }
        let mem = llama_get_memory(ctx)
        if common < cached.count {
            if !llama_memory_seq_rm(mem, 0, Int32(common), -1) { llama_memory_clear(mem, true); common = 0 }
        }
        cached = Array(input.prefix(common))
        var i = common
        while i < input.count {
            let n = min(512, input.count - i)
            var chunk = Array(input[i..<(i + n)])
            let rc = chunk.withUnsafeMutableBufferPointer { llama_decode(ctx, llama_batch_get_one($0.baseAddress, Int32(n))) }
            if rc != 0 { llama_memory_clear(mem, true); cached = []; throw fail("Ошибка обработки запроса (\(rc))") }
            cached += chunk
            i += n
        }

        guard let sp = llama_sampler_chain_init(llama_sampler_chain_default_params()) else { throw fail("Сэмплер не создан") }
        defer { llama_sampler_free(sp) }
        if json, let g = llama_sampler_init_grammar(vocab, LlamaEngine.jsonGBNF, "root") { llama_sampler_chain_add(sp, g) }
        llama_sampler_chain_add(sp, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, 1.1, 0.0, 0.0))
        llama_sampler_chain_add(sp, llama_sampler_init_top_k(40))
        llama_sampler_chain_add(sp, llama_sampler_init_top_p(0.95, 1))
        llama_sampler_chain_add(sp, llama_sampler_init_temp(temp))
        llama_sampler_chain_add(sp, llama_sampler_init_dist(UInt32.random(in: 1..<UInt32.max)))

        var pending: [UInt8] = []
        var out = ""
        var produced = 0
        let t0 = Date()
        for _ in 0..<maxTokens {
            if cancel() { break }
            let tok = llama_sampler_sample(sp, ctx, -1)
            if llama_vocab_is_eog(vocab, tok) { break }
            produced += 1
            pending += piece(tok)
            if let s = String(bytes: pending, encoding: .utf8) {
                pending = []
                let before = out
                out += s
                if let stop = LlamaEngine.stops.first(where: { out.contains($0) }), let r = out.range(of: stop) {
                    let safe = String(out[..<r.lowerBound])
                    if safe.count > before.count { emit(String(safe.dropFirst(before.count))) }
                    break
                }
                emit(s)
            } else if pending.count > 8 {
                emit(String(decoding: pending, as: UTF8.self)); pending = []
            }
            var t = tok
            if llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0 { break }
            cached.append(tok)
        }
        let dt = Date().timeIntervalSince(t0)
        return dt > 0 ? Double(produced) / dt : 0
    }
}

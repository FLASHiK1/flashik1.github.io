import UIKit
import WebKit
import UniformTypeIdentifiers

/// Прокси, чтобы WKUserContentController не держал контроллер сильной ссылкой.
private final class WeakHandler: NSObject, WKScriptMessageHandler {
    weak var target: WKScriptMessageHandler?
    init(_ t: WKScriptMessageHandler) { target = t }
    func userContentController(_ c: WKUserContentController, didReceive m: WKScriptMessage) {
        target?.userContentController(c, didReceive: m)
    }
}

final class GameViewController: UIViewController, WKScriptMessageHandler, UIDocumentPickerDelegate {
    private var web: WKWebView!
    private let engine = LlamaEngine()
    private let queue = DispatchQueue(label: "rico.llm", qos: .userInitiated)
    private var loading: String?
    private var progress: Float?
    private var lastError: String?
    private var scopedURL: URL?
    private var autoTried = false
    private var cancelGen = false
    private let hud = UILabel()
    private var hudHide: DispatchWorkItem?
    private var pickCopy = false

    override var prefersStatusBarHidden: Bool { false }
    override var preferredStatusBarStyle: UIStatusBarStyle { .lightContent }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = UIColor(red: 0.086, green: 0.067, blue: 0.114, alpha: 1)
        let cfg = WKWebViewConfiguration()
        cfg.userContentController.add(WeakHandler(self), name: "llm")
        cfg.allowsInlineMediaPlayback = true
        cfg.mediaTypesRequiringUserActionForPlayback = []
        web = WKWebView(frame: view.bounds, configuration: cfg)
        web.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        web.isOpaque = false
        web.backgroundColor = view.backgroundColor
        web.scrollView.backgroundColor = view.backgroundColor
        if #available(iOS 16.4, *) { web.isInspectable = true }
        view.addSubview(web)
        hud.translatesAutoresizingMaskIntoConstraints = false
        hud.numberOfLines = 0
        hud.textAlignment = .center
        hud.font = .systemFont(ofSize: 14, weight: .semibold)
        hud.textColor = .white
        hud.backgroundColor = UIColor.black.withAlphaComponent(0.78)
        hud.layer.cornerRadius = 10
        hud.clipsToBounds = true
        hud.isHidden = true
        view.addSubview(hud)
        NSLayoutConstraint.activate([
            hud.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 8),
            hud.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            hud.widthAnchor.constraint(lessThanOrEqualTo: view.widthAnchor, constant: -24)
        ])
        if let url = Bundle.main.url(forResource: "index", withExtension: "html", subdirectory: "web") {
            web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
        }
        engine.onProgress = { [weak self] p in
            DispatchQueue.main.async { self?.progress = p; self?.sendStatus(); self?.showHud("Загрузка модели: \(Int(p * 100))%", sticky: true) }
        }
    }

    // MARK: - Мост JS <-> Swift

    func userContentController(_ c: WKUserContentController, didReceive m: WKScriptMessage) {
        guard let body = m.body as? [String: Any], let cmd = body["cmd"] as? String else { return }
        switch cmd {
        case "status":
            sendStatus()
            autoLoad()
        case "load":
            if let name = body["name"] as? String { loadModel(named: name) }
        case "pick":
            pickModel(copy: false)
        case "pickCopy":
            pickModel(copy: true)
        case "chat":
            chat(body)
        case "embed":
            embed(body)
        default:
            break
        }
    }

    private func js(_ code: String) {
        DispatchQueue.main.async { [weak self] in self?.web.evaluateJavaScript(code, completionHandler: nil) }
    }

    private func jsonString(_ obj: Any) -> String {
        guard let d = try? JSONSerialization.data(withJSONObject: obj, options: [.fragmentsAllowed]),
              let s = String(data: d, encoding: .utf8) else { return "null" }
        return s
    }

    private var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    private func modelFiles() -> [[String: Any]] {
        var out: [[String: Any]] = []
        let items = (try? FileManager.default.contentsOfDirectory(at: documents, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        for u in items where u.pathExtension.lowercased() == "gguf" {
            let size = (try? u.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            out.append(["name": u.lastPathComponent, "size": size])
        }
        if let u = resolveBookmark(), !out.contains(where: { ($0["name"] as? String) == u.lastPathComponent }) {
            let size = (try? u.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            out.append(["name": u.lastPathComponent, "size": size])
        }
        return out.sorted { ($0["name"] as? String ?? "") < ($1["name"] as? String ?? "") }
    }

    private func sendStatus(justLoaded: Bool = false) {
        var st: [String: Any] = ["type": "status", "models": modelFiles()]
        if let n = engine.loadedName { st["loaded"] = n }
        if let l = loading { st["loading"] = l }
        if let p = progress, loading != nil { st["progress"] = p }
        if let e = lastError { st["error"] = e }
        if justLoaded { st["justLoaded"] = true }
        js("window.__llm && window.__llm.event(\(jsonString(st)))")
    }

    // MARK: - Модель

    private func resolveBookmark() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: "modelBookmark") else { return nil }
        var stale = false
        return try? URL(resolvingBookmarkData: data, options: [], relativeTo: nil, bookmarkDataIsStale: &stale)
    }

    private func autoLoad() {
        guard !autoTried, engine.loadedName == nil, loading == nil else { return }
        autoTried = true
        if let last = UserDefaults.standard.string(forKey: "lastModel") { loadModel(named: last) }
    }

    private func loadModel(named name: String, direct: URL? = nil) {
        guard loading == nil else { showHud("Модель уже загружается, подожди", sticky: false); return }
        var url = direct ?? documents.appendingPathComponent(name)
        if !FileManager.default.fileExists(atPath: url.path), let b = resolveBookmark(), b.lastPathComponent == name {
            if b.startAccessingSecurityScopedResource() { scopedURL = b }
            url = b
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            lastError = "Файл «\(name)» не найден."
            showHud(lastError!, sticky: true)
            sendStatus(); return
        }
        showHud("Загружаю \(name)…", sticky: true)
        loading = name; progress = 0; lastError = nil
        sendStatus()
        queue.async { [weak self] in
            guard let self = self else { return }
            do {
                try self.engine.load(path: url.path, name: name)
                UserDefaults.standard.set(name, forKey: "lastModel")
                DispatchQueue.main.async { self.loading = nil; self.sendStatus(justLoaded: true); self.showHud("Модель загружена: \(name)", sticky: false) }
            } catch {
                DispatchQueue.main.async {
                    self.loading = nil
                    self.lastError = error.localizedDescription
                    self.sendStatus()
                    self.showHud("Ошибка: \(error.localizedDescription)", sticky: true)
                }
            }
        }
    }

    private func pickModel(copy: Bool) {
        pickCopy = copy
        showHud(copy ? "Выбери файл модели: он будет скопирован в приложение" : "Выбери файл модели .gguf", sticky: false)
        var types: [UTType] = [.data, .item]
        if let g = UTType("io.github.flashik1.gguf") { types.insert(g, at: 0) }
        if let g = UTType(filenameExtension: "gguf") { types.insert(g, at: 0) }
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: types, asCopy: copy)
        picker.delegate = self
        picker.allowsMultipleSelection = false
        present(picker, animated: true)
    }

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let u = urls.first else { showHud("Файл не получен от системы. Попробуй «Скопировать модель в приложение».", sticky: false); return }
        showHud("Получен файл: \(u.lastPathComponent)", sticky: false)
        if pickCopy {
            let dest = documents.appendingPathComponent(u.lastPathComponent)
            if u.standardizedFileURL != dest.standardizedFileURL {
                try? FileManager.default.removeItem(at: dest)
                do { try FileManager.default.moveItem(at: u, to: dest) } catch {
                    do { try FileManager.default.copyItem(at: u, to: dest) } catch { showHud("Не удалось скопировать: \(error.localizedDescription)", sticky: true); return }
                }
            }
            openModel(dest)
        } else {
            openModel(u)
        }
    }

    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        note("Выбор модели отменён.")
        showHud("Выбор отменён", sticky: false)
    }

    /// Открыть модель из выбора файла или из «Поделиться» в приложении «Файлы».
    func openModel(_ u: URL) {
        var url = u
        // Файл, присланный копией, лежит во входящих: переносим его в папку приложения
        if url.path.contains("/Inbox/") {
            let dest = documents.appendingPathComponent(url.lastPathComponent)
            try? FileManager.default.removeItem(at: dest)
            if (try? FileManager.default.moveItem(at: url, to: dest)) != nil { url = dest }
        }
        let ok = url.startAccessingSecurityScopedResource()
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? -1
        let readable = FileManager.default.isReadableFile(atPath: url.path)
        if !readable {
            showHud("Нет доступа к файлу \(url.lastPathComponent). Нажми «Скопировать модель в приложение».", sticky: true)
            note("Нет доступа к файлу \(url.lastPathComponent) (доступ: \(ok ? "да" : "нет")). Попробуй кнопку «Скопировать модель в приложение».")
            return
        }
        showHud("Файл \(url.lastPathComponent), \(String(format: "%.1f", Double(size) / 1e9)) ГБ", sticky: false)
        if !url.path.hasPrefix(documents.path),
           let data = try? url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(data, forKey: "modelBookmark")
        }
        if ok { scopedURL = url }
        guard url.pathExtension.lowercased() == "gguf" else {
            note("Это не файл модели .gguf: \(url.lastPathComponent)")
            return
        }
        note("Выбран файл \(url.lastPathComponent). Загружаю модель, большая модель грузится до минуты…")
        loadModel(named: url.lastPathComponent, direct: url)
    }

    private func showHud(_ text: String, sticky: Bool) {
        let work = { [weak self] in
            guard let self = self else { return }
            self.hud.text = "  \(text)  "
            self.hud.isHidden = false
            self.view.bringSubviewToFront(self.hud)
            self.hudHide?.cancel()
            let w = DispatchWorkItem { [weak self] in self?.hud.isHidden = true }
            self.hudHide = w
            DispatchQueue.main.asyncAfter(deadline: .now() + (sticky ? 20 : 5), execute: w)
        }
        if Thread.isMainThread { work() } else { DispatchQueue.main.async(execute: work) }
    }

    private func note(_ text: String) {
        js("window.__llm && window.__llm.event(\(jsonString(["type": "note", "text": text])))")
    }

    // MARK: - Смысловые отпечатки

    private func embed(_ body: [String: Any]) {
        guard let id = body["id"] as? Int else { return }
        let texts = body["texts"] as? [String] ?? []
        queue.async { [weak self] in
            guard let self = self else { return }
            do {
                let vs = try self.engine.embed(texts)
                let rounded: [[Double]] = vs.map { $0.map { (Double($0) * 10000).rounded() / 10000 } }
                self.js("window.__llm.emb(\(id), \(self.jsonString(rounded)))")
            } catch {
                self.js("window.__llm.emb(\(id), null)")
            }
        }
    }

    // MARK: - Генерация

    private func chat(_ body: [String: Any]) {
        guard let id = body["id"] as? Int else { return }
        let raw = body["messages"] as? [[String: Any]] ?? []
        let msgs: [(String, String)] = raw.map { (($0["role"] as? String) ?? "user", ($0["content"] as? String) ?? "") }
        let maxTokens = (body["max_tokens"] as? Int) ?? 200
        let temp = Float((body["temperature"] as? Double) ?? 0.7)
        let json = (body["json"] as? Bool) ?? false
        queue.async { [weak self] in
            guard let self = self else { return }
            do {
                let tps = try self.engine.generate(messages: msgs, maxTokens: maxTokens, temp: temp, json: json,
                                                   cancel: { false },
                                                   emit: { piece in self.js("window.__llm.chunk(\(id), \(self.jsonString(piece)))") })
                self.js("window.__llm.done(\(id), {tps: \(tps)})")
            } catch {
                self.js("window.__llm.error(\(id), \(self.jsonString(error.localizedDescription)))")
            }
        }
    }
}

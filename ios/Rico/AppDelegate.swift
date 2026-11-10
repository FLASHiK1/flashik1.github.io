import UIKit

@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Папка приложения видна в «Файлах» только если в ней есть файл: кладём подсказку
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let readme = docs.appendingPathComponent("Положи сюда модель .gguf.txt")
        if !FileManager.default.fileExists(atPath: readme.path) {
            let text = "Перемести сюда файл модели (.gguf), затем в игре: Чат → «Нейросеть» → «Обновить список» → «Загрузить».\n"
            try? text.write(to: readme, atomically: true, encoding: .utf8)
        }
        let w = UIWindow(frame: UIScreen.main.bounds)
        w.rootViewController = GameViewController()
        w.makeKeyAndVisible()
        window = w
        return true
    }

    /// «Поделиться» → «Рико» или «Открыть в…» из приложения «Файлы»
    func application(_ app: UIApplication, open url: URL, options: [UIApplication.OpenURLOptionsKey: Any] = [:]) -> Bool {
        (window?.rootViewController as? GameViewController)?.openModel(url)
        return true
    }
}

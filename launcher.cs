// 华矩 EL 裁剪工具 - 启动器
// 编译: C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe /target:winexe /reference:System.Web.Extensions.dll /win32icon:app\logo.ico /out:EL组件裁剪服务端.exe launcher.cs

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using System.Security.Cryptography;
using System.Web.Script.Serialization;

public class Launcher : Form {
    static string Root { get { return Path.GetDirectoryName(Application.ExecutablePath); } }
    static string PyPath { get { return Path.Combine(Root, "runtime", "python.exe"); } }
    static string AppPath { get { return Path.Combine(Root, "app", "app.py"); } }
    static string UsersFile { get { return Path.Combine(Root, "users.json"); } }
    static string PrefFile { get { return Path.Combine(Root, ".launcher_prefs"); } }
    static int Port = 15789;
    static string Url { get { return "http://127.0.0.1:" + Port; } }

    Process proc;
    System.Windows.Forms.Timer statusTimer, logTimer;
    List<string> logBuf = new List<string>();
    object logLock = new object();
    bool forceOpen = false, autoOpened = false;

    Label lblMain, lblBe, lblFe;
    Button btnStart, btnStop, btnOpen;
    CheckBox chkAutoStart, chkAutoOpen;
    RichTextBox rtb;
    ListView userList;
    Label lblUserCount, lblUserMsg, lblLanIp, lblUrl, lblBottom;
    TextBox portInput;

    static JavaScriptSerializer Json = new JavaScriptSerializer();

    // ========== User Management ==========

    static string NewSalt() {
        var b = new byte[16];
        using (var rng = RandomNumberGenerator.Create()) rng.GetBytes(b);
        return Convert.ToBase64String(b);
    }
    static string HashPwd(string pwd, string salt) {
        using (var sha = SHA256.Create())
            return Convert.ToBase64String(sha.ComputeHash(Encoding.UTF8.GetBytes(salt + pwd)));
    }

    static List<Dictionary<string, object>> LoadUsers() {
        try {
            if (File.Exists(UsersFile)) {
                var raw = Json.Deserialize<Dictionary<string, object>>(File.ReadAllText(UsersFile, Encoding.UTF8));
                if (raw != null && raw.ContainsKey("users")) {
                    var list = raw["users"] as ArrayList;
                    if (list != null) {
                        var result = new List<Dictionary<string, object>>();
                        foreach (var item in list) {
                            var dict = item as Dictionary<string, object>;
                            if (dict != null) result.Add(dict);
                        }
                        return result;
                    }
                }
            }
        } catch (Exception) { }
        return new List<Dictionary<string, object>>();
    }

    static void SaveUsers(List<Dictionary<string, object>> users) {
        try {
            var wrapped = new Dictionary<string, object> { { "users", users } };
            File.WriteAllText(UsersFile, PrettyJson(Json.Serialize(wrapped)), new UTF8Encoding(false));
        } catch (Exception) { }
    }

    static string PrettyJson(string json) {
        int indent = 0; var sb = new StringBuilder(); bool inStr = false;
        foreach (char c in json) {
            if (c == '"') inStr = !inStr;
            if (!inStr && (c == '{' || c == '[')) { sb.Append(c); sb.AppendLine(); indent += 2; sb.Append(new string(' ', indent)); }
            else if (!inStr && (c == '}' || c == ']')) { sb.AppendLine(); indent -= 2; sb.Append(new string(' ', indent)); sb.Append(c); }
            else if (!inStr && c == ',') { sb.Append(c); sb.AppendLine(); sb.Append(new string(' ', indent)); }
            else if (!inStr && c == ':') sb.Append(": ");
            else sb.Append(c);
        }
        return sb.ToString();
    }

    static string G(Dictionary<string, object> d, string k, string def = "") {
        return d.ContainsKey(k) ? (d[k] == null ? def : d[k].ToString()) : def;
    }

    // ========== Network ==========

    static string GetLanIp() {
        try {
            foreach (var addr in Dns.GetHostEntry(Dns.GetHostName()).AddressList)
                if (addr.AddressFamily == AddressFamily.InterNetwork && !IPAddress.IsLoopback(addr))
                    return addr.ToString();
        } catch { }
        return "127.0.0.1";
    }

    static bool TestBackend() {
        try { using (var c = new TcpClient()) { var ar = c.BeginConnect("127.0.0.1", Port, null, null); return ar.AsyncWaitHandle.WaitOne(350); } }
        catch { return false; }
    }

    static bool TestFrontend() {
        try { var req = WebRequest.Create(Url + "/health"); req.Timeout = 700; using (var res = req.GetResponse()) using (var sr = new StreamReader(res.GetResponseStream())) return sr.ReadToEnd().IndexOf("frontend") >= 0; }
        catch { return false; }
    }

    // ========== Constructor ==========

    public Launcher() {
        Text = "EL组件裁剪服务端";
        Size = new Size(720, 700);
        MinimumSize = new Size(660, 620);
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.FromArgb(246, 248, 252);
        Font = new Font("Microsoft YaHei UI", 9);

        try { Icon = new Icon(Path.Combine(Root, "app", "logo.ico")); } catch { }
        BuildUI();
        LoadPrefs();
        if (chkAutoStart.Checked) BeginInvoke(new Action(StartService));

        statusTimer = new System.Windows.Forms.Timer { Interval = 1200 };
        statusTimer.Tick += (s, e) => {
            bool be = TestBackend(), fe = be && TestFrontend();
            SetStatus(be, fe);
            if (be && fe && !autoOpened && (chkAutoOpen.Checked || forceOpen)) {
                autoOpened = true; forceOpen = false;
                Process.Start(Url);
            }
        };
        statusTimer.Start();

        logTimer = new System.Windows.Forms.Timer { Interval = 150 };
        logTimer.Tick += (s, e) => { lock (logLock) { if (logBuf.Count > 0) { foreach (var l in logBuf) Log(l); logBuf.Clear(); } } };
        logTimer.Start();

        FormClosing += (s, e) => {
            statusTimer.Stop(); logTimer.Stop();
            if (TestBackend() && MessageBox.Show("检测到服务仍在运行。是否同时停止前端/后端本地服务？\n\n选择[是]：停止服务并关闭启动器\n选择[否]：仅关闭启动器，服务继续运行", "关闭启动器", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
                StopService();
        };
    }

    // ========== UI ==========

    void BuildUI() {
        var hdr = new Panel { Dock = DockStyle.Top, Height = 74, BackColor = Color.FromArgb(37, 99, 235) };
        MakeLabel(hdr, "华矩 EL 裁剪工具启动器", 22, 14, 360, 28, new Font("Microsoft YaHei UI", 15, FontStyle.Bold), Color.White);
        MakeLabel(hdr, "一键启动前端页面与本地后端服务，实时检查运行状态", 24, 44, 520, 18, new Font("Microsoft YaHei UI", 8), Color.FromArgb(219, 234, 254));
        Controls.Add(hdr);

        var sc = new Panel { Location = new Point(16, 86), Size = new Size(670, 108), BackColor = Color.White, BorderStyle = BorderStyle.FixedSingle };
        MakeLabel(sc, "运行状态", 14, 8, 100, 22, new Font("Microsoft YaHei UI", 10, FontStyle.Bold), Color.FromArgb(31, 41, 55));
        lblMain = MakeLabel(sc, "未启动", 14, 32, 200, 28, new Font("Microsoft YaHei UI", 15, FontStyle.Bold), Color.FromArgb(220, 38, 38));
        lblUrl = MakeLabel(sc, "本机地址: 127.0.0.1:" + Port, 14, 62, 260, 20, new Font("Microsoft YaHei UI", 9), Color.FromArgb(100, 116, 139));
        lblLanIp = MakeLabel(sc, "局域网: " + GetLanIp() + ":" + Port, 14, 84, 220, 20, new Font("Microsoft YaHei UI", 9), Color.FromArgb(100, 116, 139));
        MakeLabel(sc, "端口:", 240, 84, 34, 20, new Font("Microsoft YaHei UI", 9), Color.FromArgb(100, 116, 139));
        portInput = new TextBox { Text = Port.ToString(), Location = new Point(274, 82), Width = 60, Height = 22, Font = new Font("Microsoft YaHei UI", 9) };
        sc.Controls.Add(portInput);
        var btnApplyPort = new Button { Text = "应用", Location = new Point(340, 81), Width = 45, Height = 24, BackColor = Color.FromArgb(37, 99, 235), ForeColor = Color.White, FlatStyle = FlatStyle.Flat, Cursor = Cursors.Hand, Font = new Font("Microsoft YaHei UI", 9) };
        btnApplyPort.FlatAppearance.BorderSize = 0;
        btnApplyPort.Click += (s, e) => ApplyPort();
        sc.Controls.Add(btnApplyPort);
        lblBe = MakeLabel(sc, "后端：未运行", 460, 28, 200, 22, new Font("Microsoft YaHei UI", 10, FontStyle.Bold), Color.FromArgb(220, 38, 38));
        lblFe = MakeLabel(sc, "前端：未就绪", 460, 62, 200, 22, new Font("Microsoft YaHei UI", 10, FontStyle.Bold), Color.FromArgb(220, 38, 38));
        Controls.Add(sc);

        btnStart = MakeBtn("一键启动", 16, 206, 150, 40, Color.FromArgb(37, 99, 235), Color.White, StartService);
        btnStop = MakeBtn("停止服务", 182, 206, 150, 40, Color.FromArgb(220, 38, 38), Color.White, StopService);
        btnStop.Enabled = false;
        MakeBtn("重启服务", 348, 206, 150, 40, Color.FromArgb(217, 119, 6), Color.White, RestartService);
        btnOpen = MakeBtn("打开前端", 514, 206, 172, 40, Color.FromArgb(22, 163, 74), Color.White, () => Process.Start(Url));
        btnOpen.Enabled = false;

        chkAutoStart = new CheckBox { Text = "打开启动器后自动启动服务", Location = new Point(22, 256), AutoSize = true, BackColor = BackColor, ForeColor = Color.FromArgb(100, 116, 139) };
        chkAutoStart.CheckedChanged += (s, e) => SavePrefs();
        chkAutoOpen = new CheckBox { Text = "服务启动成功后自动打开前端页面", Location = new Point(250, 256), AutoSize = true, BackColor = BackColor, ForeColor = Color.FromArgb(100, 116, 139), Checked = false };
        chkAutoOpen.CheckedChanged += (s, e) => SavePrefs();
        Controls.Add(chkAutoStart); Controls.Add(chkAutoOpen);

        Controls.Add(new Label { Location = new Point(16, 282), Size = new Size(670, 1), BackColor = Color.FromArgb(226, 232, 240) });

        var uf = new Panel { Location = new Point(16, 290), Size = new Size(670, 26), BackColor = BackColor };
        MakeLabel(uf, "Web 账户管理", 0, 2, 130, 22, new Font("Microsoft YaHei UI", 10, FontStyle.Bold), Color.FromArgb(31, 41, 55));
        lblUserCount = MakeLabel(uf, "", 136, 4, 120, 18, new Font("Microsoft YaHei UI", 9), Color.FromArgb(148, 163, 184));
        SmallBtn(uf, "刷新", 620, 0, 50, 22, Color.FromArgb(219, 234, 254), Color.FromArgb(37, 99, 235), RefreshUsers);
        Controls.Add(uf);

        userList = new ListView { Location = new Point(16, 318), Size = new Size(670, 80), View = View.Details, FullRowSelect = true, HeaderStyle = ColumnHeaderStyle.Nonclickable, MultiSelect = false };
        userList.Columns.Add("用户名", 240); userList.Columns.Add("角色", 120); userList.Columns.Add("创建时间", 260);
        Controls.Add(userList);

        var ubf = new Panel { Location = new Point(16, 402), Size = new Size(670, 26), BackColor = BackColor };
        SmallBtn(ubf, "添加用户", 0, 0, 74, 24, Color.FromArgb(37, 99, 235), Color.White, AddUserDlg);
        SmallBtn(ubf, "删除用户", 80, 0, 74, 24, Color.FromArgb(220, 38, 38), Color.White, DelUser);
        SmallBtn(ubf, "修改密码", 160, 0, 74, 24, Color.FromArgb(217, 119, 6), Color.White, ChgPwdDlg);
        SmallBtn(ubf, "修改角色", 240, 0, 74, 24, Color.FromArgb(37, 99, 235), Color.White, ChgRoleDlg);
        lblUserMsg = MakeLabel(ubf, "", 510, 3, 160, 18, new Font("Microsoft YaHei UI", 9), Color.FromArgb(100, 116, 139));
        Controls.Add(ubf);

        Controls.Add(new Label { Location = new Point(16, 434), Size = new Size(670, 1), BackColor = Color.FromArgb(226, 232, 240) });

        MakeLabel(this, "运行日志", 18, 442, 120, 22, new Font("Microsoft YaHei UI", 10, FontStyle.Bold), Color.FromArgb(31, 41, 55));
        SmallBtn(null, "清空", 626, 440, 60, 22, Color.FromArgb(219, 234, 254), Color.FromArgb(37, 99, 235), () => rtb.Clear());
        rtb = new RichTextBox { Location = new Point(16, 466), Size = new Size(670, 130), Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Bottom,
                                BackColor = Color.FromArgb(15, 23, 42), ForeColor = Color.FromArgb(226, 232, 240), Font = new Font("Consolas", 9), ReadOnly = true, BorderStyle = BorderStyle.None, WordWrap = true };
        Controls.Add(rtb);

        lblBottom = new Label { Dock = DockStyle.Bottom, Height = 24,
            Text = "  端口：" + Port + "    路径：" + Root,
            ForeColor = Color.FromArgb(100, 116, 139), BackColor = Color.FromArgb(238, 242, 247),
            TextAlign = ContentAlignment.MiddleLeft, Font = new Font("Microsoft YaHei UI", 8) };
        Controls.Add(lblBottom);

        RefreshUsers();
    }

    Label MakeLabel(Control parent, string t, int x, int y, int w, int h, Font f, Color c) {
        var l = new Label { Text = t, Location = new Point(x, y), Size = new Size(w, h), Font = f, ForeColor = c, BackColor = parent.BackColor == Color.Empty ? Color.Transparent : parent.BackColor };
        if (parent.BackColor != Color.Transparent && parent.BackColor != Color.Empty) l.BackColor = parent.BackColor;
        parent.Controls.Add(l); return l;
    }
    Button MakeBtn(string t, int x, int y, int w, int h, Color bg, Color fg, Action act) {
        var b = new Button { Text = t, Location = new Point(x, y), Size = new Size(w, h), BackColor = bg, ForeColor = fg, FlatStyle = FlatStyle.Flat, Cursor = Cursors.Hand, Font = new Font("Microsoft YaHei UI", 10, FontStyle.Bold) };
        b.FlatAppearance.BorderSize = 0; b.Click += (s, e) => act(); Controls.Add(b); return b;
    }
    void SmallBtn(Panel parent, string t, int x, int y, int w, int h, Color bg, Color fg, Action act) {
        var b = new Button { Text = t, Location = new Point(x, y), Size = new Size(w, h), BackColor = bg, ForeColor = fg, FlatStyle = FlatStyle.Flat, Cursor = Cursors.Hand, Font = new Font("Microsoft YaHei UI", 9) };
        b.FlatAppearance.BorderSize = 0; b.Click += (s, e) => act();
        if (parent != null) parent.Controls.Add(b); else Controls.Add(b);
    }

    // ========== Status ==========

    void SetStatus(bool be, bool fe) {
        if (InvokeRequired) { BeginInvoke(new Action(() => SetStatus(be, fe))); return; }
        lblMain.Text = be ? "运行中" : "未启动";
        lblMain.ForeColor = be ? Color.FromArgb(22, 163, 74) : Color.FromArgb(220, 38, 38);
        lblBe.Text = be ? "后端：运行中" : "后端：未运行";
        lblBe.ForeColor = be ? Color.FromArgb(22, 163, 74) : Color.FromArgb(220, 38, 38);
        lblFe.Text = fe ? "前端：可访问" : (be ? "前端：检查中" : "前端：未就绪");
        lblFe.ForeColor = fe ? Color.FromArgb(22, 163, 74) : (be ? Color.FromArgb(217, 119, 6) : Color.FromArgb(220, 38, 38));
        btnStart.Enabled = !be; btnStop.Enabled = be; btnOpen.Enabled = be;
    }

    void SetStarting() {
        lblMain.Text = "启动中"; lblMain.ForeColor = Color.FromArgb(217, 119, 6);
        lblBe.Text = "后端：启动中"; lblBe.ForeColor = Color.FromArgb(217, 119, 6);
        lblFe.Text = "前端：等待后端"; lblFe.ForeColor = Color.FromArgb(217, 119, 6);
        btnStart.Enabled = false;
    }

    // ========== Log ==========

    void Log(string text) {
        if (rtb.IsDisposed) return;
        rtb.Invoke(new Action(() => {
            rtb.SelectionStart = rtb.TextLength; rtb.SelectionLength = 0;
            var lo = text.ToLower();
            rtb.SelectionColor = (lo.Contains("error") || lo.Contains("traceback") || lo.Contains("exception") || lo.Contains("fail"))
                ? Color.FromArgb(252, 165, 165)
                : (lo.Contains("running") || lo.Contains("started") || lo.Contains("ready") || lo.Contains("ok") || lo.Contains("http"))
                ? Color.FromArgb(125, 211, 252)
                : Color.FromArgb(134, 239, 172);
            rtb.AppendText(DateTime.Now.ToString("HH:mm:ss") + "  " + text + "\n");
            rtb.ScrollToCaret();
        }));
    }
    void QueueLog(string s) { lock (logLock) logBuf.Add(s); }

    // ========== Service Control ==========

    void StartService() {
        if (!File.Exists(PyPath)) { QueueLog("未找到 Python：" + PyPath); return; }
        if (!File.Exists(AppPath)) { QueueLog("未找到后端入口：" + AppPath); return; }
        if (TestBackend()) { QueueLog("服务已在运行，直接打开前端。"); Process.Start(Url); return; }
        autoOpened = false; forceOpen = true;
        SetStarting();
        QueueLog("启动本地服务：" + Url);
        var psi = new ProcessStartInfo {
            FileName = PyPath, Arguments = "\"" + AppPath + "\"", WorkingDirectory = Path.GetDirectoryName(AppPath),
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
        };
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";
        psi.EnvironmentVariables["EL_CROP_PORT"] = Port.ToString();
        proc = new Process { StartInfo = psi };
        proc.Start();
        new Thread(() => { try { string l; while ((l = proc.StandardOutput.ReadLine()) != null) QueueLog(l); } catch { } }) { IsBackground = true }.Start();
        new Thread(() => { try { string l; while ((l = proc.StandardError.ReadLine()) != null) QueueLog(l); } catch { } }) { IsBackground = true }.Start();
        new Thread(() => { try { proc.WaitForExit(); QueueLog("服务进程已退出，代码：" + proc.ExitCode); } catch { } }) { IsBackground = true }.Start();
    }

    void StopService() {
        var pids = GetPortPids();
        if (pids.Count == 0) { QueueLog("没有发现需要停止的服务进程。"); proc = null; return; }
        QueueLog("停止服务进程：" + string.Join(", ", pids));
        foreach (int pid in pids)
            try { Process.Start("taskkill", "/PID " + pid + " /T /F").WaitForExit(); } catch { }
        Thread.Sleep(500);
        proc = null; autoOpened = false; forceOpen = false;
    }

    void RestartService() { StopService(); Thread.Sleep(500); StartService(); }

    List<int> GetPortPids() {
        var pids = new List<int>();
        try {
            var p = Process.Start(new ProcessStartInfo("netstat", "-ano -p tcp") { UseShellExecute = false, RedirectStandardOutput = true, CreateNoWindow = true });
            foreach (var line in p.StandardOutput.ReadToEnd().Split('\n')) {
                if (line.Contains(":" + Port) && line.Contains("LISTENING")) {
                    var parts = line.Trim().Split(' ');
                    int pid;
                    if (int.TryParse(parts[parts.Length - 1], out pid) && pid != Process.GetCurrentProcess().Id && !pids.Contains(pid))
                        pids.Add(pid);
                }
            }
        } catch { }
        return pids;
    }

    // ========== User Management ==========

    void RefreshUsers() {
        userList.Items.Clear();
        var users = LoadUsers();
        foreach (var u in users) {
            var item = new ListViewItem(G(u, "username"));
            item.SubItems.Add(G(u, "role") == "admin" ? "管理员" : "普通用户");
            item.SubItems.Add(G(u, "created_at"));
            userList.Items.Add(item);
        }
        lblUserCount.Text = "共 " + users.Count + " 个用户";
    }

    void AddUserDlg() {
        var d = new Form { Text = "添加用户", Size = new Size(340, 230), FormBorderStyle = FormBorderStyle.FixedDialog, StartPosition = FormStartPosition.CenterParent, MinimizeBox = false, MaximizeBox = false };
        var fl = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, Padding = new Padding(16) };
        var inpU = new TextBox { Font = Font };
        var inpP = new TextBox { Font = Font, UseSystemPasswordChar = true };
        var msg = new Label { Height = 20, Font = new Font("Microsoft YaHei UI", 9), ForeColor = Color.FromArgb(220, 38, 38) };
        fl.Controls.Add(new Label { Text = "用户名" }); fl.Controls.Add(inpU);
        fl.Controls.Add(new Label { Text = "密码" }); fl.Controls.Add(inpP);
        fl.Controls.Add(new Label { Text = "角色" });
        var inpR = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Font = Font };
        inpR.Items.Add("普通用户"); inpR.Items.Add("管理员"); inpR.SelectedIndex = 0;
        fl.Controls.Add(inpR);
        fl.Controls.Add(msg);
        var bp = new Panel { Dock = DockStyle.Bottom, Height = 44, BackColor = Color.Transparent };
        var btnOk = new Button { Text = "确定", BackColor = Color.FromArgb(37, 99, 235), ForeColor = Color.White, FlatStyle = FlatStyle.Flat, Width = 80, Height = 28, Cursor = Cursors.Hand, Anchor = AnchorStyles.Top | AnchorStyles.Right };
        btnOk.FlatAppearance.BorderSize = 0;
        btnOk.Click += (s, e) => {
            var users = LoadUsers();
            foreach (var u in users) if (G(u, "username") == inpU.Text.Trim()) { msg.Text = "用户名已存在"; return; }
            if (inpU.Text.Trim().Length < 2) { msg.Text = "用户名至少2个字符"; return; }
            if (inpP.Text.Length < 4) { msg.Text = "密码至少4个字符"; return; }
            var salt = NewSalt();
            users.Add(new Dictionary<string,object>{ {"username",inpU.Text.Trim()},{"password_hash",HashPwd(inpP.Text,salt)},{"salt",salt},{"role",inpR.SelectedIndex==1?"admin":"user"},{"created_at",DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")} });
            SaveUsers(users); RefreshUsers(); QueueLog("添加用户：" + inpU.Text.Trim());
            d.Close();
        };
        bp.Controls.Add(btnOk);
        d.Controls.Add(fl); d.Controls.Add(bp);
        d.Shown += (s, e) => { btnOk.Location = new Point(bp.ClientSize.Width - 96, 8); };
        d.ShowDialog(this);
    }

    void DelUser() {
        if (userList.SelectedItems.Count == 0) { lblUserMsg.Text = "请先在列表中选择用户"; return; }
        var name = userList.SelectedItems[0].Text;
        if (name == "admin") { lblUserMsg.Text = "不能删除管理员"; return; }
        if (MessageBox.Show("确定删除用户 \"" + name + "\" 吗？", "确认", MessageBoxButtons.YesNo) != DialogResult.Yes) return;
        var users = LoadUsers();
        users.RemoveAll(u => G(u, "username") == name);
        SaveUsers(users); RefreshUsers(); QueueLog("删除用户：" + name); lblUserMsg.Text = "已删除";
    }

    void ChgPwdDlg() {
        if (userList.SelectedItems.Count == 0) { lblUserMsg.Text = "请先在列表中选择用户"; return; }
        var name = userList.SelectedItems[0].Text;
        var d = new Form { Text = "修改密码 - " + name, Size = new Size(300, 160), FormBorderStyle = FormBorderStyle.FixedDialog, StartPosition = FormStartPosition.CenterParent, MinimizeBox = false, MaximizeBox = false };
        var fl = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, Padding = new Padding(16) };
        var inpP = new TextBox { Font = Font, UseSystemPasswordChar = true };
        var msg = new Label { Height = 20, Font = new Font("Microsoft YaHei UI", 9), ForeColor = Color.FromArgb(220, 38, 38) };
        fl.Controls.Add(new Label { Text = "新密码（至少4位）" }); fl.Controls.Add(inpP);
        fl.Controls.Add(msg);
        var bp = new Panel { Dock = DockStyle.Bottom, Height = 44, BackColor = Color.Transparent };
        var btnOk = new Button { Text = "确定", BackColor = Color.FromArgb(37, 99, 235), ForeColor = Color.White, FlatStyle = FlatStyle.Flat, Width = 80, Height = 28, Cursor = Cursors.Hand, Anchor = AnchorStyles.Top | AnchorStyles.Right };
        btnOk.FlatAppearance.BorderSize = 0;
        btnOk.Click += (s, e) => {
            if (inpP.Text.Length < 4) { msg.Text = "密码至少4个字符"; return; }
            var users = LoadUsers();
            foreach (var u in users) {
                if (G(u, "username") == name) {
                    u["salt"] = NewSalt(); u["password_hash"] = HashPwd(inpP.Text, u["salt"].ToString());
                    SaveUsers(users); RefreshUsers(); QueueLog("修改密码：" + name); lblUserMsg.Text = "密码已修改"; d.Close(); return;
                }
            }
            msg.Text = "用户不存在";
        };
        bp.Controls.Add(btnOk);
        d.Controls.Add(fl); d.Controls.Add(bp);
        d.Shown += (s, e) => { btnOk.Location = new Point(bp.ClientSize.Width - 96, 8); };
        d.ShowDialog(this);
    }

    void ChgRoleDlg() {
        if (userList.SelectedItems.Count == 0) { lblUserMsg.Text = "请先在列表中选择用户"; return; }
        var name = userList.SelectedItems[0].Text;
        if (name == "admin") { lblUserMsg.Text = "不能修改管理员角色"; return; }
        var d = new Form { Text = "修改角色 - " + name, Size = new Size(300, 160), FormBorderStyle = FormBorderStyle.FixedDialog, StartPosition = FormStartPosition.CenterParent, MinimizeBox = false, MaximizeBox = false };
        var fl = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, Padding = new Padding(16) };
        var msg = new Label { Height = 20, Font = new Font("Microsoft YaHei UI", 9), ForeColor = Color.FromArgb(220, 38, 38) };
        var inpR = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Font = Font };
        inpR.Items.Add("普通用户"); inpR.Items.Add("管理员");
        var users = LoadUsers();
        foreach (var u in users) {
            if (G(u, "username") == name) { inpR.SelectedIndex = G(u, "role") == "admin" ? 1 : 0; break; }
        }
        fl.Controls.Add(new Label { Text = "新角色" }); fl.Controls.Add(inpR);
        fl.Controls.Add(msg);
        var bp = new Panel { Dock = DockStyle.Bottom, Height = 44, BackColor = Color.Transparent };
        var btnOk = new Button { Text = "确定", BackColor = Color.FromArgb(37, 99, 235), ForeColor = Color.White, FlatStyle = FlatStyle.Flat, Width = 80, Height = 28, Cursor = Cursors.Hand, Anchor = AnchorStyles.Top | AnchorStyles.Right };
        btnOk.FlatAppearance.BorderSize = 0;
        btnOk.Click += (s, e) => {
            var ulist = LoadUsers();
            foreach (var u in ulist) {
                if (G(u, "username") == name) {
                    u["role"] = inpR.SelectedIndex == 1 ? "admin" : "user";
                    SaveUsers(ulist); RefreshUsers(); QueueLog("修改角色：" + name + " -> " + u["role"]); lblUserMsg.Text = "角色已修改"; d.Close(); return;
                }
            }
            msg.Text = "用户不存在";
        };
        bp.Controls.Add(btnOk);
        d.Controls.Add(fl); d.Controls.Add(bp);
        d.Shown += (s, e) => { btnOk.Location = new Point(bp.ClientSize.Width - 96, 8); };
        d.ShowDialog(this);
    }

    // ========== Preferences ==========

    void SavePrefs() {
        try { File.WriteAllText(PrefFile, "auto_start=" + (chkAutoStart.Checked ? "1" : "0") + "\nauto_open=" + (chkAutoOpen.Checked ? "1" : "0") + "\nport=" + Port, Encoding.UTF8); } catch { }
    }
    void LoadPrefs() {
        try { if (File.Exists(PrefFile)) foreach (var l in File.ReadAllLines(PrefFile)) { if (l == "auto_start=1") chkAutoStart.Checked = true; if (l == "auto_open=1") chkAutoOpen.Checked = true; if (l.StartsWith("port=")) { int p; if (int.TryParse(l.Substring(5), out p) && p >= 1 && p <= 65535) { Port = p; if (portInput != null) portInput.Text = p.ToString(); } } } } catch { }
        UpdatePortDisplay();
    }

    void ApplyPort() {
        int newPort;
        if (portInput.Text.Trim() == Port.ToString()) return;
        if (int.TryParse(portInput.Text.Trim(), out newPort) && newPort >= 1 && newPort <= 65535) {
            Port = newPort;
            UpdatePortDisplay();
            SavePrefs();
            if (TestBackend() && MessageBox.Show("端口已更改，是否重启服务？", "端口更改", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
                RestartService();
        } else {
            MessageBox.Show("请输入有效端口号 (1-65535)", "端口无效", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            portInput.Text = Port.ToString();
        }
    }

    void UpdatePortDisplay() {
        if (lblLanIp != null) lblLanIp.Text = "局域网: " + GetLanIp() + ":" + Port;
        if (lblUrl != null) lblUrl.Text = "本机地址: 127.0.0.1:" + Port;
        if (lblBottom != null) lblBottom.Text = "  端口：" + Port + "    路径：" + Root;
    }

    // ========== Entry ==========

    static bool CheckPassword() {
        var d = new Form { Text = "启动验证", Size = new Size(360, 220), FormBorderStyle = FormBorderStyle.FixedDialog, StartPosition = FormStartPosition.CenterScreen, MinimizeBox = false, MaximizeBox = false, TopMost = true };
        var lbl = new Label { Text = "请输入启动密码", Font = new Font("Microsoft YaHei UI", 11), ForeColor = Color.FromArgb(31, 41, 55), Location = new Point(24, 28), AutoSize = true };
        var inp = new TextBox { Font = new Font("Microsoft YaHei UI", 13), UseSystemPasswordChar = true, Location = new Point(24, 56), Width = 300, Height = 32 };
        var msg = new Label { Font = new Font("Microsoft YaHei UI", 9), ForeColor = Color.Red, Location = new Point(26, 96), AutoSize = true, Height = 20 };
        var btn = new Button { Text = "确定", BackColor = Color.FromArgb(37, 99, 235), ForeColor = Color.White, FlatStyle = FlatStyle.Flat, Width = 80, Height = 32, Cursor = Cursors.Hand, Location = new Point(244, 130) };
        btn.FlatAppearance.BorderSize = 0;
        bool ok = false;
        var pwd = "";
        foreach (int x in new int[] { 0x3d, 0x3f, 0x3f, 0x36 })
            pwd += (char)(x ^ 0x55);
        btn.Click += (s, e) => {
            if (inp.Text == pwd) { ok = true; d.Close(); }
            else { msg.Text = "密码错误"; inp.SelectAll(); inp.Focus(); }
        };
        inp.KeyDown += (s, e) => { if (e.KeyCode == Keys.Enter) btn.PerformClick(); };
        d.Controls.Add(lbl); d.Controls.Add(inp); d.Controls.Add(msg); d.Controls.Add(btn);
        d.ActiveControl = inp;
        d.ShowDialog();
        return ok;
    }

    [STAThread]
    static void Main() {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        if (!CheckPassword()) return;
        Application.Run(new Launcher());
    }
}

static class Ext {
    public static void RemoveAll<T>(this List<T> list, Func<T, bool> pred) {
        for (int i = list.Count - 1; i >= 0; i--) if (pred(list[i])) list.RemoveAt(i);
    }
}

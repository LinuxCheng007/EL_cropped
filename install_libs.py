import sys, os, ssl, urllib.request, zipfile, io, subprocess

ROOT   = os.path.dirname(os.path.abspath(__file__))
PYDIR  = os.path.join(ROOT, 'runtime')
PY     = os.path.join(PYDIR, 'python.exe')
TARGET = os.path.join(PYDIR, 'Lib', 'site-packages')

print()
print('=' * 52)
print('  EL Tool - Library Installer')
print('  Target:', TARGET)
print('=' * 52)
print()

for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy']:
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'
print('[0] Proxy cleared.')

os.makedirs(TARGET, exist_ok=True)
print('[1] Target folder ready.')

print('[2] Fixing pth file...')
fixed = False
for fname in os.listdir(PYDIR):
    if fname.startswith('python3') and fname.endswith('.pth'):
        fpath = os.path.join(PYDIR, fname)
        txt = open(fpath, 'r', errors='ignore').read()
        new = txt
        if 'Lib\\site-packages' not in new:
            new += '\nLib\\site-packages\n'
        new = new.replace('#import site', 'import site')
        if 'import site' not in new:
            new += '\nimport site\n'
        if new != txt:
            open(fpath, 'w').write(new)
            print('  Fixed:', fname)
        else:
            print('  OK:', fname)
        fixed = True
if not fixed:
    pth = os.path.join(PYDIR, 'python38._pth')
    open(pth, 'w').write('python38.zip\n.\nLib\\site-packages\nimport site\n')
    print('  Created pth file.')

def dl(url, label):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ctx)
    )
    opener.addheaders = [('User-Agent', 'Python')]
    print('  Downloading', label, '...', end='', flush=True)
    with opener.open(url, timeout=120) as r:
        data = r.read()
    print(' %d KB' % (len(data)//1024))
    return data

print()
print('[3] Installing pip...')
PIP_URL = ('https://files.pythonhosted.org/packages/ef/71/'
           'c6b3e2db2861c65e3a35a04264e0d7d67b56e3e2ddd9d2fef6e27d96b07c/'
           'pip-24.0-py3-none-any.whl')
pip_ok = False
try:
    data = dl(PIP_URL, 'pip-24.0')
    if len(data) < 10000:
        raise Exception('File too small: %d bytes' % len(data))
    zf = zipfile.ZipFile(io.BytesIO(data))
    zf.extractall(TARGET)
    print('  Extracted %d files' % len(zf.namelist()))
    pip_ok = True
except Exception as e:
    print('  WHL failed:', e)

if not pip_ok:
    try:
        data2 = dl('https://bootstrap.pypa.io/pip/3.8/get-pip.py', 'get-pip.py')
        gp = os.path.join(PYDIR, 'get-pip.py')
        open(gp, 'wb').write(data2)
        subprocess.run([PY, gp, '--target', TARGET,
                        '--trusted-host', 'pypi.org',
                        '--trusted-host', 'files.pythonhosted.org'], check=True)
        os.remove(gp)
        pip_ok = True
    except Exception as e2:
        print('  [ERROR]', e2)
        input('Press Enter...')
        sys.exit(1)

env = {**os.environ, 'PYTHONPATH': TARGET,
       'HTTP_PROXY': '', 'HTTPS_PROXY': '', 'http_proxy': '', 'https_proxy': ''}

r = subprocess.run([PY, '-c',
    'import sys; sys.path.insert(0,r"' + TARGET + '"); '
    'import pip; print("pip", pip.__version__)'],
    capture_output=True, text=True, env=env)
if r.returncode != 0:
    print('[ERROR] pip not working:', r.stderr[:300])
    input('Press Enter...')
    sys.exit(1)
print('  pip OK:', r.stdout.strip())
print()

PKGS = ['numpy', 'flask', 'opencv-python']
print('[4] Installing packages (3-8 min)...')
print()
for pkg in PKGS:
    print('  >', pkg)
    r = subprocess.run(
        [PY, '-m', 'pip', 'install', pkg,
         '--target', TARGET,
         '--trusted-host', 'pypi.org',
         '--trusted-host', 'files.pythonhosted.org',
         '--trusted-host', 'pypi.python.org',
         '--proxy', '',
         '--no-warn-script-location'],
        env=env)
    if r.returncode != 0:
        subprocess.run(
            [PY, '-m', 'pip', 'install', pkg,
             '--target', TARGET,
             '--trusted-host', 'pypi.org',
             '--trusted-host', 'files.pythonhosted.org',
             '--no-warn-script-location'],
            env=env)
    print()

print('[5] Verifying...')
verify_code = (
    'import sys; sys.path.insert(0,r"' + TARGET + '"); '
    'import numpy,cv2,flask; '
    'print("numpy:", numpy.__version__); '
    'print("opencv:", cv2.__version__); '
    'print("flask:", flask.__version__); '
    'print("ALL OK")'
)
r = subprocess.run([PY, '-c', verify_code], capture_output=True, text=True, env=env)
print(r.stdout)
if r.returncode != 0 or 'ALL OK' not in r.stdout:
    print('[FAIL]', r.stderr[-600:])
    input('Press Enter...')
    sys.exit(1)

print()
print('=' * 52)
print('  SUCCESS! Run 3_start_tool.bat to launch.')
print('=' * 52)
print()
input('Press Enter to exit...')
